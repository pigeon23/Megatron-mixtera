# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from functools import partial
from typing import List, Optional, Tuple
from megatron.core import parallel_state
from megatron.training import inprocess_restart
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from model_provider import model_provider
from gpt_builders import gpt_builder

from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids

from typing import Any, Optional
import os
import time

from mixtera.torch import MixteraTorchDataset
from mixtera.core.client import MixteraClient, QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.query import Query
from mixtera.core.query.mixture import StaticMixture, MixtureKey
from mixtera.core.query.mixture.dynamic_mixture import DynamicMixture
from mixtera.core.algo.ado.ado import AdoDynamicMixing
from mixtera.utils.feedback import handle_mixtera_feedback
from mixtera.core.datacollection.datasets import JSONLDataset

from pathlib import Path

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()
USING_KEYIDS = False

def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage):
        next(data_iterator)
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)
    
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    
    # print(f"Rank {torch.distributed.get_rank()} has {batch.keys()}", flush=True)
    # if torch.distributed.get_rank() == 0:
    #     print("*****************************************")

    return batch.values()
 

# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


class PerDomainLoss(torch.nn.Module):
    def __init__(self, initial_num_domains=32, device=None):
        super().__init__()
        self._default_domains = initial_num_domains
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize losses_tensor and counts_tensor
        self.losses_tensor = torch.zeros(self._default_domains, dtype=torch.float32, device=self.device)
        self.counts_tensor = torch.zeros(self._default_domains, dtype=torch.int64, device=self.device)
        # Initialize max_domain_id as a tensor
        self.max_domain_id = torch.tensor(self._default_domains - 1, dtype=torch.int32, device=self.device)
        self.has_per_domain_loss = False

    def forward(self, loss, key_ids=None):
        loss = loss.reshape(-1)
        if key_ids is not None:
            with torch.no_grad():
                loss = loss.to(torch.float32)
                self.has_per_domain_loss = True

                # Flatten key_ids
                key_ids = key_ids.reshape(-1)

                # Get maximum domain ID in the batch
                batch_max_domain_id = key_ids.max()

                # Update max_domain_id if necessary
                if batch_max_domain_id > self.max_domain_id:
                    self.max_domain_id = batch_max_domain_id
                    self._default_domains = self.max_domain_id + 1

                num_domains = self.max_domain_id + 1

                # Ensure tensors are large enough
                if num_domains > self.losses_tensor.size(0):
                    extra_size = num_domains - self.losses_tensor.size(0)
                    self.losses_tensor = torch.cat([
                        self.losses_tensor,
                        torch.zeros(extra_size, dtype=torch.float32, device=self.device)
                    ], dim=0)
                    self.counts_tensor = torch.cat([
                        self.counts_tensor,
                        torch.zeros(extra_size, dtype=torch.int64, device=self.device)
                    ], dim=0)

                # Accumulate per-domain losses and counts
                self.losses_tensor.index_add_(0, key_ids, loss)
                self.counts_tensor.index_add_(0, key_ids, torch.ones_like(key_ids, dtype=torch.int64))


    def get_per_domain_stats(self):
        return self.losses_tensor.clone(), self.counts_tensor.clone(), self.max_domain_id.clone()

    def reset_per_domain_stats(self):
        self.losses_tensor.zero_()
        self.counts_tensor.zero_()
        self.max_domain_id = torch.tensor(self._default_domains - 1, dtype=torch.int32, device=self.device)

per_domain_loss_module = None

feed_back_time_accumulator = 0.0
wait_time_accumulator = 0.0

def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None, key_ids: Optional[torch.Tensor] = None, train_data_loader: Optional[torch.utils.data.DataLoader] = None, iteration: Optional[int] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    timers = get_timers()
    
    # print(f"[Rank {torch.distributed.get_rank()}] loss func output tensor of shape {output_tensor.shape}, output tensor: {output_tensor}")

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    
    if args.dataloader_type == 'mixtera' and key_ids is not None:
        assert train_data_loader is not None
        assert iteration is not None
        global per_domain_loss_module
        if per_domain_loss_module is None:
            per_domain_loss_module = PerDomainLoss(device=torch.cuda.current_device())
        per_domain_loss_module(output_tensor, key_ids)
        pp_tp_group = parallel_state.get_data_parallel_group()  # should be all last stage of pp in all dp groups
        # print(f"[Rank {torch.distributed.get_rank()}]", parallel_state._DATA_PARALLEL_GLOBAL_RANKS)
        # print(f"[Rank {torch.distributed.get_rank()}]", pp_tp_group)
        init_async_start = time.perf_counter()
        with torch.no_grad():
            losses_tensor, counts_tensor, max_id_tensor = per_domain_loss_module.get_per_domain_stats()
            max_handle = torch.distributed.all_reduce(max_id_tensor, op=torch.distributed.ReduceOp.MAX, async_op=True, group=pp_tp_group) # TODO: dp mesh vs dp group?
            per_domain_loss_module.reset_per_domain_stats()
            max_handle.wait()
            max_domain_id = max_id_tensor.item()
            # Resize tensors to the maximum domain ID
            if losses_tensor.size(0) < max_domain_id + 1:
                new_size = max_domain_id + 1 - losses_tensor.size(0)
                losses_tensor = torch.cat(
                    [losses_tensor, torch.zeros(new_size, dtype=losses_tensor.dtype, device=losses_tensor.device)], dim=0)
                counts_tensor = torch.cat(
                    [counts_tensor, torch.zeros(new_size, dtype=counts_tensor.dtype, device=counts_tensor.device)], dim=0)
            handle_losses = torch.distributed.all_reduce(losses_tensor, op=torch.distributed.ReduceOp.SUM, async_op=True, group=pp_tp_group)
            handle_counts = torch.distributed.all_reduce(counts_tensor, op=torch.distributed.ReduceOp.SUM, async_op=True, group=pp_tp_group)
        init_async_time = time.perf_counter() - init_async_start
        wait_mixtera_start = time.perf_counter()
        handle_losses.wait()
        handle_counts.wait()
        wait_mixtera_time = time.perf_counter() - wait_mixtera_start
        global wait_time_accumulator
        wait_time_accumulator += wait_mixtera_time
        # print(f"[Rank {torch.distributed.get_rank()}] [Iteration {iteration}] handle mixtera feedback ")
        dp_rank = parallel_state._DATA_PARALLEL_GLOBAL_RANKS.index(torch.distributed.get_rank())
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        mixtera_feedback_start = time.perf_counter()
        timers('mixtera-feedback', log_level=2).start()
        handle_mixtera_feedback(
            train_data_loader,
            iteration,
            losses_tensor,
            counts_tensor,
            dp_rank,
            tp_rank,
        )
        timers('mixtera-feedback').stop()
        mixtera_feedback_time = time.perf_counter() - mixtera_feedback_start
        global feed_back_time_accumulator
        feed_back_time_accumulator += mixtera_feedback_time
        
        # if iteration % args.log_interval == (args.log_interval - 1) and torch.distributed.get_rank() == 0:
        #     print(f"mixtera average feedback time {feed_back_time_accumulator/args.log_interval:.4f}s")
        #     print(f"mixtera average wait time {wait_time_accumulator/args.log_interval:.4f}s")
        #     feed_back_time_accumulator = 0.0
        #     wait_time_accumulator = 0.0

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False, train_data_loader: Optional[torch.utils.data.DataLoader] = None, iteration: Optional[int] = None):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, key_ids = get_batch(data_iterator, vp_stage)
        # print(f"[Rank {torch.distributed.get_rank()}] obtained batch at iteration {iteration}, {tokens.shape}, {labels.shape if hasattr(labels, "shape") else None}, {loss_mask.shape if hasattr(loss_mask, "shape") else None}, {attention_mask.shape}, {position_ids.shape}")
    timers('batch-generator').stop()
    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model, key_ids=key_ids, train_data_loader=train_data_loader, iteration=iteration)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model, key_ids=key_ids, train_data_loader=train_data_loader, iteration=iteration)


def is_dataset_built_on_rank(vp_stage=None):
    return is_first_or_last_pipeline_stage(vp_stage) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    
    if args.dataloader_type == 'mixtera':
        return mixtera_provider(train_val_test_num_samples, vp_stage)

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


class MixteraWrapper(torch.utils.data.IterableDataset):
    def __init__(self, torch_ds: MixteraTorchDataset):
        self.torch_ds = torch_ds
        self.return_key_id = torch_ds._return_key_id
        
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(self.torch_ds._res_str_args.chunk_reading_tokenizer, use_fast=True)
        self.eos = _tokenizer.eos_token_id

    def __iter__(self):
        for item in self.torch_ds:
            assert (self.return_key_id and isinstance(item, tuple) and len(item) == 2) or (not isinstance(item, tuple)), f"Inconsistent state:\n self.return_key_id = {self.return_key_id}\n item = {item}\n type(item)={type(item)}"

            if self.return_key_id:
                key_id = item[0]
                sample = item[1]
            else:
                sample = item
                key_id = None
            
            del item
            assert isinstance(key_id, int) or (key_id is None and not self.return_key_id), f"key id = {key_id} sample = {sample} item = {item} return_key_id = {self.return_key_id}"
            assert isinstance(sample, list), f"Sample type is {type(sample)}"
            assert isinstance(sample[0], int)

            x = torch.LongTensor(sample)
            input = x[:-1]
            label = x[1:]
            seq_len = len(input)

            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                input,
                self.eos,
                reset_attention_mask=False,
                reset_position_ids=False,
                eod_mask_loss=False,
                create_attention_mask=True,
            )
            if not self.return_key_id:
                yield {"tokens": input, "labels": label, "attention_mask":attention_mask, "loss_mask": loss_mask, "position_ids": position_ids}
            else:
                key_ids = torch.full((seq_len,), key_id, dtype=torch.long) 
                yield {"tokens": input, "labels": label, "attention_mask":attention_mask, "loss_mask": loss_mask, "position_ids": position_ids, "key_ids": key_ids}

    def __getstate__(self):
        state = self.__dict__.copy()
        if "torch_ds" in state: # Not pickable, and is pickled on checkpoint.
            del state["torch_ds"]
        return state
    

def mixtera_provider(train_val_test_num_samples, vp_stage=None):
    args = get_args()
    
    server_host = args.mixtera_ip # '172.28.41.32'
    server_port = 8088
    client = MixteraClient.from_remote(server_host, server_port)
        
    if torch.distributed.get_rank() == 0 and not client.register_dataset(
            "pile",
            Path("/capstor/store/cscs/swissai/infra02/mixtera_a09/data/tp_processed/train_processed_split1k"),
            # Path("/iopsstor/scratch/cscs/yiswang/data/pile"),
            JSONLDataset,
            parsing_func,
            "PILE" # "SLIM_PAJAMA",
        ):
        print("already registered dataset")
    torch.distributed.barrier()
    chunk_size = args.mixtera_chunk_size
    seq_len = args.seq_length
    
    mixture_static = StaticMixture(chunk_size=chunk_size, strict=False, mixture={
                MixtureKey({"pile_set_name": ["Pile-CC"]}): 0.1121,
                MixtureKey({"pile_set_name": ["PubMed Central"]}): 0.1071,
                MixtureKey({"pile_set_name": ["Books3"]}): 0.0676,
                MixtureKey({"pile_set_name": ["OpenWebText2"]}): 0.1247,
                MixtureKey({"pile_set_name": ["ArXiv"]}): 0.1052,
                MixtureKey({"pile_set_name": ["Github"]}): 0.0427,
                MixtureKey({"pile_set_name": ["FreeLaw"]}): 0.0386,
                MixtureKey({"pile_set_name": ["StackExchange"]}): 0.0929,
                MixtureKey({"pile_set_name": ["USPTO Backgrounds"]}): 0.0420,
                MixtureKey({"pile_set_name": ["PubMed Abstracts"]}): 0.0845,
                MixtureKey({"pile_set_name": ["Gutenberg (PG-19)"]}): 0.0199,
                MixtureKey({"pile_set_name": ["OpenSubtitles"]}): 0.0124,
                MixtureKey({"pile_set_name": ["Wikipedia (en)"]}): 0.0919,
                MixtureKey({"pile_set_name": ["DM Mathematics"]}): 0.0198,
                MixtureKey({"pile_set_name": ["Ubuntu IRC"]}): 0.0074,
                MixtureKey({"pile_set_name": ["BookCorpus2"]}): 0.0044,
                MixtureKey({"pile_set_name": ["EuroParl"]}): 0.0043,
                MixtureKey({"pile_set_name": ["HackerNews"]}): 0.0075,
                MixtureKey({"pile_set_name": ["YoutubeSubtitles"]}): 0.0042,
                MixtureKey({"pile_set_name": ["PhilPapers"]}): 0.0027,
                MixtureKey({"pile_set_name": ["NIH ExPorter"]}): 0.0052,
                MixtureKey({"pile_set_name": ["Enron Emails"]}): 0.0030,
            })
    
    ado_log_dir = f"/iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/adolog"
    if torch.distributed.get_rank() == 0 and not os.path.exists(ado_log_dir):
        os.mkdir(ado_log_dir)
    torch.distributed.barrier()
        
    num_workers = args.num_workers
    job_id = args.mixtera_job_id    

    mixture_ado_def = DynamicMixture(strict=False, chunk_size=chunk_size, initial_mixture=mixture_static, mixing_alg=AdoDynamicMixing(gamma2=0.1, count_normalizer=seq_len, use_same_step_size=True, delta_min=0.01, subsampling_interval=10, scaling_law_update_interval=1000, ignore_initial_steps=500, start_step=1000, logging_path=f"/iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/adolog/{job_id}_seqfix.json", variant="vanilla"))   
    
    mixture = mixture_ado_def
    
    query = Query.for_job(job_id).select(None) # ("redpajama_set_name", "!=", "RedPajamaCommonCrawl")
    
    world_size: int = torch.distributed.get_world_size()
    nodes_per_dp_group = world_size // parallel_state.get_data_parallel_world_size()
    # dp_group_id = parallel_state.get_data_parallel_rank() # rank in its dp group or its dp group id ?
    dp_group_id = parallel_state._DATA_PARALLEL_GLOBAL_RANKS.index(torch.distributed.get_rank())
    dp_degree = parallel_state.get_data_parallel_world_size()
    dp_group = parallel_state.get_model_parallel_group()
    assert dp_group.size() == nodes_per_dp_group, f"dp_group handler has size of {dp_group.size()} not {nodes_per_dp_group}."
    node_id = parallel_state._MODEL_PARALLEL_GLOBAL_RANKS.index(torch.distributed.get_rank()) # Is this global rank?
    print(f"world_size: {world_size}, nodes_per_dp_group: {nodes_per_dp_group}, dp_group_id: {dp_group_id}, dp_degree: {dp_degree}, node_id: {node_id}")
    torch.distributed.barrier()

    
    qea = QueryExecutionArgs(mixture=mixture, dp_groups=dp_degree, nodes_per_group=nodes_per_dp_group, num_workers=num_workers)
    # TODO set tunnel_via_server to False 
    rsa = ResultStreamingArgs(job_id=job_id, dp_group_id=dp_group_id, node_id=node_id, tunnel_via_server=False, # set tunnel_via_server to False 
                             chunk_reading_degree_of_parallelism=1,
                             chunk_reading_tokenizer="EleutherAI/gpt-neox-20b",  # EleutherAI/gpt-neox-20b
                             chunk_reading_mixture_type="token", 
                             chunk_reading_sequence_len=seq_len,
                             chunk_reading_token_overlapping=False, 
                             chunk_reading_eos=True)
    # using eos, require extra process with built-in method: _get_ltor_masks_and_position_ids
    # bos and eos token id: 50256
    load_path = Path(args.load)
    if not os.path.exists(load_path / "mixtera.id"):
        load_path = None
        
    return_key_id = isinstance(mixture, DynamicMixture) # not the best criterion to decide this on, but suffices for now.
    global USING_KEYIDS
    USING_KEYIDS = return_key_id
        
    mixtera_ds = MixteraTorchDataset(client, query, qea, rsa, checkpoint_path=load_path, return_key_id=return_key_id)
    
    train_ds = MixteraWrapper(mixtera_ds)    
    
    valid_ds = None
    test_ds = None

    return train_ds, valid_ds, test_ds



if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,  # train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
