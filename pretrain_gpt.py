# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import torch

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


from mixtera.torch import MixteraTorchDataset
from mixtera.core.client import MixteraClient, QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.query import Query
from mixtera.core.query.mixture import InferringMixture, StaticMixture, MixtureKey
from mixtera.core.query.mixture.dynamic_mixture import DynamicMixture
from mixtera.core.algo.ado.ado import AdoDynamicMixing
from mixtera.utils.feedback import handle_mixtera_feedback
from mixtera.utils.checkpoint import handle_mixtera_checkpoint

from time import time
from pathlib import Path

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage):
        next(data_iterator)
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    
    print(f"Rank {torch.distributed.get_rank()}, tokens {batch['tokens']}", flush=True)
    if torch.distributed.get_rank() == 0:
        print("*****************************************")

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
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

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

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


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
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
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, vp_stage)
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
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


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

from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.datacollection.index.parser.metadata_parser import MetadataProperty
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.datacollection.index.parser import MetadataParser
from mixtera.core.datacollection.index.parser.metadata_parser import MetadataProperty
from mixtera.core.query import Query
from mixtera.core.query.mixture import ArbitraryMixture, Mixture
from mixtera.torch import MixteraTorchDataset

class TestMetadataParser(MetadataParser):
    @classmethod
    def get_properties(cls) -> list[MetadataProperty]:
        return [
            MetadataProperty(
                name="language",
                dtype="ENUM",
                multiple=False,
                nullable=False,
                enum_options={"JavaScript", "HTML"},
            ),
            MetadataProperty(
                name="license",
                dtype="STRING",
                multiple=False,
                nullable=False,
                enum_options={"CC", "MIT"},
            ),  # Could be ENUM but we are using string to test
            MetadataProperty(
                name="doublelanguage",
                dtype="ENUM",
                multiple=True,
                nullable=False,
                enum_options={"JavaScript", "HTML"},
            ),
        ]

    def parse(
        self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]
    ) -> None:
        metadata = payload["meta"]
        self.add_metadata(
            sample_id=line_number,
            language=metadata["language"],
            license=metadata["license"],
            doublelanguage=[metadata["language"], metadata["language"]],
        )

def parsing_func(sample):
    import json

    return json.loads(sample)["text"]


class MixteraWrapper(torch.utils.data.IterableDataset):
    def __init__(self, torch_ds: MixteraTorchDataset, return_key_id: bool):
        self.torch_ds = torch_ds
        self.return_key_id = return_key_id
        
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
                key_ids = torch.full((seq_len,), key_id, dtype=torch.long) if self.return_key_id else None
                yield {"tokens": input}, label, key_ids

    def __getstate__(self):
        state = self.__dict__.copy()
        if "torch_ds" in state: # Not pickable, and is pickled on checkpoint.
            del state["torch_ds"]
        return state
    

def mixtera_provider(train_val_test_num_samples, vp_stage=None):
    args = get_args()
    
    server_host = "172.28.46.204"
    server_port = 8088
    client = MixteraClient.from_remote(server_host, server_port)
    # client.register_metadata_parser("TEST_PARSER", TestMetadataParser)
    # 
    # if not client.register_dataset(
    #         "server_integrationtest_dataset_megatron_1",
    #         Path("/iopsstor/scratch/cscs/yiswang/tmp") / "testd.jsonl",
    #         JSONLDataset,
    #         parsing_func,
    #         "TEST_PARSER",
    #     ):
    #     print("Dataset already registered.")
        
    if not client.register_dataset(
            "slimpajama_chunk1_2",
            Path("/iopsstor/scratch/cscs/yiswang/data/mixtera"),
            JSONLDataset,
            parsing_func,
            "SLIM_PAJAMA",
        ):
        print("already registered dataset")
    
    mixture = InferringMixture(chunk_size=42)
    num_workers = args.num_workers
    job_id = "Megatron-mixtera" + "22"
    query = Query.for_job(job_id).select(("redpajama_set_name", "==", "RedPajamaCommonCrawl"))
    
    world_size: int = torch.distributed.get_world_size()
    nodes_per_dp_group = world_size // parallel_state.get_data_parallel_world_size()
    # dp_group_id = parallel_state.get_data_parallel_rank() # rank in its dp group or its dp group id ?
    dp_group_id = parallel_state._DATA_PARALLEL_GLOBAL_RANKS.index(torch.distributed.get_rank())
    dp_degree = parallel_state.get_data_parallel_world_size()
    node_id = parallel_state._MODEL_PARALLEL_GLOBAL_RANKS.index(torch.distributed.get_rank()) # Is this global rank?
    print(f"world_size: {world_size}, nodes_per_dp_group: {nodes_per_dp_group}, dp_group_id: {dp_group_id}, dp_degree: {dp_degree}, node_id: {node_id}")
    torch.distributed.barrier()

    
    qea = QueryExecutionArgs(mixture=mixture, dp_groups=dp_degree, nodes_per_group=nodes_per_dp_group, num_workers=num_workers)
    rsa = ResultStreamingArgs(job_id=job_id, dp_group_id=dp_group_id, node_id=node_id, tunnel_via_server=True,
                             chunk_reading_tokenizer="EleutherAI/gpt-neox-20b",  # EleutherAI/gpt-neox-20b
                             chunk_reading_mixture_type="token", 
                             chunk_reading_sequence_len=4096,
                             chunk_reading_eos=True)
    # using eos, require extra process with built-in method: _get_ltor_masks_and_position_ids
    # bos and eos token id: 50256
    load_path = Path(args.load)
    if not os.path.exists(load_path / "mixtera.id"):
        load_path = None
        
    mixtera_ds = MixteraTorchDataset(client, query, qea, rsa, checkpoint_path=load_path)
    
    train_ds = MixteraWrapper(mixtera_ds, return_key_id=False)    
    
    valid_ds = None
    test_ds = None
    
    valid_jobid = job_id + "_valid"
    valid_rand = torch.randint(1, 1000, (1,)).cuda()
    torch.distributed.broadcast(valid_rand, src=0)
    valid_jobid += str(valid_rand.item())
    rsa_valid = ResultStreamingArgs(job_id=valid_jobid, dp_group_id=dp_group_id, node_id=node_id, tunnel_via_server=True,
                             chunk_reading_tokenizer="EleutherAI/gpt-neox-20b",  # EleutherAI/gpt-neox-20b
                             chunk_reading_mixture_type="token", 
                             chunk_reading_sequence_len=4096,
                             chunk_reading_eos=True)
    query_valid = Query.for_job(valid_jobid).select(("redpajama_set_name", "==", "RedPajamaC4"))
    valid_ds = MixteraWrapper(MixteraTorchDataset(client, query_valid, qea, rsa_valid), return_key_id=False)  
    
    # test_jobid = job_id + "_test"
    # rsa_test = ResultStreamingArgs(job_id=test_jobid, dp_group_id=dp_group_id, node_id=node_id, tunnel_via_server=True,
    #                          chunk_reading_tokenizer="EleutherAI/gpt-neox-20b",  # EleutherAI/gpt-neox-20b
    #                          chunk_reading_mixture_type="token", 
    #                          chunk_reading_sequence_len=4096,
    #                          chunk_reading_eos=True)
    # query_test = Query.for_job(test_jobid).select(("redpajama_set_name", "==", "RedPajamaC4"))
    # test_ds = MixteraWrapper(MixteraTorchDataset(client, query_test, qea, rsa_test), return_key_id=False)  
    
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
