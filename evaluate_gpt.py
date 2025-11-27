# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

from functools import reduce
from logging import logMultiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))

from lm_eval.models.huggingface import HFLM
from lm_eval import tasks, evaluator, utils, api
from lm_eval.api.model import CacheHook

from tqdm import tqdm
import torch.nn.functional as F

from lm_eval.api.registry import ALL_TASKS
from pretrain_gpt import model_provider
import numpy as np
import time

import torch
from megatron.training.global_vars import get_args
from megatron.training.utils import print_rank_0
from megatron.training.global_vars import get_tokenizer
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.training.training import get_model
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.utils import get_model_config


from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.training.utils import unwrap_model
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
import pickle
import json

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.core.transformer.module import Float16Module
# from deepspeed.runtime.pipe import schedule
# from deepspeed.accelerator import get_accelerator

class EvalHarnessAdaptor(HFLM):
    def __init__(self, model, tokenizer):
        args = get_args()
        self.args = args
        self._model = model
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.EOT_TOKEN_ID = tokenizer.eos_token_id

        self._max_length = args.seq_length

        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = args.micro_batch_size

        self.cache_hook = CacheHook(None)
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self.is_main = self._rank == 0
        self.is_local_main = self._rank == 0
        self._device = torch.cuda.current_device()
        self.is_model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        self.adaptive_seq_len = args.adaptive_seq_len
        if self.is_data_parallel and args.moe_expert_parallel_size == 1: # For MoE model, allow a "fake data parallel" in order to partition model into multiple gpus
            raise NotImplementedError("Data parallelism is currently not supported for evaluation")

        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

        config = get_model_config(model)
        self.p2p_communicator = P2PCommunicator(
            pp_group=mpu.get_pipeline_model_parallel_group(), config=config
        )


    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device


    # def loglikelihood(self, requests):
    #     new_reqs = []
    #     for inst in requests:
    #         context = inst.args[0]
    #         continuation = task.doc_to_target(inst.doc)
    #         if context == "":
    #             # end of text as context
    #             context_enc = [self.EOT_TOKEN_ID]
    #         else:
    #             context_enc = self.tokenizer_encode(context)
# 
    #         continuation_enc = self.tokenizer_encode(continuation)
# 
    #         new_reqs.append(((context, continuation), context_enc, continuation_enc))
# 
    #     return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer_encode(string),
                    prefix_token=self.EOT_TOKEN_ID,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        self.model.eval()
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen
                    if not self.adaptive_seq_len:
                        padding_length = self.max_length
                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))

                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1).cpu()  # [batch, seq, vocab]

                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                        max_equal = (greedy_tokens == cont_toks).all()
                        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))
                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                        res.append(answer)

        if not mpu.is_pipeline_last_stage():
            # @HACK: To make the eval harness happy on threads that don't have access to the results.
            #        We just randomly generate some data.
            res = [(np.random.rand(), np.random.rand()>0.5) for _ in requests]

        return reord.get_original(res)

    def create_model_inputs(self, tokens):
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            self.EOT_TOKEN_ID,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            create_attention_mask=True,
        )

        return (tokens, position_ids, attention_mask), (tokens, loss_mask)

    def _model_call(self, inps):
        args = get_args()

        
        # Since the shape of the micro-batch will change
        # We need set the correct shapes here
        # So that latter pipeline stages knows which shapes to expect.
        # Otherwise we will deadlock.
        args.micro_batch_size = len(inps)
        args.seq_length = len(inps[0])
        args.max_position_embeddings = args.seq_length
        
        input_tensor = self.p2p_communicator.recv_forward()
        
        # Forward pass through the model.
        unwrapped_model = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.set_input_tensor(input_tensor)
        output = self.model(*self.create_model_inputs(inps)[0])
        self.p2p_communicator.send_forward(output)

        if mpu.is_pipeline_last_stage():
            return gather_from_tensor_model_parallel_region(output)[..., :self.tokenizer.vocab_size]
        else:
            return None

    def tokenizer_encode(self, text):
        """Tokenize text *without* adding special tokens."""
        self.tokenizer.encode(text, add_special_tokens=False)


from megatron.training.initialize import initialize_megatron
import megatron
from megatron.training.checkpointing import load_checkpoint

from model_provider import model_provider
from gpt_builders import gpt_builder
from functools import partial

# from tools.convert_checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
# from tools.convert_checkpoint.deepspeed_to_megatron import _create_rank_checkpoint

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


# Note(Hesslow):
# The model loading is a bit convoluted.
# We want to parse out the model arguments from the checkpoint and use those to initialize megatron-ds.
#
# However megatron-ds expects its arguments on the command line.
# And at that point we don't know them.
#
# Instead we use Jasons way: we load the arguments form the checkpoint and then override _parse_args to return whatever args we want.
#
# If the checkpoint is old, some new arguments may have been introduced and the code will expect these arguments to exist.
# In order to support this we _first_ parse the arguments normally, and then override them with the arguments from the checkpoint.
# Keeping the default-value of newer arguments.
#
# We then use the megatron deepspeed converter to load the deepspeed checkpoints as if they we're megatron checkpoints.
def load_ds_checkpoint_and_setup_megatron(extra_args_provider):
    args = parse_args(extra_args_provider=extra_args_provider)

    initialize_megatron(extra_args_provider=extra_args_provider)
    torch.distributed.barrier()
    
    model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
    load_checkpoint(model, None, None, strict=False)
    model = model[0]

    if args.eval_fp32:
        model = model.float()

    torch.distributed.barrier()
    return model

def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Evaluation options')
    group.add_argument('--task_list', type=str, default = "lambada", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--results_path', type=str, default = "./results.json", help='Path to where the results will be stored.')
    group.add_argument('--adaptive_seq_len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--num_fewshot', type=int, default = 0, help='Number of few-shot prompts.')
    group.add_argument('--eval_fp32',  default = False, action='store_true', help='Should the evaluation run in fp32')
    return parser

from megatron.training.arguments import parse_args

def main():
    start = time.time()
    model = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)

    args = get_args()

    task_list = args.task_list.split(',')
    print(task_list)
    task_dict = tasks.get_task_dict(task_list)

    model.module.activation_checkpoint_interval = 0
    model._compute_loss = False
    model.fwd_outputs = []

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True)
    adaptor = EvalHarnessAdaptor(model, tokenizer)
    results = evaluator.evaluate(adaptor, task_dict, 5)

    if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        print(json.dumps(results, indent=2))
        with open(args.results_path, 'w') as outfile:
            json.dump(results, outfile, indent = 4)
    end = time.time()

if __name__ == '__main__':
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    main()