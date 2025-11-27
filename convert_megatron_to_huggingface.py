####################################################################################################

# Copyright (c) 2021-, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

####################################################################################################

#
# Note: If when running this conversion script you're getting an exception:
#     ModuleNotFoundError: No module named 'megatron.model.enums'
# you need to tell python where to find the clone of Megatron-LM, e.g.:
#
# cd /tmp
# git clone https://github.com/NVIDIA/Megatron-LM
# PYTHONPATH=/tmp/Megatron-LM python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py ...
#
# if you already have it cloned elsewhere, simply adjust the path to the existing path
#
# If the training was done using a Megatron-LM fork, e.g.,
# https://github.com/microsoft/Megatron-DeepSpeed/ then chances are that you need to have that one
# in your path, i.e., /path/to/Megatron-DeepSpeed/
#

import argparse
import os
import re
import zipfile
from typing import Dict, Any, List, Tuple

import torch

from transformers import AutoTokenizer, LlamaConfig, AutoModelForCausalLM


####################################################################################################


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val:
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


####################################################################################################


def convert_megatron_checkpoint(args, input_state_dict, config):
    # The converted output model.
    output_state_dict = {}

    # old versions did not store training args
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # do not make the user write a config file when the exact dimensions/sizes are already in the checkpoint
        # from pprint import pprint
        # pprint(vars(ds_args))

        config.vocab_size = ds_args.padded_vocab_size
        config.max_position_embeddings = ds_args.max_position_embeddings
        config.hidden_size = ds_args.hidden_size
        config.num_hidden_layers = ds_args.num_layers
        config.num_attention_heads = ds_args.num_attention_heads
        config.intermediate_size = ds_args.ffn_hidden_size
        # pprint(config)

    # The number of heads.
    heads = config.n_head
    # The hidden_size per head.
    hidden_size_per_head = config.n_embd // config.n_head
    # Megatron-LM checkpoint version
    if "checkpoint_version" in input_state_dict:
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # The model.
    model = input_state_dict["model"]
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: config.vocab_size, :]
    output_state_dict["transformer.wte.weight"] = word_embeddings

    # # The position embeddings.
    # pos_embeddings = embeddings["position_embeddings"]["weight"]
    # # Read the causal mask dimension (seqlen). [max_sequence_length, hidden_size]
    n_positions = 2048 # pos_embeddings.size(0)
    # if n_positions != config.n_positions:
    #     raise ValueError(
    #         f"pos_embeddings.max_sequence_length={n_positions} and config.n_positions={config.n_positions} don't match"
    #     )
    # # Store the position embeddings.
    # output_state_dict["transformer.wpe.weight"] = pos_embeddings

    # The transformer.
    transformer = lm["transformer"] if "transformer" in lm else lm["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z0-9_]+)")

    # The simple map of names for "automated" rules.
    megatron_to_transformers = {
        "attention.dense": ".attn.c_proj.",
        "self_attention.dense": ".attn.c_proj.",
        "self_attention.proj": ".attn.c_proj.",  # New format
        "mlp.dense_h_to_4h": ".mlp.c_fc.",
        "mlp.dense_4h_to_h": ".mlp.c_proj.",
        "layernorm_mlp.fc1": ".mlp.c_fc.",  # New format
        "layernorm_mlp.fc2": ".mlp.c_proj.",  # New format
    }

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            continue

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)
        # The name of the layer.
        layer_name = f"transformer.h.{layer_idx}"

        # Handle _extra_state keys (skip them)
        if weight_or_bias == "_extra_state":
            continue

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm") or weight_or_bias.startswith("layer_norm"):
            if weight_or_bias.startswith("layer_norm"):
                # New format: layers.X.self_attention.layernorm_qkv.layer_norm_weight
                if op_name == "self_attention.layernorm_qkv":
                    ln_name = "ln_1"  # Pre-attention layer norm
                elif op_name == "layernorm_mlp":
                    ln_name = "ln_2"  # Pre-MLP layer norm
                else:
                    ln_name = "ln_1" if op_name.startswith("input") else "ln_2"

                param_name = "weight" if weight_or_bias == "layer_norm_weight" else "bias"
                output_state_dict[layer_name + "." + ln_name + "." + param_name] = val
            else:
                # Old format
                ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # Handle QKV projections - new format: self_attention.layernorm_qkv.weight/bias
        elif op_name == "self_attention.layernorm_qkv" and weight_or_bias in ["weight", "bias"]:
            if weight_or_bias == "weight":
                # Insert a tensor of 1x1xDxD bias.
                causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float16)).view(
                    1, 1, n_positions, n_positions
                )
                output_state_dict[layer_name + ".attn.bias"] = causal_mask

                # Insert a "dummy" tensor for masked_bias.
                masked_bias = torch.tensor(-1e4, dtype=torch.float16)
                output_state_dict[layer_name + ".attn.masked_bias"] = masked_bias

                out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
                # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
                out_val = out_val.transpose(0, 1).contiguous()
                # Store.
                output_state_dict[layer_name + ".attn.c_attn.weight"] = out_val
            else:  # bias
                out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
                # Store. No change of shape.
                output_state_dict[layer_name + ".attn.c_attn.bias"] = out_val

        # Transpose the QKV matrix - old format.
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "weight":
            # Insert a tensor of 1x1xDxD bias.
            causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float16)).view(
                1, 1, n_positions, n_positions
            )
            output_state_dict[layer_name + ".attn.bias"] = causal_mask

            # Insert a "dummy" tensor for masked_bias.
            masked_bias = torch.tensor(-1e4, dtype=torch.float16)
            output_state_dict[layer_name + ".attn.masked_bias"] = masked_bias

            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
            out_val = out_val.transpose(0, 1).contiguous()
            # Store.
            output_state_dict[layer_name + ".attn.c_attn.weight"] = out_val

        # Transpose the bias - old format.
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "bias":
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Store. No change of shape.
            output_state_dict[layer_name + ".attn.c_attn.bias"] = out_val

        # Transpose the weights.
        elif weight_or_bias == "weight":
            # DEBUG: Check if op_name exists in the mapping
            if op_name not in megatron_to_transformers:
                continue
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "weight"] = val.transpose(0, 1)

        # Copy the bias.
        elif weight_or_bias == "bias":
            # DEBUG: Check if op_name exists in the mapping
            if op_name not in megatron_to_transformers:
                continue
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "bias"] = val

        # Handle new format MLP weights/biases
        elif weight_or_bias in ["fc1_weight", "fc2_weight", "fc1_bias", "fc2_bias"]:
            if weight_or_bias == "fc1_weight":
                output_state_dict[layer_name + ".mlp.c_fc.weight"] = val.transpose(0, 1)
            elif weight_or_bias == "fc1_bias":
                output_state_dict[layer_name + ".mlp.c_fc.bias"] = val
            elif weight_or_bias == "fc2_weight":
                output_state_dict[layer_name + ".mlp.c_proj.weight"] = val.transpose(0, 1)
            elif weight_or_bias == "fc2_bias":
                output_state_dict[layer_name + ".mlp.c_proj.bias"] = val

        else:
            print(
                f"DEBUG: Unhandled key: {key} (layer {layer_idx}, op_name: '{op_name}', weight_or_bias: '{weight_or_bias}')"
            )

    # DEBUG.
    assert config.n_layer == layer_idx + 1

    # The final layernorm - handle both old and new formats.
    if "final_layernorm.weight" in transformer:
        # Old format
        output_state_dict["transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
        output_state_dict["transformer.ln_f.bias"] = transformer["final_layernorm.bias"]
    elif "final_norm.weight" in transformer:
        # New format
        output_state_dict["transformer.ln_f.weight"] = transformer["final_norm.weight"]
        # output_state_dict["transformer.ln_f.bias"] = transformer["final_norm.bias"]
    else:
        print("WARNING: Could not find final layer norm weights!")

    # For LM head, transformers' wants the matrix to weight embeddings.
    output_state_dict["lm_head.weight"] = word_embeddings

    # It should be done!
    return output_state_dict



def infer_config_from_checkpoint(state_dict: Dict[str, Any], ffn_hidden_size: int, max_seq_len: int, n_kv_heads_override: int = None, n_heads_override: int = None, rope_theta: int = None) -> Dict:
    states = state_dict["model"]["language_model"]
    # Infer hidden_size from the token embedding weight (dimensions: vocab_size x hidden_size)
    vocab_size, hidden_size = states["embedding"]["word_embeddings"]["weight"].shape
    print(f"Inferred vocab_size={vocab_size}, hidden_size={hidden_size} from token embeddings.")
    # Count unique transformer layers using keys like "layers.{i}.attention.wq.weight"
    layer_indices = set()
    for key in states["encoder"].keys():
        if key.startswith("layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    if not layer_indices:
        raise ValueError("No transformer layer keys found in checkpoint!")
    print(f"Inferred n_layers={max(layer_indices) + 1} from transformer layer keys.")
    n_layers = max(layer_indices) + 1
    n_kv_heads = n_kv_heads_override # if n_kv_heads_override is not None else infer_heads_and_kv(states, hidden_size)[1]
    n_heads = n_heads_override # if n_heads_override is not None else infer_heads_and_kv(states, hidden_size)[0]
    # Infer intermediate dimension from one of the feed-forward weights:
    # Expected shape for feed_forward.w1.weight is (intermediate_dim, hidden_size)
    intermediate_dim = ffn_hidden_size# states["layers.0.feed_forward.w1.weight"].shape[0]

    config = {
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "max_seq_len": max_seq_len,
        "vocab_size": vocab_size,
        "intermediate_dim": intermediate_dim,
        "activation": "silu",
        "initializer_range": 0.02,
        "rope_theta": 10000 if rope_theta is None else rope_theta,
        "norm_eps": 1e-05,
    }
    return config


def to_huggingface(states: Dict[str, Any], config: Dict):
    # step 1: create a config file containing the model architecture
    hf_config = LlamaConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_dim"],
        num_hidden_layers=config["n_layers"],
        num_attention_heads=config["n_heads"],
        num_key_value_heads=config["n_kv_heads"],
        hidden_act=config["activation"],
        max_position_embeddings=config["max_seq_len"],
        initializer_range=config["initializer_range"],
        rope_theta=config["rope_theta"],
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config)
    # step 2: load the model weights, translating them accordingly
    new_state_dict = {}
    with torch.inference_mode():
        states_dicts = states["model"]["language_model"]
        
        # states_dicts = {k: v.contiguous() for k, v in states_dicts.items()}
        new_state_dict["model.embed_tokens.weight"] = states_dicts["embedding"]["word_embeddings"]["weight"].contiguous()
        
        if 'output_layer' not in states_dicts:
            new_state_dict["lm_head.weight"] = states_dicts["embedding"]["word_embeddings"]["weight"].contiguous()
        else:
            new_state_dict["lm_head.weight"] = states_dicts["output_layer"]["weight"].contiguous()
        
        
        states_dicts = states_dicts["encoder"]
        
        
        new_state_dict["model.norm.weight"] = states_dicts["final_norm.weight"]
        dims_per_head = hf_config.hidden_size // hf_config.num_attention_heads

        for i in range(hf_config.num_hidden_layers):
            qkv = states_dicts[f"layers.{i}.self_attention.query_key_value.weight"]
            
            input_shape = qkv.size()
            saved_shape = (hf_config.num_attention_heads, 3, dims_per_head) + input_shape[1:]
            qkv = qkv.view(*saved_shape)
            qkv = qkv.transpose(0, 1).contiguous()
            qkv = qkv.view(*input_shape)
            
            q, k, v = torch.chunk(qkv, chunks=3, dim=0)
            new_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = q.contiguous()
            new_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = k.contiguous()
            new_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = v.contiguous()
            new_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = states_dicts[
                f"layers.{i}.self_attention.dense.weight"
            ].contiguous()
            new_state_dict[f"model.layers.{i}.input_layernorm.weight"] = states_dicts[
                f"layers.{i}.input_norm.weight"
            ].contiguous()
            new_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = states_dicts[
                f"layers.{i}.post_attention_norm.weight"
            ].contiguous()
            
            h_to_4h = states_dicts[f"layers.{i}.mlp.dense_h_to_4h.weight"]
            # input_shape = h_to_4h.size()
            # saved_shape = (-1, 2) + input_shape[1:]
            # h_to_4h = h_to_4h.view(*saved_shape)
            # h_to_4h = h_to_4h.transpose(0, 1).contiguous()
            # h_to_4h = h_to_4h.view(*input_shape)
            
            gate_proj, up_proj = torch.chunk(h_to_4h, chunks=2)
            
            new_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = gate_proj.contiguous()
            new_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = states_dicts[
                f"layers.{i}.mlp.dense_4h_to_h.weight"
            ].contiguous()
            new_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = up_proj.contiguous()

        model.load_state_dict(new_state_dict, strict=True, assign=True)
        model.eval()
        return model, hf_config



####################################################################################################


def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument(
        "--path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output folder path for the Hugging Face model.")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b", help="Which tokenizer to use.")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--n_kv_heads", type=int, default=None, help="Override number of key-value heads if provided.")
    # TODO: check n_heads relation with kv_channels
    parser.add_argument("--n_heads", type=int, default=16 , help="Override number of heads if provided.")
    parser.add_argument("--ffn_hidden_size", type=int, default=2048 , help="Override number of ffn_hidden_size if provided.")
    parser.add_argument("--rope_theta", type=int, default=None, help="Override rope theta if provided.")
    args = parser.parse_args()

    # Extract the basename.
    basename = os.path.dirname(args.path_to_checkpoint)

    # Load the model.
    # the .zip is very optional, let's keep it for backward compatibility
    print(f"Extracting PyTorch state dictionary from {args.path_to_checkpoint}")
    if args.path_to_checkpoint.endswith(".zip"):
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location="cpu", weights_only=True)
    else:
        input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu", weights_only=False)

    if "checkpoint_version" in input_state_dict:
        print("version: ", input_state_dict["checkpoint_version"])

    ds_args = input_state_dict.get("args", None)

    # Read the config, or default to the model released by NVIDIA.
    if args.config_file == "":
        # Spell out all parameters in case the defaults change.
        args.max_seq_len = int(ds_args.max_position_embeddings)
        args.n_heads = int(ds_args.num_attention_heads)
        args.rope_theta = int(ds_args.rotary_base)
        args.ffn_hidden_size = int(ds_args.ffn_hidden_size)
        config = infer_config_from_checkpoint(input_state_dict, args.ffn_hidden_size, args.max_seq_len, args.n_kv_heads, args.n_heads, args.rope_theta)
    else:
        config = LlamaConfig.from_json_file(args.config_file)

    # config.architectures = ["GPT2LMHeadModel"]

    # Convert.
    print("Converting")
    # output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)
    model, hf_config = to_huggingface(input_state_dict, config)
    
    output_path = args.output

    print(f"Saving model and configuration to {output_path}...")
    model.save_pretrained(output_path)
    hf_config.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(output_path)


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
