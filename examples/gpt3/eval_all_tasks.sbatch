#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=/iopsstor/scratch/cscs/yiswang/server_logs/convert/convert_%j.out
#SBATCH --environment=mixtera
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --nodes=1
#SBATCH --account=a-infra02
#SBATCH --time=00:59:00

set -ex

export NCCL_NET="AWS Libfabric"
export CUDA_DEVICE_MAX_CONNECTIONS=1

LOAD_DIR="/iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/1B-default-ADO/ckpt" 
SAVE_DIR="$LOAD_DIR/../ckpt-converted" 



cd /iopsstor/scratch/cscs/yiswang/Megatron-mixtera

python eval_lmeval.py \
    --checkpoint-dir "$SAVE_DIR/iter_0030000/huggingface_model" \
    --output-dir "$SAVE_DIR/iter_0030000/huggingface_eval_all_tasks" \
    --tasks "hellaswag,winogrande,arc_easy,arc_challenge,lambada_openai,openbookqa,piqa"