#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=/iopsstor/scratch/cscs/yiswang/server_logs/convert/convert_%j.out
#SBATCH --environment=mixtera
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --nodes=1
#SBATCH --account=a-infra02
#SBATCH --time=00:59:00

set -ex

LOAD_DIR="/iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/1B-default-4096chunk/ckpt" 
SAVE_DIR="$LOAD_DIR/../ckpt-converted" 

ITER_PATH="$LOAD_DIR/latest_checkpointed_iteration.txt"


export NCCL_NET="AWS Libfabric"
export CUDA_DEVICE_MAX_CONNECTIONS=1

pushd /iopsstor/scratch/cscs/yiswang/mixtera && pip install -e . && popd
pip install megatron-core
pip install --no-build-isolation transformer-engine[pytorch]


cd /iopsstor/scratch/cscs/yiswang/Megatron-mixtera

# Loop over ITER from 2500 to 27500 with a step of 2500
for i in $(seq 2500 2500 27500)
do
    # Pad the iteration number to 7 digits (e.g., 0002500)
    ITER=$(printf %07d $i)
    echo "Processing iteration: $ITER"

    # Conversion step
    python convert_megatron_to_huggingface_decoder.py \
        --path_to_checkpoint "$LOAD_DIR/iter_$ITER/mp_rank_00/model_optim_rng.pt" \
        --output "$SAVE_DIR/iter_$ITER/huggingface_model"

    # Evaluation step
    python eval_lmeval.py \
        --checkpoint-dir "$SAVE_DIR/iter_$ITER/huggingface_model" \
        --output-dir "$SAVE_DIR/iter_$ITER/huggingface_eval" \
        --tasks "lambada_openai,hellaswag,openbookqa,arc_easy"
done