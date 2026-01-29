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

pushd /iopsstor/scratch/cscs/yiswang/mixtera && pip install -e . && popd
pip install megatron-core
pip install --no-build-isolation transformer-engine[pytorch]

LOAD_DIR="/iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/ckpt" 
SAVE_DIR="$LOAD_DIR/../ckpt-converted" 

ITER_PATH="$LOAD_DIR/latest_checkpointed_iteration.txt"

ITER=$(cat $ITER_PATH)
ITER=$(printf %07d $ITER)

echo $ITER

pip install transformers==4.46.3
cd /iopsstor/scratch/cscs/yiswang/Megatron-mixtera/tools/checkpoint

torchrun --nproc_per_node 1 --nnodes 1 --master_addr localhost --master_port 1234 \
     --node_rank 0  convert.py --model-type GPT --loader mcore --saver legacy \
     --load-dir $LOAD_DIR \
     --save-dir $SAVE_DIR \
     --megatron-path /iopsstor/scratch/cscs/yiswang/Megatron-mixtera

pip uninstall --yes transformers
pip install transformers

cd /iopsstor/scratch/cscs/yiswang/Megatron-mixtera

python convert_megatron_to_huggingface.py \
    --path_to_checkpoint "$SAVE_DIR/iter_$ITER/mp_rank_00/model_optim_rng.pt" \
    --output "$SAVE_DIR/iter_$ITER/huggingface_model"

# transformers: PYTHONPATH=/users/yiswang/scratch/Megatron-mixtera/ python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py  --path_to_checkpoint /iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/ckpt-converted/iter_0030000/mp_rank_00/model_optim_rng.pt 

python eval_lmeval.py \
    --checkpoint-dir "$SAVE_DIR/iter_$ITER/huggingface_model" \
    --output-dir "$SAVE_DIR/iter_$ITER/huggingface_eval" \
    --tasks "lambada_openai,hellaswag,openbookqa,arc_easy"