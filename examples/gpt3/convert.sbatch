#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=/iopsstor/scratch/cscs/yiswang/server_logs/convert_%j.out
#SBATCH --environment=mixtera
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --nodes=1
#SBATCH --account=a-infra02
#SBATCH --time=00:59:00

export NCCL_NET="AWS Libfabric"
export CUDA_DEVICE_MAX_CONNECTIONS=1

pip install megatron-core
pip install --no-build-isolation transformer-engine[pytorch]

load_dir="/iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/ckpt" 
save_dir="$load_dir/ckpt-converted" 

pip install transformers==4.46.3
cd /iopsstor/scratch/cscs/yiswang/Megatron-mixtera/tools/checkpoint

torchrun --nproc_per_node 1 --nnodes 1 --master_addr localhost --master_port 1234 \
     --node_rank 0  convert.py --model-type GPT --loader mcore --saver legacy \
     --load-dir $load_dir \
     --save-dir $save_dir \
     --megatron-path /iopsstor/scratch/cscs/yiswang/Megatron-mixtera

pip uninstall transformers
pip install transformers

cd /iopsstor/scratch/cscs/yiswang/Megatron-mixtera

python convert_megatron_to_huggingface.py \
    --path_to_checkpoint experiments/24-11-2025/ckpt-converted/iter_0030000/mp_rank_00/model_optim_rng.pt \
    --output experiments/24-11-2025/huggingface

# transformers: PYTHONPATH=/users/yiswang/scratch/Megatron-mixtera/ python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py  --path_to_checkpoint /iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/ckpt-converted/iter_0030000/mp_rank_00/model_optim_rng.pt 

python eval_lmeval.py --checkpoint-dir experiments/ --output-dir experiments/24-11-2025/huggingface_eval 