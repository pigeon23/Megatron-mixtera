#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=/iopsstor/scratch/cscs/yiswang/server_logs/pretrain_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/yiswang/server_logs/pretrain_%j.err
#SBATCH --environment=mixtera
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --nodes=1
#SBATCH --account=a-infra02
#SBATCH --time=01:29:00


pushd /iopsstor/scratch/cscs/yiswang/mixtera && pip install -e . && popd
pip install megatron-core
pip install --no-build-isolation transformer-engine[pytorch]

rm -rf /iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/ckpt/* 
rm -rf /iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/tensorboard/*
pushd /iopsstor/scratch/cscs/yiswang/Megatron-mixtera

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_NET="AWS Libfabric"

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
export MASTER_PORT=1234   # Choose an unused port
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
# MASTER_ADDR=localhost
# MASTER_PORT=1234
NUM_NODES=$SLURM_NNODES
# NODE_RANK=0
# WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES)) 

CHECKPOINT_PATH="experiments/ckpt/" # $1 #<Specify path>
TENSORBOARD_LOGS_PATH="experiments/tensorboard/" # $2 #<Specify path>
VOCAB_FILE="experiments/gpt_vocab.json" # $3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="experiments/gpt_merges.txt" # $4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH="processed_data/processed_data_text_document" # $5 #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --node_rank \$SLURM_NODEID
)

GPT_MODEL_ARGS=(
    --num-layers 12 
    --hidden-size 512 
    --num-attention-heads 8
    --seq-length 4096 
    --max-position-embeddings 4096 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 32
    # --global-batch-size 32 
    # --rampup-batch-size 16 16 5859375 
    --train-iters 500
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size 2
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --dataloader-type mixtera
    --num-workers 0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 25 
    --eval-interval 100 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

echo "[sbatch-master] running on $(hostname)"

echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"


CMD="
# print current environment variables
echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

export node_rank=\$SLURM_NODEID
cd /iopsstor/scratch/cscs/yiswang/Megatron-mixtera

torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_gpt.py \
    "${GPT_MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}"
    "

srun bash -c "$CMD"

# bash ./examples/gpt3/train_gpt3_175b_distributed.sh experiments/ckpt/ experiments/tensorboard/ experiments/gpt_vocab.json experiments/gpt_merges.txt processed_data/processed_data_text_document