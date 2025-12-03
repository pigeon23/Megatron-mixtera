#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=/iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/pretrain_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/pretrain_%j.err
#SBATCH --environment=mixtera
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --nodes=16
#SBATCH --account=a-infra02
#SBATCH --time=11:59:00

# CUDA_LAUNCH_BLOCKING=1 
# TORCH_DISTRIBUTED_DEBUG=DETAIL

# decorating main function with: from torch.distributed.elastic.multiprocessing.errors import record

# # Set maximum stack size to unlimited
# ulimit -s unlimited
# # Set max number of open file descriptors to a high value
# ulimit -n 65536
# Allow core dumps for post-mortem debugging
ulimit -c 0

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export TRITON_HOME=/dev/shm/ 

export CUDA_CACHE_DISABLE=1 

# export LOGLEVEL=DEBUG
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NET="AWS Libfabric"

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_NCHANNELS_PER_NET_PEER=4


export FI_EFA_FORK_SAFE=1
export NCCL_IB_TIMEOUT=22         # Increases retry timeout significantly
export TORCH_DISTRIBUTED_TIMEOUT=5400 # 90 minutes
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_RETRY_CNT=7

pushd /iopsstor/scratch/cscs/yiswang/mixtera && pip install -e . && popd
pip install megatron-core
pip install --no-build-isolation transformer-engine[pytorch]

rm -rf /iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/ckpt
rm -rf /iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/tensorboard
rm -rf /iopsstor/scratch/cscs/yiswang/Megatron-mixtera/experiments/adolog
pushd /iopsstor/scratch/cscs/yiswang/Megatron-mixtera


export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
export MASTER_PORT=1234   # Choose an unused port
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

# export CUDA_DEVICE_MAX_CONNECTIONS=1

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
    --num-layers 24 
    --hidden-size 2048
    --ffn-hidden-size 5464
    --num-attention-heads 16
    --seq-length 2048 
    --max-position-embeddings 2048 
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --normalization RMSNorm
    --swiglu
    --attention-dropout 0
    --hidden-dropout 0
    # --no-bias-swiglu-fusion
    # --no-bias-dropout-fusion
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

TRAINING_ARGS=(
    --micro-batch-size 32
    # --global-batch-size 64 
    # --rampup-batch-size 16 16 5859375 
    --train-iters 30000
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 0.001 # 6.0e-5 
    --lr-decay-style WSD 
    --lr-wsd-decay-style linear
    --lr-wsd-decay-iters 3000
    --lr-warmup-iters 500
    # --lr-decay-iters 10000
    # --lr-warmup-iters 500
    --lr-warmup-init 0
    # --min-lr 6.0e-6
    # --lr-warmup-fraction .001 
    # --lr-decay-iters 430000 
    # --optimizer-cpu-offload
    # --use-precision-aware-optimizer
    # --use-distributed-optimizer
    # --main-grads-dtype bf16
    # --exp-avg-dtype bf16
    # --exp-avg-sq-dtype bf16
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
    # --use-torch-fsdp2
    # --no-gradient-accumulation-fusion
    --distributed-timeout-minutes 30
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 6
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --dataloader-type mixtera # mixtera
    --num-workers 0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 2500
    --eval-interval 100000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --record-memory-history
    --log-memory-to-tensorboard
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