#!/bin/bash
# LoLA Azure 分布式训练脚本
#
# 此脚本用于在 Azure ML 上运行分布式训练。
# Azure ML 会为每个节点运行一次此脚本，并自动传入以下参数：
#   --nnodes: 节点数量
#   --nproc_per_node: 每个节点的 GPU 数量
#   --node_rank: 当前节点的 rank
#   --master_addr: 主节点 IP
#   --master_port: 主节点端口
#
# 使用方法:
#   bash test_azure_stream.sh --nnodes $NODES --nproc_per_node $GPUS \
#       --node_rank $AZUREML_CR_NODE_RANK \
#       --master_addr $AZ_BATCHAI_JOB_MASTER_NODE_IP \
#       --master_port 9901

set -e

# 环境变量设置
export OPENSSL_FIPS=0  # 禁用 FIPS 避免自检失败
export TOKENIZERS_PARALLELISM=false
# Add conda env lib to LD_LIBRARY_PATH so torchcodec can find ffmpeg shared libs
# Also ensures conda's newer libstdc++ is used (avoids CXXABI_1.3.15 not found error)
if [ -d "/opt/conda/envs/lerobot/lib" ]; then
    export LD_LIBRARY_PATH="/opt/conda/envs/lerobot/lib:${LD_LIBRARY_PATH:-}"
fi

# Ensure kernel cache directory exists (avoids "Specified kernel cache directory
# could not be created" warning from torch.cuda on first CUDA JIT compile)
mkdir -p /root/.cache/torch/kernels

# ----------------------------------------------------------------------
# CUDA / GPU Compatibility Diagnostic
# For Blackwell (B200, sm_100) GPUs, ensure CUDA 12.8 runtime libraries
# are installed. The "no kernel image" error means nvidia-* pip packages
# are too old (12.6.x lacks sm_100 kernels).
# Reference: https://download.pytorch.org/whl/cu128
# ----------------------------------------------------------------------
PYTHON=/opt/conda/envs/lerobot/bin/python
if [ -x "$PYTHON" ]; then
    echo "=== CUDA Compatibility Check ==="
    TORCH_CUDA=$($PYTHON -c "import torch; print(torch.version.cuda)")
    GPU_CAP=$($PYTHON -c "import torch; cap=torch.cuda.get_device_capability(0); print(f'{cap[0]}.{cap[1]}')")
    CUBLAS_VER=$($PYTHON -c "import importlib.metadata as im; print(im.version('nvidia-cublas-cu12'))" 2>/dev/null || echo "NOT_FOUND")
    CUDNN_VER=$($PYTHON -c "import importlib.metadata as im; print(im.version('nvidia-cudnn-cu12'))" 2>/dev/null || echo "NOT_FOUND")
    echo "PyTorch CUDA: ${TORCH_CUDA}"
    echo "GPU Compute Capability: sm_${GPU_CAP}"
    echo "nvidia-cublas-cu12: ${CUBLAS_VER}"
    echo "nvidia-cudnn-cu12: ${CUDNN_VER}"

    # Check for Blackwell + mismatched runtime
    SM_MAJOR=$(echo "$GPU_CAP" | cut -d. -f1)
    if [ "$SM_MAJOR" -ge 10 ] 2>/dev/null; then
        if ! echo "${TORCH_CUDA}" | grep -q "^12.8"; then
            echo "ERROR: Blackwell GPU requires CUDA 12.8, got torch CUDA ${TORCH_CUDA}"
            echo "  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu128"
            exit 1
        fi
        if ! echo "${CUBLAS_VER}" | grep -q "^12\.8"; then
            echo "ERROR: nvidia-cublas-cu12 is ${CUBLAS_VER}, needs 12.8.x for Blackwell"
            echo "  Fix: pip install --force-reinstall nvidia-cublas-cu12==12.8.4.1"
            exit 1
        fi
        if ! echo "${CUDNN_VER}" | grep -q "^9\.10"; then
            echo "ERROR: nvidia-cudnn-cu12 is ${CUDNN_VER}, needs 9.10.x for Blackwell"
            echo "  Fix: pip install --force-reinstall nvidia-cudnn-cu12==9.10.2.21"
            exit 1
        fi
    fi
    echo "=== CUDA Check Passed ==="
fi

# ----------------------------------------------------------------------
# 默认参数（可被命令行参数覆盖）
# ----------------------------------------------------------------------
# Azure 分布式参数
NNODES=1
NPROC_PER_NODE=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"  # 使用 IP 而非 localhost，避免 IPv6 问题
MASTER_PORT=29500

# 训练参数
STRATEGY="fsdp"
FSDP_SHARDING="full_shard"
BATCH_SIZE=4
MAX_STEPS=10000
LEARNING_RATE=2.5e-5
LOG_EVERY_N_STEPS=10
SAVE_INTERVAL=5000
GRADIENT_CLIP_VAL=1.0
DISABLE_GRADIENT_CHECKPOINTING=false  # Disable checkpointing to save bwd recomputation overhead (increases memory)
COMPILE_MODEL=false  # Enable torch.compile for DiT (kernel fusion, reduces kernel launch overhead)
COMPILE_MODE="reduce-overhead"  # torch.compile mode: "reduce-overhead" (CUDA graphs), "default", "max-autotune"

# 数据集参数
DATASET_REPO_ID=""
DATASET_ROOT="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/merged_0422_sub1"
SUB_ROOT=""              # pretrain 模式：子数据集根目录
DATASET_TO_EPISODES_PATH=""  # pretrain 模式：episode 映射 JSON 路径

# 模型参数
VLM_PATH="/mnt/wangxiaofa/qwen3_5/Qwen3.5-4B/"
CKPT_DIR="/mnt/wangxiaofa/checkpoints/lola-0th-pretrain"
TRAIN_VLM=false

# VLM 图像分辨率参数
MAX_IMAGE_PIXELS=230400  # max_h≈360p for 720p images
MIN_IMAGE_PIXELS=65536   # min 64 visual tokens per image

# 模型维度参数
ACTION_DIM=20
ACTION_CHUNK_SIZE=10
PRED_CHUNK_SIZE=50
N_OBS_STEPS=1

# 历史action加载参数
MAX_HISTORY_LENGTH=100
HISTORY_PADDING_SIDE="left"

# 解码参数
NO_DEFERRED=false
ASYNC_DECODE=false
DECODE_DEVICE="cpu"
DECODE_NUM_THREADS=2
BUFFER_SIZE=5000
STREAMING_SEED=42
NUM_WORKERS=8
PREFETCH_FACTOR=4
PREFETCH_QUEUE_SIZE=4
DATALOADER_TIMEOUT=600
NO_SHUFFLE=false

# Pretrain 参数
PRETRAIN=false
TEMP_PROCESS=false
EPISODE_CHUNK_SIZE=16

# Tier-based batching 参数
TIER_CONFIG_PATH=""
EFFECTIVE_BATCH_SIZE=2048
BALANCE_MODE="frame_weighted"
GPU_UTILIZATION_TARGET=0.92
GPU_MEMORY_BUDGET_GB=""
TIER_MICRO_BATCHES_OVERRIDE=""

# Wandb 参数
WANDB_PROJECT="lola-azure-pretrain"
WANDB_NAME=""
WANDB_ENTITY=""
WANDB_ID=""
DISABLE_WANDB=false

# Resume 参数
RESUME=""

# ----------------------------------------------------------------------
# 解析命令行参数
# ----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        # Azure 分布式参数
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;

        # 训练参数
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --fsdp_sharding)
            FSDP_SHARDING="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --log_every_n_steps)
            LOG_EVERY_N_STEPS="$2"
            shift 2
            ;;
        --save_every_n_steps)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --gradient_clip_val)
            GRADIENT_CLIP_VAL="$2"
            shift 2
            ;;
        --disable_gradient_checkpointing)
            DISABLE_GRADIENT_CHECKPOINTING=true
            shift
            ;;
        --compile_model)
            COMPILE_MODEL=true
            shift
            ;;
        --compile_mode)
            COMPILE_MODE="$2"
            shift 2
            ;;

        # 数据集参数
        --dataset_repo_id)
            DATASET_REPO_ID="$2"
            shift 2
            ;;
        --dataset_root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --sub_root)
            SUB_ROOT="$2"
            shift 2
            ;;
        --dataset_to_episodes_path)
            DATASET_TO_EPISODES_PATH="$2"
            shift 2
            ;;

        # 模型参数
        --vlm_path)
            VLM_PATH="$2"
            shift 2
            ;;
        --ckpt_dir)
            CKPT_DIR="$2"
            shift 2
            ;;
        --train_vlm)
            TRAIN_VLM=true
            shift
            ;;
        --max_image_pixels)
            MAX_IMAGE_PIXELS="$2"
            shift 2
            ;;
        --min_image_pixels)
            MIN_IMAGE_PIXELS="$2"
            shift 2
            ;;

        # 模型维度参数
        --action_dim)
            ACTION_DIM="$2"
            shift 2
            ;;
        --action_chunk_size)
            ACTION_CHUNK_SIZE="$2"
            shift 2
            ;;
        --pred_chunk_size)
            PRED_CHUNK_SIZE="$2"
            shift 2
            ;;
        --n_obs_steps)
            N_OBS_STEPS="$2"
            shift 2
            ;;

        # 历史action参数
        --max_history_length)
            MAX_HISTORY_LENGTH="$2"
            shift 2
            ;;
        --history_padding_side)
            HISTORY_PADDING_SIDE="$2"
            shift 2
            ;;

        # 解码参数
        --no_deferred)
            NO_DEFERRED=true
            shift
            ;;
        --async_decode)
            ASYNC_DECODE=true
            shift
            ;;
        --decode_device)
            DECODE_DEVICE="$2"
            shift 2
            ;;
        --decode_num_threads)
            DECODE_NUM_THREADS="$2"
            shift 2
            ;;
        --buffer_size)
            BUFFER_SIZE="$2"
            shift 2
            ;;
        --streaming_seed)
            STREAMING_SEED="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --prefetch_factor)
            PREFETCH_FACTOR="$2"
            shift 2
            ;;
        --prefetch_queue_size)
            PREFETCH_QUEUE_SIZE="$2"
            shift 2
            ;;
        --dataloader_timeout)
            DATALOADER_TIMEOUT="$2"
            shift 2
            ;;
        --buffer_size)
            BUFFER_SIZE="$2"
            shift 2
            ;;
        --no_shuffle)
            NO_SHUFFLE=true
            shift
            ;;

        # Pretrain 参数
        --pretrain)
            PRETRAIN=true
            shift
            ;;
        --temp_process)
            TEMP_PROCESS=true
            shift
            ;;
        --episode_chunk_size)
            EPISODE_CHUNK_SIZE="$2"
            shift 2
            ;;
        --tier_config_path)
            TIER_CONFIG_PATH="$2"
            shift 2
            ;;
        --effective_batch_size)
            EFFECTIVE_BATCH_SIZE="$2"
            shift 2
            ;;
        --balance_mode)
            BALANCE_MODE="$2"
            shift 2
            ;;
        --gpu_utilization_target)
            GPU_UTILIZATION_TARGET="$2"
            shift 2
            ;;
        --gpu_memory_budget_gb)
            GPU_MEMORY_BUDGET_GB="$2"
            shift 2
            ;;
        --tier_micro_batches_override)
            TIER_MICRO_BATCHES_OVERRIDE="$2"
            shift 2
            ;;

        # Wandb 参数
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --wandb_entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --wandb_id)
            WANDB_ID="$2"
            shift 2
            ;;
        --disable_wandb)
            DISABLE_WANDB=true
            shift
            ;;

        # Resume
        --resume)
            RESUME="$2"
            shift 2
            ;;

        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# 打印配置信息
echo "========================================"
echo "LoLA Azure Distributed Training"
echo "========================================"
echo "Distributed Config:"
echo "  - Nodes: ${NNODES}"
echo "  - GPUs per node: ${NPROC_PER_NODE}"
echo "  - Node rank: ${NODE_RANK}"
echo "  - Master addr: ${MASTER_ADDR}"
echo "  - Master port: ${MASTER_PORT}"
echo ""
echo "Training Config:"
echo "  - Strategy: ${STRATEGY}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Max steps: ${MAX_STEPS}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo "  - Gradient clip: ${GRADIENT_CLIP_VAL}"
echo "  - Dataset: ${DATASET_REPO_ID:-$DATASET_ROOT}"
echo "  - VLM path: ${VLM_PATH}"
echo "  - Pretrain: ${PRETRAIN}"
echo "  - Buffer size: ${BUFFER_SIZE}"
echo "  - Prefetch factor: ${PREFETCH_FACTOR}"
echo "  - Prefetch queue: ${PREFETCH_QUEUE_SIZE}"
echo "  - Disable gradient checkpointing: ${DISABLE_GRADIENT_CHECKPOINTING}"
echo "  - Compile model (torch.compile): ${COMPILE_MODEL} (mode: ${COMPILE_MODE})"
if [ -n "$TIER_CONFIG_PATH" ]; then
    echo "  - Tier config: ${TIER_CONFIG_PATH}"
    echo "  - Effective batch size: ${EFFECTIVE_BATCH_SIZE}"
    echo "  - Balance mode: ${BALANCE_MODE}"
    echo "  - GPU utilization target: ${GPU_UTILIZATION_TARGET}"
    [ -n "$GPU_MEMORY_BUDGET_GB" ] && echo "  - GPU memory budget (GB): ${GPU_MEMORY_BUDGET_GB}"
    [ -n "$TIER_MICRO_BATCHES_OVERRIDE" ] && echo "  - Tier micro-batches override: ${TIER_MICRO_BATCHES_OVERRIDE}"
fi
echo "========================================"

# ----------------------------------------------------------------------
# 构建训练命令
# ----------------------------------------------------------------------
if [ "$NNODES" -eq 1 ]; then
    cmd="/opt/conda/envs/lerobot/bin/torchrun --nproc_per_node=${NPROC_PER_NODE} \
        src/lerobot/scripts/train_lola_azure_stream.py"
else
    cmd="/opt/conda/envs/lerobot/bin/torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        src/lerobot/scripts/train_lola_azure_stream.py"
fi

# 通用训练参数
cmd="${cmd} \
    --strategy ${STRATEGY} \
    --fsdp_sharding ${FSDP_SHARDING} \
    --batch_size ${BATCH_SIZE} \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --log_every_n_steps ${LOG_EVERY_N_STEPS} \
    --save_every_n_steps ${SAVE_INTERVAL} \
    --gradient_clip_val ${GRADIENT_CLIP_VAL} \
    --vlm_path ${VLM_PATH} \
    --ckpt_dir ${CKPT_DIR} \
    --max_image_pixels ${MAX_IMAGE_PIXELS} \
    --min_image_pixels ${MIN_IMAGE_PIXELS} \
    --action_dim ${ACTION_DIM} \
    --action_chunk_size ${ACTION_CHUNK_SIZE} \
    --pred_chunk_size ${PRED_CHUNK_SIZE} \
    --n_obs_steps ${N_OBS_STEPS} \
    --max_history_length ${MAX_HISTORY_LENGTH} \
    --history_padding_side ${HISTORY_PADDING_SIDE} \
    --buffer_size ${BUFFER_SIZE} \
    --streaming_seed ${STREAMING_SEED} \
    --decode_device ${DECODE_DEVICE} \
    --decode_num_threads ${DECODE_NUM_THREADS} \
    --num_workers ${NUM_WORKERS} \
    --prefetch_factor ${PREFETCH_FACTOR} \
    --prefetch_queue_size ${PREFETCH_QUEUE_SIZE} \
    --dataloader_timeout ${DATALOADER_TIMEOUT} \
    --wandb_project ${WANDB_PROJECT}"

# 数据集参数
if [ -n "$DATASET_REPO_ID" ]; then
    cmd="${cmd} --dataset_repo_id ${DATASET_REPO_ID}"
else
    cmd="${cmd} --dataset_root ${DATASET_ROOT}"
fi

# Pretrain 模式参数
if [ "$PRETRAIN" = true ]; then
    cmd="${cmd} --pretrain"
fi
if [ -n "$SUB_ROOT" ]; then
    cmd="${cmd} --sub_root ${SUB_ROOT}"
fi
if [ -n "$DATASET_TO_EPISODES_PATH" ]; then
    cmd="${cmd} --dataset_to_episodes_path ${DATASET_TO_EPISODES_PATH}"
fi
if [ "$TEMP_PROCESS" = true ]; then
    cmd="${cmd} --temp_process"
fi
if [ "$EPISODE_CHUNK_SIZE" != 8 ]; then
    cmd="${cmd} --episode_chunk_size ${EPISODE_CHUNK_SIZE}"
fi

# Tier-based batching 参数
if [ -n "$TIER_CONFIG_PATH" ]; then
    cmd="${cmd} --tier_config_path ${TIER_CONFIG_PATH}"
    cmd="${cmd} --effective_batch_size ${EFFECTIVE_BATCH_SIZE}"
    cmd="${cmd} --balance_mode ${BALANCE_MODE}"
    cmd="${cmd} --gpu_utilization_target ${GPU_UTILIZATION_TARGET}"
    if [ -n "$GPU_MEMORY_BUDGET_GB" ]; then
        cmd="${cmd} --gpu_memory_budget_gb ${GPU_MEMORY_BUDGET_GB}"
    fi
    if [ -n "$TIER_MICRO_BATCHES_OVERRIDE" ]; then
        cmd="${cmd} --tier_micro_batches_override ${TIER_MICRO_BATCHES_OVERRIDE}"
    fi
fi

# 训练 VLM 参数
if [ "$TRAIN_VLM" = true ]; then
    cmd="${cmd} --train_vlm"
fi

# 解码参数
if [ "$DISABLE_GRADIENT_CHECKPOINTING" = true ]; then
    cmd="${cmd} --disable_gradient_checkpointing"
fi
if [ "$COMPILE_MODEL" = true ]; then
    cmd="${cmd} --compile_model"
    cmd="${cmd} --compile_mode ${COMPILE_MODE}"
fi
if [ "$NO_DEFERRED" = true ]; then
    cmd="${cmd} --no_deferred"
fi
if [ "$ASYNC_DECODE" = true ]; then
    cmd="${cmd} --async_decode"
fi
if [ "$NO_SHUFFLE" = true ]; then
    cmd="${cmd} --no_shuffle"
fi

# Wandb 参数
if [ -n "$WANDB_NAME" ]; then
    cmd="${cmd} --wandb_name ${WANDB_NAME}"
fi
if [ -n "$WANDB_ENTITY" ]; then
    cmd="${cmd} --wandb_entity ${WANDB_ENTITY}"
fi
if [ -n "$WANDB_ID" ]; then
    cmd="${cmd} --wandb_id ${WANDB_ID}"
fi
if [ "$DISABLE_WANDB" = true ]; then
    cmd="${cmd} --disable_wandb"
fi

# Resume 参数
if [ -n "$RESUME" ]; then
    cmd="${cmd} --resume ${RESUME}"
fi

echo "Running: $cmd"
eval $cmd

echo "Training completed!"
