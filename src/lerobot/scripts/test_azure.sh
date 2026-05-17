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
#   bash test_azure.sh --nnodes $NODES --nproc_per_node $GPUS \
#       --node_rank $AZUREML_CR_NODE_RANK \
#       --master_addr $AZ_BATCHAI_JOB_MASTER_NODE_IP \
#       --master_port 9901

set -e

# 环境变量设置
export OPENSSL_FIPS=0  # 禁用 FIPS 避免自检失败
export TOKENIZERS_PARALLELISM=false
export PATH=/opt/conda/envs/lerobot/bin:$PATH
# Add conda env lib to LD_LIBRARY_PATH so torchcodec can find ffmpeg shared libs
# Also ensures conda's newer libstdc++ is used (avoids CXXABI_1.3.15 not found error)
if [ -d "/opt/conda/envs/lerobot/lib" ]; then
    export LD_LIBRARY_PATH="/opt/conda/envs/lerobot/lib:${LD_LIBRARY_PATH:-}"
fi

# Ensure kernel cache directory exists (avoids "Specified kernel cache directory
# could not be created" warning from torch.cuda on first CUDA JIT compile)
mkdir -p /root/.cache/torch/kernels

# ----------------------------------------------------------------------
# 默认参数（可被命令行参数覆盖）
# ----------------------------------------------------------------------
NNODES=1
NPROC_PER_NODE=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"  # 使用 IP 而非 localhost，避免 IPv6 问题
MASTER_PORT=29500

# 训练参数
STRATEGY="deepspeed"
BATCH_SIZE=4
MAX_STEPS=""
MAX_EPOCHS=10
LEARNING_RATE=2.5e-5
LOG_EVERY_N_STEPS=10
SAVE_INTERVAL=''
SAVE_EVERY_N_EPOCHS=""
GRADIENT_CLIP_VAL=1.0

# 数据集参数
DATASET_REPO_ID=""
DATASET_ROOT="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/simpler_bridge_v3"

# 模型参数
VLM_PATH="/mnt/wangxiaofa/qwen3_5/Qwen3.5-4B/"
CKPT_DIR="/mnt/wangxiaofa/checkpoints/lola-simpler"
TRAIN_VLM=false
ACTION_DIM=14
ACTION_CHUNK_SIZE=10
PRED_CHUNK_SIZE=50
N_OBS_STEPS=1

# 历史action加载参数
LOAD_FULL_HISTORY=true
MAX_HISTORY_LENGTH=1024
HISTORY_PADDING_SIDE="left"
HISTORY_TYPE="action"
STATE_DIM=""
STATE_ENCODER_MODE="unified"

# LoLA 模型配置
GRADIENT_CHECKPOINTING=true
COMPILE_MODEL=false
COMPILE_MODE="max-autotune"
VLM_LR=1e-6
VLM_EXTRACT_LAYERS="8 16 24"
MAX_IMAGE_PIXELS=230400
MIN_IMAGE_PIXELS=65536
NUM_INFERENCE_STEPS=10
GRIPPER_DIMS="-1"
ACTION_LOSS_WEIGHT=1.0
GRIPPER_LOSS_WEIGHT=1.0
HIST_ACTION_TOKEN_DROP_RATE=0.0

# 归一化参数 (default=LoLA默认MEAN_STD, robovlm=min-max→[-1,1]全IDENTITY, zscore=arm=z-score/gripper=二值化{0,1})
NORM_MODE="zscore"
NORM_MIN=-0.65
NORM_MAX=0.65

# Wandb 参数
WANDB_PROJECT="lola-azure-calvin"
WANDB_NAME=""
WANDB_ENTITY=""
DISABLE_WANDB=false

# DataLoader 参数
NUM_WORKERS=8

# DeepSpeed 参数
DEEPSPEED_CONFIG=""
DEEPSPEED_ZERO_STAGE=2
DEEPSPEED_REDUCE_BUCKET_SIZE=5e7
DEEPSPEED_ALLGATHER_BUCKET_SIZE=5e7

# Static padding 参数
STATIC_COLLATE_PADDING=true
STATIC_VLM_PADDING=false
VLM_MAX_LENGTH=""

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
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
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
        --save_every_n_epochs)
            SAVE_EVERY_N_EPOCHS="$2"
            shift 2
            ;;
        --gradient_clip_val)
            GRADIENT_CLIP_VAL="$2"
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
        --load_full_history)
            LOAD_FULL_HISTORY=true
            shift
            ;;
        --no_load_full_history)
            LOAD_FULL_HISTORY=false
            shift
            ;;
        --max_history_length)
            MAX_HISTORY_LENGTH="$2"
            shift 2
            ;;
        --history_padding_side)
            HISTORY_PADDING_SIDE="$2"
            shift 2
            ;;

        # 历史类型参数
        --history_type)
            HISTORY_TYPE="$2"
            shift 2
            ;;
        --state_dim)
            STATE_DIM="$2"
            shift 2
            ;;
        --state_encoder_mode)
            STATE_ENCODER_MODE="$2"
            shift 2
            ;;

        # LoLA 模型配置参数
        --no_gradient_checkpointing)
            GRADIENT_CHECKPOINTING=false
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
        --vlm_lr)
            VLM_LR="$2"
            shift 2
            ;;
        --vlm_extract_layers)
            VLM_EXTRACT_LAYERS="$2"
            shift 2
            ;;
        --max_image_pixels)
            MAX_IMAGE_PIXELS="$2"
            shift 2
            ;;
        --min_image_pixels)
            MIN_IMAGE_PIXELS="$2"
            shift 2
            ;;
        --num_inference_steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --gripper_dims)
            GRIPPER_DIMS="$2"
            shift 2
            ;;
        --gripper_loss_weight)
            GRIPPER_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --action_loss_weight)
            ACTION_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --hist_action_token_drop_rate)
            HIST_ACTION_TOKEN_DROP_RATE="$2"
            shift 2
            ;;

        # 归一化参数
        --norm_mode)
            NORM_MODE="$2"
            shift 2
            ;;
        --norm_min)
            NORM_MIN="$2"
            shift 2
            ;;
        --norm_max)
            NORM_MAX="$2"
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
        --disable_wandb)
            DISABLE_WANDB=true
            shift
            ;;

        # Resume
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --deepspeed_config)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --deepspeed_zero_stage)
            DEEPSPEED_ZERO_STAGE="$2"
            shift 2
            ;;
        --deepspeed_reduce_bucket_size)
            DEEPSPEED_REDUCE_BUCKET_SIZE="$2"
            shift 2
            ;;
        --deepspeed_allgather_bucket_size)
            DEEPSPEED_ALLGATHER_BUCKET_SIZE="$2"
            shift 2
            ;;
        --no_static_collate_padding)
            STATIC_COLLATE_PADDING=false
            shift
            ;;
        --static_vlm_padding)
            STATIC_VLM_PADDING=true
            shift
            ;;
        --vlm_max_length)
            VLM_MAX_LENGTH="$2"
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
echo "  - World size: ${WORLD_SIZE}"
echo "  - Node rank: ${NODE_RANK}"
echo "  - Master addr: ${MASTER_ADDR}"
echo "  - Master port: ${MASTER_PORT}"
echo ""
echo "Training Config:"
echo "  - Strategy: ${STRATEGY}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Max steps: ${MAX_STEPS:-N/A}"
echo "  - Max epochs: ${MAX_EPOCHS:-N/A}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo "  - Gradient clip: ${GRADIENT_CLIP_VAL}"
echo "  - Norm mode: ${NORM_MODE}"
echo "  - Dataset: ${DATASET_REPO_ID:-$DATASET_ROOT}"
echo "  - VLM path: ${VLM_PATH}"
echo "  - DeepSpeed config: ${DEEPSPEED_CONFIG:-default}"
echo "  - DeepSpeed ZeRO stage: ${DEEPSPEED_ZERO_STAGE}"
echo "========================================"

# ----------------------------------------------------------------------
# 启动训练
# 使用 torchrun 来管理多 GPU，每个节点运行一次
# 单节点时使用简化命令，多节点时使用完整参数
# ----------------------------------------------------------------------
if [ "$NNODES" -eq 1 ]; then
    # 单节点：使用简化的 torchrun 命令
    cmd="/opt/conda/envs/lerobot/bin/torchrun --nproc_per_node=${NPROC_PER_NODE} \
        src/lerobot/scripts/train_lola_azure.py \
        --strategy ${STRATEGY} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --log_every_n_steps ${LOG_EVERY_N_STEPS} \
        --gradient_clip_val ${GRADIENT_CLIP_VAL} \
        --vlm_path ${VLM_PATH} \
        --ckpt_dir ${CKPT_DIR} \
        --action_dim ${ACTION_DIM} \
        --action_chunk_size ${ACTION_CHUNK_SIZE} \
        --pred_chunk_size ${PRED_CHUNK_SIZE} \
        --n_obs_steps ${N_OBS_STEPS} \
        --vlm_extract_layers ${VLM_EXTRACT_LAYERS} \
        --max_image_pixels ${MAX_IMAGE_PIXELS} \
        --min_image_pixels ${MIN_IMAGE_PIXELS} \
        --num_inference_steps ${NUM_INFERENCE_STEPS} \
        --gripper_dims ${GRIPPER_DIMS} \
        --action_loss_weight ${ACTION_LOSS_WEIGHT} \
        --gripper_loss_weight ${GRIPPER_LOSS_WEIGHT} \
        --hist_action_token_drop_rate ${HIST_ACTION_TOKEN_DROP_RATE} \
        --num_workers ${NUM_WORKERS} \
        --norm_mode ${NORM_MODE} \
        --norm_min ${NORM_MIN} \
        --norm_max ${NORM_MAX} \
        --deepspeed_reduce_bucket_size ${DEEPSPEED_REDUCE_BUCKET_SIZE} \
        --deepspeed_allgather_bucket_size ${DEEPSPEED_ALLGATHER_BUCKET_SIZE} \
        --deepspeed_zero_stage ${DEEPSPEED_ZERO_STAGE} \
        --wandb_project ${WANDB_PROJECT}"
else
    # 多节点：使用完整的分布式参数
    cmd="/opt/conda/envs/lerobot/bin/torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        src/lerobot/scripts/train_lola_azure.py \
        --strategy ${STRATEGY} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --log_every_n_steps ${LOG_EVERY_N_STEPS} \
        --gradient_clip_val ${GRADIENT_CLIP_VAL} \
        --vlm_path ${VLM_PATH} \
        --ckpt_dir ${CKPT_DIR} \
        --action_dim ${ACTION_DIM} \
        --action_chunk_size ${ACTION_CHUNK_SIZE} \
        --pred_chunk_size ${PRED_CHUNK_SIZE} \
        --n_obs_steps ${N_OBS_STEPS} \
        --vlm_extract_layers ${VLM_EXTRACT_LAYERS} \
        --max_image_pixels ${MAX_IMAGE_PIXELS} \
        --min_image_pixels ${MIN_IMAGE_PIXELS} \
        --num_inference_steps ${NUM_INFERENCE_STEPS} \
        --gripper_dims ${GRIPPER_DIMS} \
        --action_loss_weight ${ACTION_LOSS_WEIGHT} \
        --gripper_loss_weight ${GRIPPER_LOSS_WEIGHT} \
        --hist_action_token_drop_rate ${HIST_ACTION_TOKEN_DROP_RATE} \
        --num_workers ${NUM_WORKERS} \
        --norm_mode ${NORM_MODE} \
        --norm_min ${NORM_MIN} \
        --norm_max ${NORM_MAX} \
        --deepspeed_reduce_bucket_size ${DEEPSPEED_REDUCE_BUCKET_SIZE} \
        --deepspeed_allgather_bucket_size ${DEEPSPEED_ALLGATHER_BUCKET_SIZE} \
        --deepspeed_zero_stage ${DEEPSPEED_ZERO_STAGE} \
        --wandb_project ${WANDB_PROJECT}"
fi

# 训练终止条件参数（二选一）
if [ -n "$MAX_STEPS" ]; then
    cmd="${cmd} --max_steps ${MAX_STEPS}"
elif [ -n "$MAX_EPOCHS" ]; then
    cmd="${cmd} --max_epochs ${MAX_EPOCHS}"
fi

# 保存间隔参数
if [ -n "$SAVE_INTERVAL" ]; then
    cmd="${cmd} --save_every_n_steps ${SAVE_INTERVAL}"
fi
if [ -n "$SAVE_EVERY_N_EPOCHS" ]; then
    cmd="${cmd} --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS}"
fi

# 数据集参数
if [ -n "$DATASET_REPO_ID" ]; then
    cmd="${cmd} --dataset_repo_id ${DATASET_REPO_ID}"
else
    cmd="${cmd} --dataset_root ${DATASET_ROOT}"
fi

# 历史action参数
if [ "$LOAD_FULL_HISTORY" = true ]; then
    cmd="${cmd} --load_full_history --max_history_length ${MAX_HISTORY_LENGTH} --history_padding_side ${HISTORY_PADDING_SIDE}"
fi

# 历史类型参数
cmd="${cmd} --history_type ${HISTORY_TYPE} --state_encoder_mode ${STATE_ENCODER_MODE}"
if [ -n "$STATE_DIM" ]; then
    cmd="${cmd} --state_dim ${STATE_DIM}"
fi

# 训练 VLM 参数
if [ "$TRAIN_VLM" = true ]; then
    cmd="${cmd} --train_vlm --vlm_lr ${VLM_LR}"
fi

# 梯度检查点 & compile
if [ "$GRADIENT_CHECKPOINTING" = false ]; then
    cmd="${cmd} --no_gradient_checkpointing"
fi
if [ "$COMPILE_MODEL" = true ]; then
    cmd="${cmd} --compile_model --compile_mode ${COMPILE_MODE}"
fi

# Wandb 参数
if [ -n "$WANDB_NAME" ]; then
    cmd="${cmd} --wandb_name ${WANDB_NAME}"
fi
if [ -n "$WANDB_ENTITY" ]; then
    cmd="${cmd} --wandb_entity ${WANDB_ENTITY}"
fi
if [ "$DISABLE_WANDB" = true ]; then
    cmd="${cmd} --disable_wandb"
fi

# Resume 参数
if [ -n "$RESUME" ]; then
    cmd="${cmd} --resume ${RESUME}"
fi

# DeepSpeed 参数
if [ -n "$DEEPSPEED_CONFIG" ]; then
    cmd="${cmd} --deepspeed_config ${DEEPSPEED_CONFIG}"
fi

# Static padding 参数
if [ "$STATIC_COLLATE_PADDING" = false ]; then
    cmd="${cmd} --no_static_collate_padding"
fi
if [ "$STATIC_VLM_PADDING" = true ]; then
    cmd="${cmd} --static_vlm_padding"
fi
if [ -n "$VLM_MAX_LENGTH" ]; then
    cmd="${cmd} --vlm_max_length ${VLM_MAX_LENGTH}"
fi

echo "Running: $cmd"
eval $cmd

echo "Training completed!"
