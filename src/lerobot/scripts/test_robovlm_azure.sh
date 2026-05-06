#!/bin/bash
# RoboVLM Azure 分布式微调脚本
#
# 基于 Kosmos-2 + LSTMDecoder 的 RoboVLM 模型微调。
# 参数与 train_robovlm.py 的 argparse 对齐。
#
# 使用方法:
#   # 单 GPU 测试
#   bash test_robovlm_azure.sh --dataset_repo_id <repo_id> --dataset_root <path>
#
#   # 多 GPU (torchrun)
#   bash test_robovlm_azure.sh --nproc_per_node 4 \
#       --dataset_repo_id <repo_id> --dataset_root <path>
#
#   # Azure 多节点
#   bash test_robovlm_azure.sh --nnodes $NODES --nproc_per_node $GPUS \
#       --node_rank $AZUREML_CR_NODE_RANK \
#       --master_addr $AZ_BATCHAI_JOB_MASTER_NODE_IP \
#       --master_port 9901 \
#       --dataset_repo_id <repo_id> --dataset_root <path>

set -e

# 环境变量设置
export OPENSSL_FIPS=0
export TOKENIZERS_PARALLELISM=false
if [ -d "/opt/conda/envs/lerobot/lib" ]; then
    export LD_LIBRARY_PATH="/opt/conda/envs/lerobot/lib:${LD_LIBRARY_PATH:-}"
fi

# ----------------------------------------------------------------------
# CUDA / GPU Compatibility Diagnostic
# For Blackwell (B200, sm_100) GPUs, ensure CUDA 12.8 runtime libraries
# are installed.
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
# 默认参数（与 train_robovlm.py argparse 对齐）
# ----------------------------------------------------------------------

# --- 分布式参数 ---
NNODES=1
NPROC_PER_NODE=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

# --- 训练参数 ---
STRATEGY="fsdp"
FSDP_SHARDING="full_shard"
BATCH_SIZE=4
MAX_STEPS=""              # 与 MAX_EPOCHS 互斥
MAX_EPOCHS=""             # 与 MAX_STEPS 互斥
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.0
LOG_EVERY_N_STEPS=10
GRADIENT_CLIP_VAL=1.0
NUM_WORKERS=8

# --- 数据集参数 ---
DATASET_REPO_ID=""
DATASET_ROOT=""

# --- 模型参数 ---
VLM_PRETRAINED_PATH="/data_16T/deepseek/kosmos-2-patch14-224/"
CKPT_DIR="/mnt/wangxiaofa/checkpoints/robovlm-finetune"
SAVE_EVERY_N_STEPS=500
SAVE_EVERY_N_EPOCHS=""
FREEZE_BACKBONE=false
NO_TRAIN_VISION=false
NO_TRAIN_TEXT_EMBEDDING=false
USE_STATE=false
NO_USE_STATE=false
USE_HAND_RGB=true
NO_USE_HAND_RGB=false
SKIP_ACTION_NORMALIZE=false    # false = 应用 normalize_action (对应 --no_skip_action_normalize)
WINDOW_SIZE=8
FWD_PRED_NEXT_N=10

# --- 调度器参数 ---
SCHEDULER="constant"       # "constant" or "cosine"
SCHEDULER_WARMUP_STEPS=250
ACTION_HEAD_LR_SCALE=5.0   # LSTMDecoder / action_token LR = learning_rate * this value
ARM_GRIPPER_LOSS_RATIO=1.0 # Weight for gripper loss relative to arm loss

# --- Wandb 参数 ---
WANDB_PROJECT="robovlm"
WANDB_NAME=""
WANDB_ENTITY=""
DISABLE_WANDB=false

# --- 预训练/Resume ---
PRETRAINED_CHECKPOINT=""
RESUME=""

# ----------------------------------------------------------------------
# 解析命令行参数
# ----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        # 分布式参数
        --nnodes)          NNODES="$2"; shift 2 ;;
        --nproc_per_node)  NPROC_PER_NODE="$2"; shift 2 ;;
        --node_rank)       NODE_RANK="$2"; shift 2 ;;
        --master_addr)     MASTER_ADDR="$2"; shift 2 ;;
        --master_port)     MASTER_PORT="$2"; shift 2 ;;

        # 训练参数
        --strategy)          STRATEGY="$2"; shift 2 ;;
        --fsdp_sharding)     FSDP_SHARDING="$2"; shift 2 ;;
        --batch_size)        BATCH_SIZE="$2"; shift 2 ;;
        --max_steps)         MAX_STEPS="$2"; MAX_EPOCHS=""; shift 2 ;;
        --max_epochs)        MAX_EPOCHS="$2"; MAX_STEPS=""; shift 2 ;;
        --learning_rate)     LEARNING_RATE="$2"; shift 2 ;;
        --weight_decay)      WEIGHT_DECAY="$2"; shift 2 ;;
        --log_every_n_steps) LOG_EVERY_N_STEPS="$2"; shift 2 ;;
        --save_every_n_steps) SAVE_EVERY_N_STEPS="$2"; shift 2 ;;
        --save_every_n_epochs) SAVE_EVERY_N_EPOCHS="$2"; shift 2 ;;
        --gradient_clip_val) GRADIENT_CLIP_VAL="$2"; shift 2 ;;
        --num_workers)       NUM_WORKERS="$2"; shift 2 ;;

        # 数据集参数
        --dataset_repo_id)  DATASET_REPO_ID="$2"; shift 2 ;;
        --dataset_root)     DATASET_ROOT="$2"; shift 2 ;;

        # 模型参数
        --vlm_pretrained_path)   VLM_PRETRAINED_PATH="$2"; shift 2 ;;
        --ckpt_dir)              CKPT_DIR="$2"; shift 2 ;;
        --freeze_backbone)       FREEZE_BACKBONE=true; shift ;;
        --no_train_vision)       NO_TRAIN_VISION=true; shift ;;
        --no_train_text_embedding) NO_TRAIN_TEXT_EMBEDDING=true; shift ;;
        --use_state)             USE_STATE=true; shift ;;
        --no_use_state)          NO_USE_STATE=true; shift ;;
        --use_hand_rgb)          USE_HAND_RGB=true; shift ;;
        --no_use_hand_rgb)       NO_USE_HAND_RGB=true; shift ;;
        --skip_action_normalize) SKIP_ACTION_NORMALIZE=true; shift ;;
        --no_skip_action_normalize) SKIP_ACTION_NORMALIZE=false; shift ;;
        --window_size)           WINDOW_SIZE="$2"; shift 2 ;;
        --fwd_pred_next_n)       FWD_PRED_NEXT_N="$2"; shift 2 ;;

        # 调度器参数
        --scheduler)              SCHEDULER="$2"; shift 2 ;;
        --scheduler_warmup_steps) SCHEDULER_WARMUP_STEPS="$2"; shift 2 ;;
        --action_head_lr_scale)   ACTION_HEAD_LR_SCALE="$2"; shift 2 ;;
        --arm_gripper_loss_ratio) ARM_GRIPPER_LOSS_RATIO="$2"; shift 2 ;;

        # Wandb 参数
        --wandb_project)   WANDB_PROJECT="$2"; shift 2 ;;
        --wandb_name)      WANDB_NAME="$2"; shift 2 ;;
        --wandb_entity)    WANDB_ENTITY="$2"; shift 2 ;;
        --disable_wandb)   DISABLE_WANDB=true; shift ;;

        # 预训练/Resume
        --pretrained_checkpoint) PRETRAINED_CHECKPOINT="$2"; shift 2 ;;
        --resume)                RESUME="$2"; shift 2 ;;

        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------
# 打印配置信息
# ----------------------------------------------------------------------
echo "========================================"
echo "RoboVLM Azure Distributed Fine-tuning"
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
if [ -n "$MAX_STEPS" ]; then
    echo "  - Max steps: ${MAX_STEPS} (step-based mode)"
elif [ -n "$MAX_EPOCHS" ]; then
    echo "  - Max epochs: ${MAX_EPOCHS} (epoch-based mode)"
fi
echo "  - Learning rate: ${LEARNING_RATE}"
echo "  - Weight decay: ${WEIGHT_DECAY}"
echo "  - Gradient clip: ${GRADIENT_CLIP_VAL}"
echo "  - Scheduler: ${SCHEDULER} (warmup: ${SCHEDULER_WARMUP_STEPS} steps)"
echo "  - Action head LR scale: ${ACTION_HEAD_LR_SCALE}"
echo "  - Arm gripper loss ratio: ${ARM_GRIPPER_LOSS_RATIO}"
echo "  - Save every N steps: ${SAVE_EVERY_N_STEPS}"
if [ -n "$SAVE_EVERY_N_EPOCHS" ]; then
    echo "  - Save every N epochs: ${SAVE_EVERY_N_EPOCHS}"
fi
echo "  - VLM path: ${VLM_PRETRAINED_PATH}"
echo "  - Window size: ${WINDOW_SIZE}"
echo "  - Fwd pred next n: ${FWD_PRED_NEXT_N}"
echo "  - Freeze backbone: ${FREEZE_BACKBONE}"
echo "  - Train vision: $([ "$NO_TRAIN_VISION" = true ] && echo false || echo true)"
echo "  - Train text embedding: $([ "$NO_TRAIN_TEXT_EMBEDDING" = true ] && echo false || echo true)"
echo "  - Use state: $([ "$NO_USE_STATE" = true ] && echo false || echo ${USE_STATE})"
echo "  - Use hand rgb: $([ "$NO_USE_HAND_RGB" = true ] && echo false || echo ${USE_HAND_RGB})"
echo "  - Skip action normalize: ${SKIP_ACTION_NORMALIZE}"
if [ -n "$PRETRAINED_CHECKPOINT" ]; then
    echo "  - Pretrained checkpoint: ${PRETRAINED_CHECKPOINT}"
fi
echo "  - Dataset: ${DATASET_REPO_ID:-$DATASET_ROOT}"
echo "========================================"

# ----------------------------------------------------------------------
# 构建训练命令
# ----------------------------------------------------------------------
if [ "$NNODES" -eq 1 ]; then
    cmd="/opt/conda/envs/lerobot/bin/torchrun --nproc_per_node=${NPROC_PER_NODE} \
        src/lerobot/scripts/train_robovlm.py"
else
    cmd="/opt/conda/envs/lerobot/bin/torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        src/lerobot/scripts/train_robovlm.py"
fi

# 通用训练参数
cmd="${cmd} \
    --strategy ${STRATEGY} \
    --fsdp_sharding ${FSDP_SHARDING} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --log_every_n_steps ${LOG_EVERY_N_STEPS} \
    --save_every_n_steps ${SAVE_EVERY_N_STEPS} \
    --gradient_clip_val ${GRADIENT_CLIP_VAL} \
    --num_workers ${NUM_WORKERS} \
    --vlm_pretrained_path ${VLM_PRETRAINED_PATH} \
    --ckpt_dir ${CKPT_DIR} \
    --window_size ${WINDOW_SIZE} \
    --fwd_pred_next_n ${FWD_PRED_NEXT_N} \
    --scheduler ${SCHEDULER} \
    --scheduler_warmup_steps ${SCHEDULER_WARMUP_STEPS} \
    --action_head_lr_scale ${ACTION_HEAD_LR_SCALE} \
    --arm_gripper_loss_ratio ${ARM_GRIPPER_LOSS_RATIO} \
    --wandb_project ${WANDB_PROJECT}"

# 训练模式：步数 vs Epoch（互斥）
if [ -n "$MAX_STEPS" ]; then
    cmd="${cmd} --max_steps ${MAX_STEPS}"
elif [ -n "$MAX_EPOCHS" ]; then
    cmd="${cmd} --max_epochs ${MAX_EPOCHS}"
fi

# Epoch 保存间隔
if [ -n "$SAVE_EVERY_N_EPOCHS" ]; then
    cmd="${cmd} --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS}"
fi

# 数据集参数
if [ -n "$DATASET_REPO_ID" ]; then
    cmd="${cmd} --dataset_repo_id ${DATASET_REPO_ID}"
fi
if [ -n "$DATASET_ROOT" ]; then
    cmd="${cmd} --dataset_root ${DATASET_ROOT}"
fi

# 模型开关参数
if [ "$FREEZE_BACKBONE" = true ]; then
    cmd="${cmd} --freeze_backbone"
fi
if [ "$NO_TRAIN_VISION" = true ]; then
    cmd="${cmd} --no_train_vision"
fi
if [ "$NO_TRAIN_TEXT_EMBEDDING" = true ]; then
    cmd="${cmd} --no_train_text_embedding"
fi
if [ "$USE_STATE" = true ] && [ "$NO_USE_STATE" = false ]; then
    cmd="${cmd} --use_state"
fi
if [ "$NO_USE_STATE" = true ]; then
    cmd="${cmd} --no_use_state"
fi
if [ "$NO_USE_HAND_RGB" = true ]; then
    cmd="${cmd} --no_use_hand_rgb"
fi
if [ "$SKIP_ACTION_NORMALIZE" = true ]; then
    cmd="${cmd} --skip_action_normalize"
else
    cmd="${cmd} --no_skip_action_normalize"
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

# 预训练/Resume
if [ -n "$PRETRAINED_CHECKPOINT" ]; then
    cmd="${cmd} --pretrained_checkpoint ${PRETRAINED_CHECKPOINT}"
fi
if [ -n "$RESUME" ]; then
    cmd="${cmd} --resume ${RESUME}"
fi

echo "Running: $cmd"
eval $cmd

echo "Training completed!"
