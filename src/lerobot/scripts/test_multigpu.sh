#!/bin/bash
# LoLA 多卡分布式训练测试脚本（非流式）
# 使用 LoLADataset (--load_full_history) 进行训练

eval "$(conda shell.bash hook)"
conda activate lerobot-gcr3

# 基础训练参数
STRATEGY="fsdp"
DEVICES=2
NUM_NODES=1
BATCH_SIZE=2
MAX_STEPS=""
MAX_EPOCHS=10
LEARNING_RATE=2.5e-5
PRECISION="bf16-mixed"
LOG_EVERY_N_STEPS=10
SAVE_INTERVAL=''
SAVE_EVERY_N_EPOCHS="1"

# 数据集参数
DATASET_REPO_ID="calvin_task_ABC_D_training_v3"
DATASET_ROOT="/data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v3/"

# 模型参数
VLM_PATH="/data_16T/deepseek/qwen3_5/Qwen3.5-4B/"
ACTION_DIM=7
ACTION_CHUNK_SIZE=8
PRED_CHUNK_SIZE=40
N_OBS_STEPS=1

# LoLA 模型配置
TRAIN_VLM=false
VLM_LR=1e-6
VLM_EXTRACT_LAYERS="8 16 24"
GRADIENT_CHECKPOINTING=true
COMPILE_MODEL=false
COMPILE_MODE="max-autotune"
MAX_IMAGE_PIXELS=230400
MIN_IMAGE_PIXELS=65536
NUM_INFERENCE_STEPS=10
CKPT_DIR="/data_16T/deepseek/checkpoints/lola"

# 历史 action 加载参数
LOAD_FULL_HISTORY=true
MAX_HISTORY_LENGTH=1024
HISTORY_PADDING_SIDE="left"

# DataLoader 参数
NUM_WORKERS=8

# 归一化参数
NORM_MODE="robovlm"
NORM_MIN=-0.65
NORM_MAX=0.65

# 运行训练
cmd="torchrun --nproc_per_node=${DEVICES} src/lerobot/scripts/train_lola_multigpu.py \
    --dataset_repo_id ${DATASET_REPO_ID} \
    --dataset_root ${DATASET_ROOT} \
    --strategy ${STRATEGY} \
    --devices ${DEVICES} \
    --num_nodes ${NUM_NODES} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --precision ${PRECISION} \
    --log_every_n_steps ${LOG_EVERY_N_STEPS} \
    --vlm_path ${VLM_PATH} \
    --action_dim ${ACTION_DIM} \
    --action_chunk_size ${ACTION_CHUNK_SIZE} \
    --pred_chunk_size ${PRED_CHUNK_SIZE} \
    --n_obs_steps ${N_OBS_STEPS} \
    --max_history_length ${MAX_HISTORY_LENGTH} \
    --history_padding_side ${HISTORY_PADDING_SIDE} \
    --num_workers ${NUM_WORKERS} \
    --vlm_extract_layers ${VLM_EXTRACT_LAYERS} \
    --max_image_pixels ${MAX_IMAGE_PIXELS} \
    --min_image_pixels ${MIN_IMAGE_PIXELS} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --ckpt_dir ${CKPT_DIR} \
    --norm_mode ${NORM_MODE} \
    --norm_min ${NORM_MIN} \
    --norm_max ${NORM_MAX}"

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

if [ "$LOAD_FULL_HISTORY" = true ]; then
    cmd="${cmd} --load_full_history"
fi
if [ "$TRAIN_VLM" = true ]; then
    cmd="${cmd} --train_vlm --vlm_lr ${VLM_LR}"
fi
if [ "$GRADIENT_CHECKPOINTING" = false ]; then
    cmd="${cmd} --no_gradient_checkpointing"
fi
if [ "$COMPILE_MODEL" = true ]; then
    cmd="${cmd} --compile_model --compile_mode ${COMPILE_MODE}"
fi

echo "Running: $cmd"
eval $cmd