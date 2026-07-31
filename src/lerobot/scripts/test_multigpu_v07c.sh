#!/bin/bash
# LoLA V07-C 多卡分布式训练测试脚本（Cosmos3-Nano Reasoner VLM 变体）
# 与 test_multigpu_v07.sh 的唯一区别: VLM backbone 从 Qwen3.5-4B 换成 Cosmos3-Nano
# 注意: Cosmos3 reasoner (hidden 4096, 36层) 参数量约为 Qwen3.5-4B 的 2 倍,
#       如 OOM 请先降低 BATCH_SIZE; train_vlm 解冻后显存开销也会显著增大

eval "$(conda shell.bash hook)"
conda activate lerobot-gcr3

# 基础训练参数
STRATEGY="deepspeed"
DEVICES=2
NUM_NODES=1
BATCH_SIZE=4
MAX_STEPS=""
MAX_EPOCHS=10
LEARNING_RATE=2.5e-5
PRECISION="bf16-mixed"
LOG_EVERY_N_STEPS=10
SAVE_INTERVAL=''
SAVE_EVERY_N_EPOCHS="1"

# 数据集参数
DATASET_REPO_ID="calvin_task_ABC_D_training_v4"
DATASET_ROOT="/data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v4/"

# 模型参数
VLM_BACKBONE="cosmos3_nano"
VLM_PATH="/data_6t_1/cosmos3/Cosmos3-Nano/"
ACTION_DIM=7
ACTION_CHUNK_SIZE=8
PRED_CHUNK_SIZE=40
N_OBS_STEPS=1

# LoLA 模型配置
TRAIN_VLM=false
VLM_LR=1e-6
VLM_EXTRACT_LAYERS="8 16 24"
# VLM 桥接器: transformer = LolaVLMContextBridge (降维2048+8层Transformer, ~0.63B, 替代1.77B legacy 方阵)
VLM_BRIDGE_MODE="transformer"
VLM_BRIDGE_WIDTH=2048
VLM_BRIDGE_LAYERS=8
GRADIENT_CHECKPOINTING=false
COMPILE_MODEL=false
COMPILE_MODE="max-autotune"
MAX_IMAGE_PIXELS=230400
MIN_IMAGE_PIXELS=65536
NUM_INFERENCE_STEPS=10
GRIPPER_DIMS="-1"
ACTION_LOSS_WEIGHT=10.0
GRIPPER_LOSS_WEIGHT=1.0
HIST_ACTION_TOKEN_DROP_RATE=0.2

# LoLA V07: Bottleneck dimensions
ACTION_BOTTLENECK_DIM=128
GRIP_BOTTLENECK_DIM=64
STATE_BOTTLENECK_DIM=128
STATE_GRIP_BOTTLENECK_DIM=64
ENCODER_LR_MULT=1.5
WARMUP_PCT=0.1

# V2: Text template + completed tasks + transition masking
TASK_TEXT_TEMPLATE_VERSION="v1_with_completed"
COMPLETED_TASKS_USE_ANN=true
COMPLETED_TASKS_HISTORY_LEN=10
TRANSITION_MASK_RATE=0.8
MAX_TRANSITION_LEN=64

# VLM dynamic unfreezing parameters
VLM_UNFREEZE_V_LOSS_THRESHOLD=0.3
VLM_LR_MULT=1.5

# Special tokens
USE_SPECIAL_TOKENS=true

CKPT_DIR="/data_16T/deepseek/checkpoints/lola_v07c"

# 历史 action 加载参数
LOAD_FULL_HISTORY=true
MAX_HISTORY_LENGTH=1024
HISTORY_PADDING_SIDE="left"
HISTORY_TYPE="state"
STATE_DIM="7"
STATE_ENCODER_MODE="unified"
USE_STATE_CONDITION=false

# DataLoader 参数
NUM_WORKERS=8

# DeepSpeed 参数
DEEPSPEED_CONFIG=""
DEEPSPEED_REDUCE_BUCKET_SIZE=5e7
DEEPSPEED_ALLGATHER_BUCKET_SIZE=5e7

# Static padding 参数
STATIC_COLLATE_PADDING=true
STATIC_VLM_PADDING=false
VLM_MAX_LENGTH=""

# 归一化参数 (default=LoLA默认MEAN_STD, robovlm=min-max→[-1,1]全IDENTITY, zscore=arm=z-score/gripper=二值化{0,1})
NORM_MODE="zscore"
NORM_MIN=-0.65
NORM_MAX=0.65
# Stats模式 (original=annotation-only stats, incremental=包含所有Calvin帧的增量stats)
STATS_MODE="original"

# 运行训练
cmd="torchrun --nproc_per_node=${DEVICES} src/lerobot/scripts/train_lola_v07_multigpu.py \
    --dataset_repo_id ${DATASET_REPO_ID} \
    --dataset_root ${DATASET_ROOT} \
    --strategy ${STRATEGY} \
    --devices ${DEVICES} \
    --num_nodes ${NUM_NODES} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --precision ${PRECISION} \
    --log_every_n_steps ${LOG_EVERY_N_STEPS} \
    --vlm_backbone ${VLM_BACKBONE} \
    --vlm_path ${VLM_PATH} \
    --action_dim ${ACTION_DIM} \
    --action_chunk_size ${ACTION_CHUNK_SIZE} \
    --pred_chunk_size ${PRED_CHUNK_SIZE} \
    --n_obs_steps ${N_OBS_STEPS} \
    --max_history_length ${MAX_HISTORY_LENGTH} \
    --history_padding_side ${HISTORY_PADDING_SIDE} \
    --num_workers ${NUM_WORKERS} \
    --vlm_extract_layers ${VLM_EXTRACT_LAYERS} \
    --vlm_bridge_mode ${VLM_BRIDGE_MODE} \
    --vlm_bridge_width ${VLM_BRIDGE_WIDTH} \
    --vlm_bridge_layers ${VLM_BRIDGE_LAYERS} \
    --max_image_pixels ${MAX_IMAGE_PIXELS} \
    --min_image_pixels ${MIN_IMAGE_PIXELS} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --gripper_dims ${GRIPPER_DIMS} \
    --action_loss_weight ${ACTION_LOSS_WEIGHT} \
    --gripper_loss_weight ${GRIPPER_LOSS_WEIGHT} \
    --hist_action_token_drop_rate ${HIST_ACTION_TOKEN_DROP_RATE} \
    --action_bottleneck_dim ${ACTION_BOTTLENECK_DIM} \
    --grip_bottleneck_dim ${GRIP_BOTTLENECK_DIM} \
    --state_bottleneck_dim ${STATE_BOTTLENECK_DIM} \
    --state_grip_bottleneck_dim ${STATE_GRIP_BOTTLENECK_DIM} \
    --encoder_lr_mult ${ENCODER_LR_MULT} \
    --warmup_pct ${WARMUP_PCT} \
    --vlm_unfreeze_v_loss_threshold ${VLM_UNFREEZE_V_LOSS_THRESHOLD} \
    --vlm_lr_mult ${VLM_LR_MULT} \
    --task_text_template_version ${TASK_TEXT_TEMPLATE_VERSION} \
    --transition_mask_rate ${TRANSITION_MASK_RATE} \
    --max_transition_len ${MAX_TRANSITION_LEN} \
    --ckpt_dir ${CKPT_DIR} \
    --norm_mode ${NORM_MODE} \
    --norm_min ${NORM_MIN} \
    --norm_max ${NORM_MAX} \
    --stats_mode ${STATS_MODE} \
    --deepspeed_reduce_bucket_size ${DEEPSPEED_REDUCE_BUCKET_SIZE} \
    --deepspeed_allgather_bucket_size ${DEEPSPEED_ALLGATHER_BUCKET_SIZE}"

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
cmd="${cmd} --history_type ${HISTORY_TYPE} --state_encoder_mode ${STATE_ENCODER_MODE}"
if [ "$USE_STATE_CONDITION" = true ]; then
    cmd="${cmd} --use_state_condition"
fi
if [ -n "$STATE_DIM" ]; then
    cmd="${cmd} --state_dim ${STATE_DIM}"
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

# Special tokens
if [ "$USE_SPECIAL_TOKENS" = true ]; then
    cmd="${cmd} --use_special_tokens"
fi

# V2: completed tasks 参数
if [ "$COMPLETED_TASKS_USE_ANN" = false ]; then
    cmd="${cmd} --no_completed_tasks_use_ann"
fi
cmd="${cmd} --completed_tasks_history_len ${COMPLETED_TASKS_HISTORY_LEN}"

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