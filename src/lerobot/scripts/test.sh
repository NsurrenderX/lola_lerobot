#!/bin/bash
# LoLA 多卡分布式训练测试脚本
# 使用 LoLAPretrainStreamingDataset 进行训练

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate lerobot-gcr3

# 基础训练参数
STRATEGY="fsdp"
DEVICES=2
NUM_NODES=1
BATCH_SIZE=2
MAX_STEPS=10000
LEARNING_RATE=2.5e-5
PRECISION="bf16-mixed"
LOG_EVERY_N_STEPS=10
SAVE_INTERVAL=100

# 数据集参数
DATASET_ROOT="/data_6t_1/lerobot-v30/merged_0422_sub1/"
SUB_ROOT=""
DATASET_TO_EPISODES_PATH=""

# 模型参数
VLM_PATH="/data_16T/deepseek/qwen3_5/Qwen3.5-4B/"
ACTION_DIM=14
ACTION_CHUNK_SIZE=10
PRED_CHUNK_SIZE=50
N_OBS_STEPS=1

# 历史action加载参数
MAX_HISTORY_LENGTH=1024
HISTORY_PADDING_SIDE="left"

# 解码参数
NUM_WORKERS=8
BUFFER_SIZE=5000
DECODE_NUM_THREADS=2
DECODE_DEVICE="cpu"
STREAMING_SEED=42

# Pretrain 参数
PRETRAIN=true
TEMP_PROCESS=true
EPISODE_CHUNK_SIZE=8

# 运行训练
cmd="torchrun --nproc_per_node=${DEVICES} src/lerobot/scripts/train_lola_multigpu_stream.py \
    --dataset_root ${DATASET_ROOT} \
    --strategy ${STRATEGY} \
    --devices ${DEVICES} \
    --num_nodes ${NUM_NODES} \
    --batch_size ${BATCH_SIZE} \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --precision ${PRECISION} \
    --log_every_n_steps ${LOG_EVERY_N_STEPS} \
    --save_every_n_steps ${SAVE_INTERVAL} \
    --vlm_path ${VLM_PATH} \
    --action_dim ${ACTION_DIM} \
    --action_chunk_size ${ACTION_CHUNK_SIZE} \
    --pred_chunk_size ${PRED_CHUNK_SIZE} \
    --n_obs_steps ${N_OBS_STEPS} \
    --max_history_length ${MAX_HISTORY_LENGTH} \
    --history_padding_side ${HISTORY_PADDING_SIDE} \
    --num_workers ${NUM_WORKERS} \
    --buffer_size ${BUFFER_SIZE} \
    --decode_num_threads ${DECODE_NUM_THREADS} \
    --decode_device ${DECODE_DEVICE} \
    --streaming_seed ${STREAMING_SEED}"

# Pretrain 模式
if [ "$PRETRAIN" = true ]; then
    cmd="${cmd} --pretrain"
fi
if [ "$TEMP_PROCESS" = true ]; then
    cmd="${cmd} --temp_process"
fi
if [ "$EPISODE_CHUNK_SIZE" != 8 ]; then
    cmd="${cmd} --episode_chunk_size ${EPISODE_CHUNK_SIZE}"
fi

# Pretrain 数据集参数
if [ -n "$SUB_ROOT" ]; then
    cmd="${cmd} --sub_root ${SUB_ROOT}"
fi
if [ -n "$DATASET_TO_EPISODES_PATH" ]; then
    cmd="${cmd} --dataset_to_episodes_path ${DATASET_TO_EPISODES_PATH}"
fi

echo "Running: $cmd"
eval $cmd
