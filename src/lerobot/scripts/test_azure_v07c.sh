#!/bin/bash
# LoLA Azure 分布式训练脚本 (V07-C: Cosmos3-Nano Reasoner VLM 变体)
#
# 与 test_azure_v07.sh 的唯一区别: VLM backbone 从 Qwen3.5-4B 换成 Cosmos3-Nano。
# 前置条件: 需先将 Cosmos3-Nano 模型文件同步到 Azure 存储 (VLM_PATH 指向的位置),
#           建议预转换 reasoner-only 权重以加快每个 rank 的加载速度。
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
#   bash test_azure_v07c.sh --nnodes $NODES --nproc_per_node $GPUS \
#       --node_rank $AZUREML_CR_NODE_RANK \
#       --master_addr $AZ_BATCHAI_JOB_MASTER_NODE_IP \
#       --master_port 9901

set -e

# 环境变量设置
export OPENSSL_FIPS=0  # 禁用 FIPS 避免自检失败
export TOKENIZERS_PARALLELISM=false
# 消除 allocator 碎片导致的虚高显存水位 (ZeRO-3 分片下大量小碎片场景效果显著)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# /home/aiscuser/.conda/envs/lerobot/bin/ for gcr
export PATH=/home/aiscuser/.conda/envs/lerobot/bin/:/opt/conda/envs/lerobot/bin:$PATH
# conda run --name lerobot which python
# Add conda env lib to LD_LIBRARY_PATH so torchcodec can find ffmpeg shared libs
# Also ensures conda's newer libstdc++ is used (avoids CXXABI_1.3.15 not found error)
if [ -d "/home/aiscuser/.conda/envs/lerobot/lib" ]; then
    export LD_LIBRARY_PATH="/home/aiscuser/.conda/envs/lerobot/lib:${LD_LIBRARY_PATH:-}"
fi

# if [ -d "/opt/conda/envs/lerobot/lib" ]; then
#     export LD_LIBRARY_PATH="/opt/conda/envs/lerobot/lib:${LD_LIBRARY_PATH:-}"
# fi

# Ensure kernel cache directory exists (avoids "Specified kernel cache directory
# could not be created" warning from torch.cuda on first CUDA JIT compile)
# mkdir -p /root/.cache/torch/kernels

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
VLM_BACKBONE="cosmos3_nano"
VLM_PATH="/mnt/wangxiaofa/cosmos3/Cosmos3-Nano/"  # 需先同步模型文件到此路径
CKPT_DIR="/mnt/wangxiaofa/checkpoints/lola-v07c"
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
USE_STATE_CONDITION=false

# LoLA 模型配置
GRADIENT_CHECKPOINTING=true
COMPILE_MODEL=false
COMPILE_MODE="max-autotune"
VLM_LR=1e-6
VLM_EXTRACT_LAYERS="8 16 24"
# VLM 桥接器: transformer = LolaVLMContextBridge (降维2048+8层Transformer, ~0.63B, 替代1.77B legacy 方阵)
VLM_BRIDGE_MODE="transformer"
VLM_BRIDGE_WIDTH=2048
VLM_BRIDGE_LAYERS=8
MAX_IMAGE_PIXELS=230400
MIN_IMAGE_PIXELS=65536
NUM_INFERENCE_STEPS=10
GRIPPER_DIMS="-1"
ACTION_LOSS_WEIGHT=10.0
GRIPPER_LOSS_WEIGHT=1.0
HIST_ACTION_TOKEN_DROP_RATE=0.0

# LoLA V07: Bottleneck dimensions
ACTION_BOTTLENECK_DIM=256
GRIP_BOTTLENECK_DIM=128
STATE_BOTTLENECK_DIM=256
STATE_GRIP_BOTTLENECK_DIM=128
ENCODER_LR_MULT=1.5
WARMUP_PCT=0.1

# V2: Text template + completed tasks + transition masking
TASK_TEXT_TEMPLATE_VERSION="raw"
COMPLETED_TASKS_USE_ANN=true
COMPLETED_TASKS_HISTORY_LEN=5
TRANSITION_MASK_RATE=0.0
MAX_TRANSITION_LEN=64

# VLM dynamic unfreezing parameters
VLM_UNFREEZE_V_LOSS_THRESHOLD=0.3
VLM_LR_MULT=1.5

# Special tokens
USE_SPECIAL_TOKENS=false

# 归一化参数 (default=LoLA默认MEAN_STD, robovlm=min-max→[-1,1]全IDENTITY, zscore=arm=z-score/gripper=二值化{0,1})
NORM_MODE="zscore"
NORM_MIN=-0.65
NORM_MAX=0.65
# Stats模式 (original=annotation-only stats, incremental=包含所有Calvin帧的增量stats)
STATS_MODE="original"

# Wandb 参数
WANDB_PROJECT="lola-azure-calvin"
WANDB_NAME=""
WANDB_ENTITY=""
DISABLE_WANDB=false

# DataLoader 参数
NUM_WORKERS=8

# DeepSpeed 参数
DEEPSPEED_CONFIG=""
# ZeRO-3: 参数/梯度/优化器全分片, 40GB 级显卡全解冻训练 8B VLM 的必需项
# (ZeRO-2 参数+梯度全量复制, 静态即 ~39GB, 40GB 卡放不下; 80GB+ 卡可改回 2)
DEEPSPEED_ZERO_STAGE=3
DEEPSPEED_REDUCE_BUCKET_SIZE=5e7
DEEPSPEED_ALLGATHER_BUCKET_SIZE=5e7

# Static padding 参数
STATIC_COLLATE_PADDING=true
STATIC_VLM_PADDING=false
VLM_MAX_LENGTH=""

# Resume 参数
RESUME=""

# ----------------------------------------------------------------------
# 本地化 IO: 数据集/ckpt 走节点本地 NVMe, blob 只做异步持久化
# (blobfuse 挂载点网络波动时 torch.save 大文件会 IO 错误, 曾崩 ZeRO-3 分片保存)
# 开启后: 启动时 azcopy 把数据集/VLM/resume ckpt 拉到 LOCAL_MIRROR; 训练只读写
# 本地盘; 后台 checkpoint_upload_watcher 把带 .upload_ready 标记的 ckpt 异步传回 blob。
# 以下均可由同名命令行参数覆盖 (--storage_account 等), 账户名不写进仓库,
# 由启动命令透传 (AMLT job 命令行或环境变量)。
# ----------------------------------------------------------------------
LOCALIZE_IO=${LOCALIZE_IO:-true}
STORAGE_ACCOUNT=${STORAGE_ACCOUNT:-""}      # LOCALIZE_IO=true 时必填 (--storage_account 透传)
STORAGE_CONTAINER=${STORAGE_CONTAINER:-""}  # LOCALIZE_IO=true 时必填 (--storage_container 透传)
MOUNT_PREFIX=${MOUNT_PREFIX:-/mnt/wangxiaofa}
LOCAL_MIRROR=${LOCAL_MIRROR:-/scratch/lola_mirror}
LOCALIZE_VLM=${LOCALIZE_VLM:-true}          # false 则 VLM 权重不从 blob 预下载, 启动时直接读挂载点
UPLOAD_KEEP_LAST=${UPLOAD_KEEP_LAST:-2}          # 本地保留的已上传 ckpt 数
UPLOAD_DRAIN_TIMEOUT=${UPLOAD_DRAIN_TIMEOUT:-7200}  # 训练结束后等上传排空的最长秒数

# ----------------------------------------------------------------------
# 解析命令行参数
# ----------------------------------------------------------------------
# 保留原始参数: resume 搜索模式的 helper (resume_search.py) 需要用与训练进程
# 完全相同的参数重建当前配置快照 (launcher 私有 flag 会被 parse_known_args 忽略)
LAUNCH_ARGS=("$@")
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
        --vlm_backbone)
            VLM_BACKBONE="$2"
            shift 2
            ;;
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
        --use_state_condition)
            USE_STATE_CONDITION=true
            shift
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
        --vlm_bridge_mode)
            VLM_BRIDGE_MODE="$2"
            shift 2
            ;;
        --vlm_bridge_width)
            VLM_BRIDGE_WIDTH="$2"
            shift 2
            ;;
        --vlm_bridge_layers)
            VLM_BRIDGE_LAYERS="$2"
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

        # V07: Bottleneck dimensions
        --action_bottleneck_dim)
            ACTION_BOTTLENECK_DIM="$2"
            shift 2
            ;;
        --grip_bottleneck_dim)
            GRIP_BOTTLENECK_DIM="$2"
            shift 2
            ;;
        --state_bottleneck_dim)
            STATE_BOTTLENECK_DIM="$2"
            shift 2
            ;;
        --state_grip_bottleneck_dim)
            STATE_GRIP_BOTTLENECK_DIM="$2"
            shift 2
            ;;
        --encoder_lr_mult)
            ENCODER_LR_MULT="$2"
            shift 2
            ;;
        --warmup_pct)
            WARMUP_PCT="$2"
            shift 2
            ;;

        # V2: Text template + completed tasks + transition masking
        --task_text_template_version)
            TASK_TEXT_TEMPLATE_VERSION="$2"
            shift 2
            ;;
        --no_completed_tasks_use_ann)
            COMPLETED_TASKS_USE_ANN=false
            shift
            ;;
        --completed_tasks_history_len)
            COMPLETED_TASKS_HISTORY_LEN="$2"
            shift 2
            ;;
        --transition_mask_rate)
            TRANSITION_MASK_RATE="$2"
            shift 2
            ;;
        --max_transition_len)
            MAX_TRANSITION_LEN="$2"
            shift 2
            ;;
        --vlm_unfreeze_v_loss_threshold)
            VLM_UNFREEZE_V_LOSS_THRESHOLD="$2"
            shift 2
            ;;
        --vlm_lr_mult)
            VLM_LR_MULT="$2"
            shift 2
            ;;
        --use_special_tokens)
            USE_SPECIAL_TOKENS=true
            shift
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
        --stats_mode)
            STATS_MODE="$2"
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

        # Localize IO
        --localize_io)
            LOCALIZE_IO=true
            shift
            ;;
        --no_localize_io)
            LOCALIZE_IO=false
            shift
            ;;
        --storage_account)
            STORAGE_ACCOUNT="$2"
            shift 2
            ;;
        --storage_container)
            STORAGE_CONTAINER="$2"
            shift 2
            ;;
        --mount_prefix)
            MOUNT_PREFIX="$2"
            shift 2
            ;;
        --local_mirror)
            LOCAL_MIRROR="$2"
            shift 2
            ;;
        --no_localize_vlm)
            LOCALIZE_VLM=false
            shift
            ;;
        --upload_keep_last)
            UPLOAD_KEEP_LAST="$2"
            shift 2
            ;;
        --upload_drain_timeout)
            UPLOAD_DRAIN_TIMEOUT="$2"
            shift 2
            ;;

        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------
# 本地化 IO 前置阶段: blob -> 节点本地 NVMe (训练开始前)
# ----------------------------------------------------------------------
# 规范化: 统一剥掉路径变量的结尾斜杠, 否则 _to_local 等拼接会产生 //
# 双斜杠路径, 导致下载落点与训练读取路径不一致 (meta/info.json 找不到)
MOUNT_PREFIX="${MOUNT_PREFIX%/}"
LOCAL_MIRROR="${LOCAL_MIRROR%/}"
DATASET_ROOT="${DATASET_ROOT%/}"
VLM_PATH="${VLM_PATH%/}"
CKPT_DIR="${CKPT_DIR%/}"
RESUME="${RESUME%/}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHER_PID=""

_rel_under_mount() { local p="${1%/}"; echo "${p#$MOUNT_PREFIX/}"; }
_to_local() { echo "$LOCAL_MIRROR/$(_rel_under_mount "$1")"; }
_to_blob() { echo "https://${STORAGE_ACCOUNT}.blob.core.windows.net/${STORAGE_CONTAINER}/$(_rel_under_mount "$1")"; }
_is_mount_path() { [ "${1#$MOUNT_PREFIX/}" != "$1" ]; }

# 下载落地结构可观测性: 若训练仍报文件找不到, 从 ls 输出可直接看出 azcopy 实际落点
_show_layout() {
    local label="$1" dir="$2"
    echo "[localize] ${label} 落地结构: $dir"
    if [ -d "$dir" ]; then
        ls -la "$dir" 2>&1 | sed 's/^/    /'
    else
        echo "    ⚠️ 目录不存在!"
    fi
}

_azcopy_transfer() {
    /home/aiscuser/.conda/envs/lerobot/bin/python "${SCRIPTS_DIR}/download_azure_azcopy.py" \
        --account "${STORAGE_ACCOUNT}" --container "${STORAGE_CONTAINER}" \
        --mount-prefix "${MOUNT_PREFIX}" --azcopy-path "${LOCAL_MIRROR}/bin/azcopy" "$@"
}

if [ "$LOCALIZE_IO" = true ]; then
    if [ -z "$STORAGE_ACCOUNT" ] || [ -z "$STORAGE_CONTAINER" ]; then
        echo "ERROR: LOCALIZE_IO=true 需要 --storage_account 和 --storage_container (或同名环境变量)"
        exit 1
    fi
    mkdir -p "${LOCAL_MIRROR}/bin"
    echo "========================================"
    echo "Localized IO: blob -> node-local NVMe mirror"
    echo "  - Mirror root: ${LOCAL_MIRROR}"
    echo "  - Blob base: https://${STORAGE_ACCOUNT}.blob.core.windows.net/${STORAGE_CONTAINER}"
    echo "========================================"

    # 1. 数据集本地化 (幂等: ifSourceNewer 跳过已有文件, 抢占重启后增量补齐)
    #    --dir: 目录传输, 抵消 azcopy 把源目录名嵌套进目标路径的语义
    if [ -n "$DATASET_ROOT" ] && _is_mount_path "$DATASET_ROOT"; then
        LOCAL_DATASET_ROOT=$(_to_local "$DATASET_ROOT")
        echo "[localize] dataset: $DATASET_ROOT -> $LOCAL_DATASET_ROOT"
        _azcopy_transfer --download "$DATASET_ROOT" "$LOCAL_DATASET_ROOT" --dir
        _show_layout "dataset" "$LOCAL_DATASET_ROOT"
        if [ -f "$LOCAL_DATASET_ROOT/meta/info.json" ]; then
            echo "[localize] ✅ meta/info.json 已就位"
        else
            echo "[localize] ⚠️ meta/info.json 未在预期位置: $LOCAL_DATASET_ROOT/meta/info.json (实际落点见上方 ls)"
        fi
        DATASET_ROOT="$LOCAL_DATASET_ROOT"
    fi

    # 2. VLM 权重本地化 (--no_localize_vlm 可跳过: 权重仅启动时读一次,
    #    直接从挂载点读的失败风险限于启动阶段, 重试成本低)
    if [ "$LOCALIZE_VLM" = true ] && [ -n "$VLM_PATH" ] && _is_mount_path "$VLM_PATH"; then
        LOCAL_VLM_PATH=$(_to_local "$VLM_PATH")
        echo "[localize] VLM: $VLM_PATH -> $LOCAL_VLM_PATH"
        _azcopy_transfer --download "$VLM_PATH" "$LOCAL_VLM_PATH" --dir
        _show_layout "VLM" "$LOCAL_VLM_PATH"
        VLM_PATH="$LOCAL_VLM_PATH"
    elif [ -n "$VLM_PATH" ] && _is_mount_path "$VLM_PATH"; then
        echo "[localize] VLM 本地化已跳过 (LOCALIZE_VLM=false), 启动时直接读挂载点: $VLM_PATH"
    fi

    # 3. resume checkpoint 拉回本地 (ZeRO-3 只需本节点 ranks 的分片;
    #    client_state 合并于每个 rank 的 model_states, peek 不受分片过滤影响)
    if [ -n "$RESUME" ] && _is_mount_path "$RESUME"; then
        RESUME_LOCAL=$(_to_local "$RESUME")
        RESUME_BLOB=$(_to_blob "$RESUME")
        RESUME_BASE=$(basename "${RESUME%/}")
        mkdir -p "$RESUME_LOCAL"
        SHARD_PATTERNS=""
        START_RANK=$((NODE_RANK * NPROC_PER_NODE))
        END_RANK=$((START_RANK + NPROC_PER_NODE - 1))
        for ((i=START_RANK; i<=END_RANK; i++)); do
            SHARD_PATTERNS="${SHARD_PATTERNS}${SHARD_PATTERNS:+;}*zero_pp_rank_${i}_mp_rank_00_*"
        done
        if [[ "$RESUME_BASE" == step_* || "$RESUME_BASE" == final ]]; then
            # 直接指向 tag 目录
            RESUME_TAG_DIR_LOCAL="$RESUME_LOCAL"
            RESUME_TAG_DIR_BLOB="$RESUME_BLOB"
        else
            # run 目录或 run 集合目录: 统一下载元数据 (latest + training_config.json,
            # 均为小文件, 一次拉完; 集合目录下每个 run 各一份)
            echo "[localize] resume: fetch metadata (latest + training_config.json) from $RESUME_BLOB"
            _azcopy_transfer --download "$RESUME_BLOB" "$RESUME_LOCAL" --include-pattern "latest;training_config.json" --overwrite true --dir
            _show_layout "resume metadata" "$RESUME_LOCAL"
            if [ -f "${RESUME_LOCAL}/latest" ]; then
                : # run 目录: latest 指针已在元数据下载中就位
            elif ! compgen -G "${RESUME_LOCAL}/*/training_config.json" > /dev/null; then
                echo "[localize] resume: $RESUME_BLOB 下未找到任何 checkpoint 元数据 (新实验或路径无内容) — 从头开始训练 (resume 已禁用)"
                RESUME=""
            else
                # 集合目录 → 搜索模式: 按训练配置匹配 run, 选 latest 步数最多者
                # (helper stdout 仅输出选中的 run 目录名, 候选表在 stderr; 空 = 无匹配)
                echo "[localize] resume 搜索模式: 在 $RESUME_LOCAL 下按训练配置匹配 run"
                RUN_NAME=$(/home/aiscuser/.conda/envs/lerobot/bin/python "${SCRIPTS_DIR}/resume_search.py" \
                    --resolve_parent "$RESUME_LOCAL" \
                    --local_dataset_root "${LOCAL_DATASET_ROOT:-$DATASET_ROOT}" \
                    --world_size "$((NNODES * NPROC_PER_NODE))" \
                    -- "${LAUNCH_ARGS[@]}")
                if [ -n "$RUN_NAME" ]; then
                    echo "[localize] resume 搜索命中: ${RUN_NAME}"
                    RESUME_LOCAL="${RESUME_LOCAL}/${RUN_NAME}"
                    RESUME_BLOB="${RESUME_BLOB}/${RUN_NAME}"
                else
                    echo "[localize] resume 搜索: 未找到配置匹配的 checkpoint — 从头开始训练 (resume 已禁用)"
                    RESUME=""
                fi
            fi
            if [ -n "$RESUME" ]; then
                # run 目录: 读出 latest 指针指向的 tag
                RESUME_TAG=$(cat "${RESUME_LOCAL}/latest")
                RESUME_TAG_DIR_LOCAL="${RESUME_LOCAL}/${RESUME_TAG}"
                RESUME_TAG_DIR_BLOB="${RESUME_BLOB}/${RESUME_TAG}"
            fi
        fi
        if [ -n "$RESUME" ]; then
            echo "[localize] resume ckpt: $RESUME_TAG_DIR_BLOB -> $RESUME_TAG_DIR_LOCAL (shard filter: ranks ${START_RANK}-${END_RANK})"
            _azcopy_transfer --download "$RESUME_TAG_DIR_BLOB" "$RESUME_TAG_DIR_LOCAL" --include-pattern "$SHARD_PATTERNS" --dir
            _show_layout "resume tag" "$RESUME_TAG_DIR_LOCAL"
            # 校验本节点分片齐全, 缺失则全量重下兜底
            SHARDS_OK=true
            for ((i=START_RANK; i<=END_RANK; i++)); do
                if ! compgen -G "${RESUME_TAG_DIR_LOCAL}/zero_pp_rank_${i}_mp_rank_00_*model_states.pt" > /dev/null; then
                    SHARDS_OK=false
                    break
                fi
            done
            if [ "$SHARDS_OK" != true ]; then
                echo "[localize] WARN: 分片过滤下载不完整, 全量重下 tag 目录兜底"
                _azcopy_transfer --download "$RESUME_TAG_DIR_BLOB" "$RESUME_TAG_DIR_LOCAL" --dir
                _show_layout "resume tag (全量兜底)" "$RESUME_TAG_DIR_LOCAL"
            fi
            RESUME="$RESUME_LOCAL"
        fi
    fi

    # 4. ckpt 目录本地化 + 启动上传 watchdog (supervisor 循环, 崩溃自动重启)
    if _is_mount_path "$CKPT_DIR"; then
        CKPT_BLOB_BASE=$(_to_blob "$CKPT_DIR")
        CKPT_DIR=$(_to_local "$CKPT_DIR")
        mkdir -p "$CKPT_DIR"
        # 透传给训练进程: ZeRO-3 解冻回环的跨节点分片互换经 blob 汇合
        # (回环 load 需要 world_size 全量分片, 本地化后每节点只有本机分片)
        export LOLA_CKPT_BLOB_BASE="$CKPT_BLOB_BASE"
        export LOLA_AZCOPY_BIN="${LOCAL_MIRROR}/bin/azcopy"
        echo "[localize] ckpt: local=$CKPT_DIR, async upload -> $CKPT_BLOB_BASE (keep_last=$UPLOAD_KEEP_LAST)"
        (
            while true; do
                /home/aiscuser/.conda/envs/lerobot/bin/python "${SCRIPTS_DIR}/checkpoint_upload_watcher.py" \
                    --local_root "$CKPT_DIR" --blob_base "$CKPT_BLOB_BASE" \
                    --keep_last "$UPLOAD_KEEP_LAST" --drain_timeout "$UPLOAD_DRAIN_TIMEOUT" \
                    --azcopy-path "${LOCAL_MIRROR}/bin/azcopy" \
                    >> "${CKPT_DIR}/_watcher.log" 2>&1
                rc=$?
                if [ -f "${CKPT_DIR}/_upload_drain" ]; then
                    echo "[supervisor] watcher exited rc=$rc after drain" >> "${CKPT_DIR}/_watcher.log"
                    exit $rc
                fi
                echo "[supervisor] watcher exited rc=$rc unexpectedly, restarting in 10s" >> "${CKPT_DIR}/_watcher.log"
                sleep 10
            done
        ) &
        WATCHER_PID=$!
    fi
fi


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
echo "  - VLM backbone: ${VLM_BACKBONE}"
echo "  - VLM bridge: ${VLM_BRIDGE_MODE} (width=${VLM_BRIDGE_WIDTH}, layers=${VLM_BRIDGE_LAYERS})"
echo "  - VLM path: ${VLM_PATH}"
echo "  - DeepSpeed config: ${DEEPSPEED_CONFIG:-default}"
echo "  - DeepSpeed ZeRO stage: ${DEEPSPEED_ZERO_STAGE}"
echo "========================================"

# ----------------------------------------------------------------------
# 启动训练
# 使用 torchrun 来管理多 GPU，每个节点运行一次
# 单节点时使用简化命令，多节点时使用完整参数
# /home/aiscuser/.conda/envs/lerobot/bin for gcr
# /opt/conda/envs/lerobot/bin for kubenets
# ----------------------------------------------------------------------
if [ "$NNODES" -eq 1 ]; then
    # 单节点：使用简化的 torchrun 命令
    cmd="/home/aiscuser/.conda/envs/lerobot/bin/torchrun --nproc_per_node=${NPROC_PER_NODE} \
        src/lerobot/scripts/train_lola_v07_azure.py \
        --strategy ${STRATEGY} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --log_every_n_steps ${LOG_EVERY_N_STEPS} \
        --gradient_clip_val ${GRADIENT_CLIP_VAL} \
        --vlm_backbone ${VLM_BACKBONE} \
        --vlm_path ${VLM_PATH} \
        --ckpt_dir ${CKPT_DIR} \
        --action_dim ${ACTION_DIM} \
        --action_chunk_size ${ACTION_CHUNK_SIZE} \
        --pred_chunk_size ${PRED_CHUNK_SIZE} \
        --n_obs_steps ${N_OBS_STEPS} \
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
        --num_workers ${NUM_WORKERS} \
        --norm_mode ${NORM_MODE} \
        --norm_min ${NORM_MIN} \
        --norm_max ${NORM_MAX} \
        --stats_mode ${STATS_MODE} \
        --deepspeed_reduce_bucket_size ${DEEPSPEED_REDUCE_BUCKET_SIZE} \
        --deepspeed_allgather_bucket_size ${DEEPSPEED_ALLGATHER_BUCKET_SIZE} \
        --deepspeed_zero_stage ${DEEPSPEED_ZERO_STAGE} \
        --wandb_project ${WANDB_PROJECT}"
else
    # 多节点：使用完整的分布式参数
    cmd="/home/aiscuser/.conda/envs/lerobot/bin/torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        src/lerobot/scripts/train_lola_v07_azure.py \
        --strategy ${STRATEGY} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --log_every_n_steps ${LOG_EVERY_N_STEPS} \
        --gradient_clip_val ${GRADIENT_CLIP_VAL} \
        --vlm_backbone ${VLM_BACKBONE} \
        --vlm_path ${VLM_PATH} \
        --ckpt_dir ${CKPT_DIR} \
        --action_dim ${ACTION_DIM} \
        --action_chunk_size ${ACTION_CHUNK_SIZE} \
        --pred_chunk_size ${PRED_CHUNK_SIZE} \
        --n_obs_steps ${N_OBS_STEPS} \
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
        --num_workers ${NUM_WORKERS} \
        --norm_mode ${NORM_MODE} \
        --norm_min ${NORM_MIN} \
        --norm_max ${NORM_MAX} \
        --stats_mode ${STATS_MODE} \
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
if [ "$USE_STATE_CONDITION" = true ]; then
    cmd="${cmd} --use_state_condition"
fi
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

# V2: completed tasks 参数
if [ "$COMPLETED_TASKS_USE_ANN" = false ]; then
    cmd="${cmd} --no_completed_tasks_use_ann"
fi
cmd="${cmd} --completed_tasks_history_len ${COMPLETED_TASKS_HISTORY_LEN}"

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

# Special tokens
if [ "$USE_SPECIAL_TOKENS" = true ]; then
    cmd="${cmd} --use_special_tokens"
fi

echo "Running: $cmd"
# 训练失败也要走 drain: 已保存的 checkpoint 上传后才能用于 resume
set +e
eval $cmd
TRAIN_EXIT=$?
set -e

# 等待 checkpoint 上传 watchdog 排空 (保证 final ckpt 落 blob 后 AMLT job 才结束,
# 否则 job 结束节点回收, 本地未上传的 checkpoint 全部丢失)
if [ -n "$WATCHER_PID" ]; then
    echo "Training exited (code=$TRAIN_EXIT), waiting for checkpoint upload drain..."
    touch "${CKPT_DIR}/_upload_drain"
    wait $WATCHER_PID
    WATCHER_EXIT=$?
    if [ "$WATCHER_EXIT" -ne 0 ]; then
        echo "ERROR: checkpoint 上传排空失败 (watcher exit=$WATCHER_EXIT), 有 checkpoint 未传到 blob!"
        exit 1
    fi
    echo "All checkpoints uploaded to blob."
fi

echo "Training completed!"
exit $TRAIN_EXIT
