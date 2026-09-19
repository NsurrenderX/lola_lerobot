#!/usr/bin/env bash
set -euo pipefail

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29501}"
TRAINING_CONFIG="${TRAINING_CONFIG:-}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOCALIZE_IO="${LOCALIZE_IO:-false}"
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-}"
STORAGE_CONTAINER="${STORAGE_CONTAINER:-}"
MOUNT_PREFIX="${MOUNT_PREFIX:-/mnt/wangxiaofa}"
LOCAL_MIRROR="${LOCAL_MIRROR:-/scratch/lola_profile_mirror}"
AZCOPY_PATH="${AZCOPY_PATH:-}"
PROFILE_ARGS=()

fail() {
    printf 'profile launcher: %s\n' "$1" >&2
    exit 2
}

while (( $# )); do
    case "$1" in
        --nnodes|--nnodes=*|--nproc_per_node|--nproc_per_node=*|--node_rank|--node_rank=*|--master_addr|--master_addr=*|--master_port|--master_port=*|--training-config|--training-config=*|--output|--output=*|--python|--python=*|--storage_account|--storage_account=*|--storage_container|--storage_container=*|--mount_prefix|--mount_prefix=*|--local_mirror|--local_mirror=*|--azcopy_path|--azcopy_path=*)
            option="${1%%=*}"
            if [[ "$1" == *=* ]]; then
                value="${1#*=}"
                shift
            else
                (( $# >= 2 )) || fail "$option requires a value"
                [[ "$2" != --* ]] || fail "$option requires a value"
                value="$2"
                shift 2
            fi
            [[ -n "$value" ]] || fail "$option requires a nonempty value"
            case "$option" in
                --nnodes) NNODES="$value" ;;
                --nproc_per_node) NPROC_PER_NODE="$value" ;;
                --node_rank) NODE_RANK="$value" ;;
                --master_addr) MASTER_ADDR="$value" ;;
                --master_port) MASTER_PORT="$value" ;;
                --training-config) TRAINING_CONFIG="$value" ;;
                --output) PROFILE_OUTPUT="$value" ;;
                --python) PYTHON_BIN="$value" ;;
                --storage_account) STORAGE_ACCOUNT="$value" ;;
                --storage_container) STORAGE_CONTAINER="$value" ;;
                --mount_prefix) MOUNT_PREFIX="$value" ;;
                --local_mirror) LOCAL_MIRROR="$value" ;;
                --azcopy_path) AZCOPY_PATH="$value" ;;
            esac
            ;;
        --localize_io|--no_localize_io)
            if [[ "$1" == --localize_io ]]; then LOCALIZE_IO=true; else LOCALIZE_IO=false; fi
            shift
            ;;
        --help|-h)
            printf '%s\n' \
                'Usage: bash profile_azure_v07c.sh --nnodes 2 --nproc_per_node 8' \
                '  --node_rank 0 --master_addr HOST --master_port 9901' \
                '  --training-config PATH --output NEW_DIR [--python EXECUTABLE]' \
                '  [--localize_io --storage_account NAME --storage_container NAME]' \
                '  [--mount_prefix /mnt/wangxiaofa --local_mirror /scratch/lola_profile_mirror]' \
                '  [--azcopy_path EXECUTABLE] (--output is the blob destination in localized mode)' \
                '  [profile options] [-- trainer options]' \
                'CLI values override environment defaults. Options accept VALUE or =VALUE.'
            exit 0
            ;;
        --)
            PROFILE_ARGS+=("$@")
            break
            ;;
        *)
            PROFILE_ARGS+=("$1")
            shift
            ;;
    esac
done

[[ -n "$TRAINING_CONFIG" ]] || fail 'provide --training-config or TRAINING_CONFIG'
[[ -n "$PROFILE_OUTPUT" ]] || fail 'provide --output or PROFILE_OUTPUT'
[[ -n "$MASTER_ADDR" ]] || fail 'provide --master_addr or MASTER_ADDR'
[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || fail '--nnodes must be a positive integer'
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || fail '--nproc_per_node must be a positive integer'
[[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || fail '--node_rank must be a nonnegative integer'
(( NODE_RANK < NNODES )) || fail '--node_rank must be less than --nnodes'
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || fail '--master_port must be an integer'
(( MASTER_PORT <= 65535 )) || fail '--master_port must be in 1..65535'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../..:${SCRIPT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON_PATH="$(command -v "$PYTHON_BIN")" || fail "Python executable not found: $PYTHON_BIN"
PYTHON_LIB="$(dirname -- "$PYTHON_PATH")/../lib"
if [[ -d "$PYTHON_LIB" ]]; then
    export LD_LIBRARY_PATH="${PYTHON_LIB}:${LD_LIBRARY_PATH:-}"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ "$LOCALIZE_IO" == true ]]; then
    [[ -n "$STORAGE_ACCOUNT" && -n "$STORAGE_CONTAINER" ]] || fail '--localize_io requires --storage_account and --storage_container'
    IO_ARGS=(--storage_account "$STORAGE_ACCOUNT" --storage_container "$STORAGE_CONTAINER"
             --mount_prefix "$MOUNT_PREFIX" --local_mirror "$LOCAL_MIRROR")
    if [[ -n "$AZCOPY_PATH" ]]; then IO_ARGS+=(--azcopy_path "$AZCOPY_PATH"); fi
    exec "$PYTHON_PATH" "$SCRIPT_DIR/profile_lola_v07.py" localize \
        --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" \
        --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
        "${IO_ARGS[@]}" --training-config "$TRAINING_CONFIG" --output "$PROFILE_OUTPUT" "${PROFILE_ARGS[@]}"
fi

exec "$PYTHON_PATH" -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" --max_restarts=0 \
    "$SCRIPT_DIR/profile_lola_v07.py" \
    --training-config "$TRAINING_CONFIG" --output "$PROFILE_OUTPUT" "${PROFILE_ARGS[@]}"