#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"
EVAL_SCRIPT="${LOLA_EVAL_SCRIPT:-${SCRIPT_DIR}/../../../../eval_on_lola_07_summary_torchrun.py}"
EVAL_ARGS=()

while (( $# )); do
    case "$1" in
        --python|--nproc_per_node|--master_port|--eval-script)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                printf '%s requires a value\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --python) PYTHON_BIN="$2" ;;
                --nproc_per_node) NPROC_PER_NODE="$2" ;;
                --master_port) MASTER_PORT="$2" ;;
                --eval-script) EVAL_SCRIPT="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            printf '%s\n' \
                'Usage: bash eval_lola_v07_summary.sh [--python EXECUTABLE] [--nproc_per_node N]' \
                '  [--master_port PORT] [--eval-script PATH] -- EVALUATOR_ARGUMENTS' \
                'Defaults: grouped vision SDPA and DiT CUDA Graph enabled.' \
                'Disable with --no_vision_batched_sdpa and/or --no_dit_cuda_graph.' \
                'The evaluator requires checkpoint, training config and CALVIN inputs; no sampling settings are overridden.'
            exit 0
            ;;
        --) shift; EVAL_ARGS+=("$@"); break ;;
        *) EVAL_ARGS+=("$1"); shift ;;
    esac
done

[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid process count\n' >&2; exit 2; }
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] && (( MASTER_PORT <= 65535 )) || { printf 'Invalid master port\n' >&2; exit 2; }
[[ -f "$EVAL_SCRIPT" ]] || { printf 'Evaluator not found: %s\n' "$EVAL_SCRIPT" >&2; exit 2; }
export LOLA_LEROBOT_SRC="${LOLA_LEROBOT_SRC:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${LOLA_LEROBOT_SRC}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
PYTHON_PATH="$(command -v "$PYTHON_BIN")"
PYTHON_LIB="$(dirname -- "$PYTHON_PATH")/../lib"
if [[ -d "$PYTHON_LIB" ]]; then
    export LD_LIBRARY_PATH="${PYTHON_LIB}:${LD_LIBRARY_PATH:-}"
fi

exec "$PYTHON_PATH" -m torch.distributed.run --standalone \
    --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
    "$EVAL_SCRIPT" --vision_batched_sdpa --dit_cuda_graph "${EVAL_ARGS[@]}"