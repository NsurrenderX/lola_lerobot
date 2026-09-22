#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_ARGS=()

fail() {
    printf 'split launcher: %s\n' "$1" >&2
    exit 2
}

while (( $# )); do
    case "$1" in
        --python|--python=*)
            if [[ "$1" == *=* ]]; then
                PYTHON_BIN="${1#*=}"
                shift
            else
                (( $# >= 2 )) || fail '--python requires a value'
                [[ "$2" != --* ]] || fail '--python requires a value'
                PYTHON_BIN="$2"
                shift 2
            fi
            [[ -n "$PYTHON_BIN" ]] || fail '--python requires a nonempty value'
            ;;
        --)
            SPLIT_ARGS+=("$@")
            break
            ;;
        *)
            SPLIT_ARGS+=("$1")
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="$(command -v "$PYTHON_BIN")" || fail "Python executable not found: $PYTHON_BIN"
export PYTHONPATH="${SCRIPT_DIR}/../..:${SCRIPT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
PYTHON_LIB="$(dirname -- "$PYTHON_PATH")/../lib"
if [[ -d "$PYTHON_LIB" ]]; then
    export LD_LIBRARY_PATH="${PYTHON_LIB}:${LD_LIBRARY_PATH:-}"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

exec "$PYTHON_PATH" -m lerobot.scripts.train_lola_v07_split run "${SPLIT_ARGS[@]}"