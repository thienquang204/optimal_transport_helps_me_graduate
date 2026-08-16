#!/usr/bin/env bash
set -Eeuo pipefail

# Complete launcher for csr_vs_mmpot_imagenet.py.
#
# Basic use:
#   ./run_full_pipeline.sh /data/imagenet
#
# Configure common parameters without editing this file:
#   DATA_ROOT=/data/imagenet MAX_TRAIN=50000 MAX_VAL=10000 EPOCHS=3 \
#     BATCH_SIZE=256 ./run_full_pipeline.sh
#
# Additional Python arguments can be appended and take precedence:
#   ./run_full_pipeline.sh /data/imagenet --epochs 5 --method mmpot

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${DATA_ROOT:-}" && $# -gt 0 && "$1" != -* ]]; then
    DATA_ROOT="$1"
    shift
fi

if [[ -z "${DATA_ROOT:-}" ]]; then
    echo "Usage: $0 /path/to/imagenet [experiment arguments...]" >&2
    echo "   or: DATA_ROOT=/path/to/imagenet $0 [experiment arguments...]" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# DEFAULT EXPERIMENT SETTINGS — edit these values directly if desired.
# Environment variables or trailing CLI arguments can still override them.
# ---------------------------------------------------------------------------

# Dataset and storage
export DATA_BACKEND="${DATA_BACKEND:-imagefolder}"
export CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/runs/csr_mmpot/resnet18_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/runs/csr_mmpot}"
export WEIGHTS_CACHE="${WEIGHTS_CACHE:-$SCRIPT_DIR/weights}"
export INSTALL_DEPS="${INSTALL_DEPS:-1}"

# Full ImageNet paper-scale defaults. Use positive MAX_TRAIN/MAX_VAL values for
# a smaller development subset; zero means use the complete corresponding split.
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_VAL="${MAX_VAL:-0}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-512}"
WORKERS="${WORKERS:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
TRAIN_K="${TRAIN_K:-32}"
TOPK="${TOPK:-8,16,32,64,128,256}"
METHOD="${METHOD:-both}"
DEVICE="${DEVICE:-auto}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CONTRAST_WEIGHT="${CONTRAST_WEIGHT:-0.1}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MMPOT_WEIGHT="${MMPOT_WEIGHT:-0.1}"
OT_MASS="${OT_MASS:-0.8}"
OT_ETA="${OT_ETA:-0.2}"
OT_ITERS="${OT_ITERS:-100}"
OT_MICROBATCH="${OT_MICROBATCH:-32}"
SEED="${SEED:-42}"
AMP="${AMP:-1}"

args=(
    --max-train "$MAX_TRAIN"
    --max-val "$MAX_VAL"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --feature-batch-size "$FEATURE_BATCH_SIZE"
    --workers "$WORKERS"
    --hidden-dim "$HIDDEN_DIM"
    --train-k "$TRAIN_K"
    --topk "$TOPK"
    --method "$METHOD"
    --device "$DEVICE"
    --lr "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --contrast-weight "$CONTRAST_WEIGHT"
    --temperature "$TEMPERATURE"
    --mmpot-weight "$MMPOT_WEIGHT"
    --ot-mass "$OT_MASS"
    --ot-eta "$OT_ETA"
    --ot-iters "$OT_ITERS"
    --ot-microbatch "$OT_MICROBATCH"
    --seed "$SEED"
)

if [[ "$REBUILD_CACHE" == "1" ]]; then
    args+=(--rebuild-cache)
elif [[ "$REBUILD_CACHE" != "0" ]]; then
    echo "Error: REBUILD_CACHE must be 0 or 1." >&2
    exit 2
fi

if [[ "$AMP" == "1" ]]; then
    args+=(--amp)
elif [[ "$AMP" == "0" ]]; then
    args+=(--no-amp)
else
    echo "Error: AMP must be 0 or 1." >&2
    exit 2
fi

# User-supplied CLI options come last, so argparse uses them as overrides for
# ordinary scalar parameters such as --epochs and --batch-size.
args+=("$@")

echo "CSR vs MMPOT full pipeline (frozen torchvision ResNet-18)"
echo "  max train/val: $MAX_TRAIN / $MAX_VAL (0 means the full split)"
echo "  epochs/batch:  $EPOCHS / $BATCH_SIZE"
echo "  method/device: $METHOD / $DEVICE"
echo "  rebuild cache: $REBUILD_CACHE"

exec bash "$SCRIPT_DIR/run_csr_vs_mmpot_imagenet.sh" "$DATA_ROOT" "${args[@]}"
