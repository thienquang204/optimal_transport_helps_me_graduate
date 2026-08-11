#!/usr/bin/env bash
set -Eeuo pipefail

# This script runs inside the Docker container. Every setting can be overridden
# through --env-file without rebuilding the image.
DATA_CACHE="${DATA_CACHE:-/data/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-/output/imagenet_rn50}"
IMAGENET_SPLITS="${IMAGENET_SPLITS:-train,validation,test}"
ARCHITECTURE="${ARCHITECTURE:-resnet50}"
METHOD="${METHOD:-both}"
EPOCHS="${EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
WORKERS="${WORKERS:-8}"
OT_GRAD="${OT_GRAD:-envelope}"
BENCHMARKS="${BENCHMARKS:-head,linear}"
SAVE_EVERY="${SAVE_EVERY:-5}"

echo "============================================================"
echo "Matryoshka-MMPOT full container pipeline"
echo "Dataset cache:  $DATA_CACHE"
echo "Output:         $OUTPUT_DIR"
echo "Splits:         $IMAGENET_SPLITS"
echo "Architecture:   $ARCHITECTURE"
echo "Method:         $METHOD"
echo "Epochs:         $EPOCHS"
echo "Physical batch: $BATCH_SIZE"
echo "Learning rate:  $LEARNING_RATE"
echo "Benchmarks:     $BENCHMARKS"
echo "============================================================"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable inside Docker"; print(f"CUDA ready: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB)")'

echo
echo "[1/2] Downloading or verifying ImageNet cache..."
python /app/download_imagenet.py \
    --cache-dir "$DATA_CACHE" \
    --splits "$IMAGENET_SPLITS"

echo
echo "[2/2] Training and benchmarking..."
exec python /app/matryoshka_mmpot_experiment.py \
    --dataset imagenet \
    --data-root "$DATA_CACHE" \
    --architecture "$ARCHITECTURE" \
    --method "$METHOD" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --workers "$WORKERS" \
    --amp \
    --ot-grad "$OT_GRAD" \
    --benchmark "$BENCHMARKS" \
    --selection-metric mean_top1 \
    --save-every "$SAVE_EVERY" \
    --resume auto \
    --output-dir "$OUTPUT_DIR"
