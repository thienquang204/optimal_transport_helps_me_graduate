#!/usr/bin/env bash
set -Eeuo pipefail

# This script runs inside the Docker container. Every setting can be overridden
# through --env-file without rebuilding the image.
DATA_CACHE="${DATA_CACHE:-/data/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-/output/imagenet_rn50_b256_e10_lr0475}"
IMAGENET_SPLITS="${IMAGENET_SPLITS:-train,validation,test}"
ARCHITECTURE="${ARCHITECTURE:-resnet50}"
METHOD="${METHOD:-both}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.475}"
ADAM_LR="${ADAM_LR:-$LEARNING_RATE}"
WORKERS="${WORKERS:-8}"
OT_GRAD="${OT_GRAD:-envelope}"
BENCHMARKS="${BENCHMARKS:-head,linear}"
DEVICE="${DEVICE:-cuda}"
OPTIMIZERS="${OPTIMIZERS:-sgd,adam}"
MOMENTUM="${MOMENTUM:-0.9}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

echo "============================================================"
echo "Matryoshka-MMPOT full container pipeline"
echo "Dataset cache:  $DATA_CACHE"
echo "Output:         $OUTPUT_DIR"
echo "Splits:         $IMAGENET_SPLITS"
echo "Architecture:   $ARCHITECTURE"
echo "Method:         $METHOD"
echo "Epochs:         $EPOCHS"
echo "Physical batch: $BATCH_SIZE"
echo "SGD LR:         $LEARNING_RATE"
echo "Adam LR:        $ADAM_LR"
echo "Optimizers:     $OPTIMIZERS"
echo "SGD momentum:   $MOMENTUM"
echo "Adam betas:     ($ADAM_BETA1, $ADAM_BETA2)"
echo "Weight decay:   $WEIGHT_DECAY"
echo "Benchmarks:     $BENCHMARKS"
echo "Device:         $DEVICE (required)"
echo "============================================================"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable inside Docker"; print(f"CUDA ready: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB)")'

echo
echo "[1/2] Downloading or verifying ImageNet cache..."
python /app/download_imagenet.py \
    --cache-dir "$DATA_CACHE" \
    --splits "$IMAGENET_SPLITS"

echo
echo "[2/2] Training and benchmarking..."
IFS=',' read -r -a OPTIMIZER_LIST <<< "$OPTIMIZERS"
for optimizer in "${OPTIMIZER_LIST[@]}"; do
    optimizer="${optimizer//[[:space:]]/}"
    case "$optimizer" in
        sgd)
            optimizer_lr="$LEARNING_RATE"
            ;;
        adam)
            optimizer_lr="$ADAM_LR"
            if python -c "raise SystemExit(0 if float('$optimizer_lr') >= 0.1 else 1)"; then
                echo "WARNING: Adam LR=$optimizer_lr is unusually high and may diverge. Set ADAM_LR=0.001 to use a conventional value."
            fi
            ;;
        *)
            echo "Unsupported optimizer in OPTIMIZERS: '$optimizer'" >&2
            exit 2
            ;;
    esac

    optimizer_output="$OUTPUT_DIR/$optimizer"
    echo
    echo "------------------------------------------------------------"
    echo "Optimizer run: $optimizer"
    echo "Learning rate: $optimizer_lr"
    echo "Output:        $optimizer_output"
    echo "------------------------------------------------------------"

    python /app/matryoshka_mmpot_experiment.py \
        --dataset imagenet \
        --data-root "$DATA_CACHE" \
        --architecture "$ARCHITECTURE" \
        --method "$METHOD" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --optimizer "$optimizer" \
        --lr "$optimizer_lr" \
        --momentum "$MOMENTUM" \
        --adam-beta1 "$ADAM_BETA1" \
        --adam-beta2 "$ADAM_BETA2" \
        --weight-decay "$WEIGHT_DECAY" \
        --workers "$WORKERS" \
        --device "$DEVICE" \
        --amp \
        --ot-grad "$OT_GRAD" \
        --benchmark "$BENCHMARKS" \
        --selection-metric mean_top1 \
        --save-every 0 \
        --resume auto \
        --output-dir "$optimizer_output"
done

echo "All optimizer runs completed successfully."
