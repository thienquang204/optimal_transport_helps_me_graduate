#!/usr/bin/env bash
set -Eeuo pipefail

# Docker launcher for csr_vs_mmpot_imagenet.py. ImageNet is read from the
# existing Docker named volume; no host /data directory is required.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-matryoshka-mmpot:latest}"
DATA_VOLUME="${DATA_VOLUME:-imagenet-data}"
DATA_ROOT="${DATA_ROOT:-/data/huggingface}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/runs}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"

# ---------------------------------------------------------------------------
# EXPERIMENT SETTINGS — edit here, set environment variables, or append CLI
# flags to this script. Appended flags take precedence for scalar arguments.
# ---------------------------------------------------------------------------
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
DEVICE="${DEVICE:-cuda}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CONTRAST_WEIGHT="${CONTRAST_WEIGHT:-0.1}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MMPOT_WEIGHTS="${MMPOT_WEIGHTS:-0.8,0.9,1.0,1.1,1.2,1.3}"
OT_MASS="${OT_MASS:-0.9}"
OT_ETA="${OT_ETA:-0.2}"
OT_ITERS="${OT_ITERS:-100}"
OT_MICROBATCH="${OT_MICROBATCH:-32}"
SEED="${SEED:-42}"
AMP="${AMP:-1}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"
FAISS_GPU_DEVICE="${FAISS_GPU_DEVICE:-0}"
FAISS_TEMP_MEMORY_MIB="${FAISS_TEMP_MEMORY_MIB:-512}"

die() {
    echo "Error: $*" >&2
    exit 1
}

command -v docker >/dev/null 2>&1 || die "docker is not installed"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    command -v sudo >/dev/null 2>&1 || die "cannot access Docker and sudo is unavailable"
    sudo docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"
    docker_cmd=(sudo docker)
fi

"${docker_cmd[@]}" volume inspect "$DATA_VOLUME" >/dev/null 2>&1 || \
    die "Docker volume '$DATA_VOLUME' does not exist; run run_imagenet_download.sh first"

mkdir -p "$OUTPUT_ROOT"

echo "Building $IMAGE_NAME with the CSR/MMPOT dependencies (including CUDA FAISS)..."
"${docker_cmd[@]}" build -t "$IMAGE_NAME" "$SCRIPT_DIR"

python_args=(
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
    --mmpot-weights "$MMPOT_WEIGHTS"
    --ot-mass "$OT_MASS"
    --ot-eta "$OT_ETA"
    --ot-iters "$OT_ITERS"
    --ot-microbatch "$OT_MICROBATCH"
    --seed "$SEED"
    --faiss-gpu
    --faiss-gpu-device "$FAISS_GPU_DEVICE"
    --faiss-temp-memory-mib "$FAISS_TEMP_MEMORY_MIB"
)

case "$AMP" in
    1) python_args+=(--amp) ;;
    0) python_args+=(--no-amp) ;;
    *) die "AMP must be 0 or 1" ;;
esac

case "$REBUILD_CACHE" in
    1) python_args+=(--rebuild-cache) ;;
    0) ;;
    *) die "REBUILD_CACHE must be 0 or 1" ;;
esac

python_args+=("$@")

docker_args=(
    run --rm
    --gpus all
    --shm-size=16g
    -v "$DATA_VOLUME:/data"
    -v "$OUTPUT_ROOT:/output"
    -v "$SCRIPT_DIR:/workspace:ro"
)

if [[ -f "$ENV_FILE" ]]; then
    docker_args+=(--env-file "$ENV_FILE")
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    docker_args+=(-e HF_TOKEN)
fi
docker_args+=(
    -e DATA_BACKEND=hf
    -e CACHE_DIR=/output/csr_mmpot/resnet18_cache
    -e OUTPUT_DIR=/output/csr_mmpot/results
    -e WEIGHTS_CACHE=/output/csr_mmpot/weights
    -e INSTALL_DEPS=1
    -e FAISS_GPU=1
)

echo "Running frozen ResNet-18 CSR vs MMPOT"
echo "  ImageNet volume: $DATA_VOLUME mounted at /data"
echo "  dataset cache:   $DATA_ROOT"
echo "  host results:    $OUTPUT_ROOT"
echo "  train/val limit: $MAX_TRAIN / $MAX_VAL (0 means full split)"
echo "  FAISS GPU:       $FAISS_GPU_DEVICE (${FAISS_TEMP_MEMORY_MIB} MiB temporary memory)"
echo "  MMPOT weights:   $MMPOT_WEIGHTS"

run_status=0
"${docker_cmd[@]}" "${docker_args[@]}" \
    --entrypoint bash \
    "$IMAGE_NAME" \
    /workspace/run_csr_vs_mmpot_imagenet.sh "$DATA_ROOT" \
    "${python_args[@]}" || run_status=$?

# The experiment runs as root so it can reuse the existing named-volume cache.
# Use a short container operation to return bind-mounted files to the server
# user whether Docker itself needs sudo or not.
"${docker_cmd[@]}" run --rm \
    -v "$OUTPUT_ROOT:/output" \
    --entrypoint chown \
    "$IMAGE_NAME" -R "$(id -u):$(id -g)" /output || true

exit "$run_status"
