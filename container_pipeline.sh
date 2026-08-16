#!/usr/bin/env bash
set -Eeuo pipefail

# This script runs inside the Docker container. Every setting can be overridden
# through --env-file without rebuilding the image.
DATA_CACHE="${DATA_CACHE:-/data/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-/output/imagenet_rn50_b256_e5_lr0475}"
IMAGENET_SPLITS="${IMAGENET_SPLITS:-train,validation,test}"
ARCHITECTURE="${ARCHITECTURE:-resnet50}"
METHOD="${METHOD:-both}"
EPOCHS="${EPOCHS:-5}"
PROBE_EPOCHS="${PROBE_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.475}"
ADAM_LR="${ADAM_LR:-$LEARNING_RATE}"
WORKERS="${WORKERS:-8}"
OT_GRAD="${OT_GRAD:-envelope}"
# The real runner solves the true multi-marginal MMPOT tensor with 3 marginals.
# Set OT_MARGINALS=2 to fall back to the pairwise two-marginal proxy.
EXPERIMENT_SCRIPT="${EXPERIMENT_SCRIPT:-/app/matryoshka_real_mmpot_experiment.py}"
DOWNLOADER_SCRIPT="${DOWNLOADER_SCRIPT:-/app/download_imagenet.py}"
OT_MARGINALS="${OT_MARGINALS:-3}"
OT_SOLVER_MODE="${OT_SOLVER_MODE:-cyclic}"
BENCHMARKS="${BENCHMARKS:-head,linear}"
DEVICE="${DEVICE:-cuda}"
OPTIMIZERS="${OPTIMIZERS:-sgd,adam}"
MOMENTUM="${MOMENTUM:-0.9}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# Host ownership for everything written under /output. run_full_pipeline.sh
# passes these so results land on the host already owned by the calling user
# instead of by root. 0 disables the chown (e.g. when /output is not a mount).
HOST_UID="${HOST_UID:-0}"
HOST_GID="${HOST_GID:-0}"
OUTPUT_MOUNT="${OUTPUT_MOUNT:-/output}"
REQUIRE_OUTPUT_MOUNT="${REQUIRE_OUTPUT_MOUNT:-1}"

echo "============================================================"
echo "Matryoshka-MMPOT full container pipeline"
echo "Experiment:     $EXPERIMENT_SCRIPT"
echo "OT marginals:   $OT_MARGINALS ($OT_SOLVER_MODE solver)"
echo "Dataset cache:  $DATA_CACHE"
echo "Output:         $OUTPUT_DIR"
echo "Splits:         $IMAGENET_SPLITS"
echo "Architecture:   $ARCHITECTURE"
echo "Method:         $METHOD"
echo "Epochs:         $EPOCHS"
echo "Probe epochs:   $PROBE_EPOCHS"
echo "Physical batch: $BATCH_SIZE"
echo "SGD LR:         $LEARNING_RATE"
echo "Adam LR:        $ADAM_LR"
echo "Optimizers:     $OPTIMIZERS"
echo "SGD momentum:   $MOMENTUM"
echo "Adam betas:     ($ADAM_BETA1, $ADAM_BETA2)"
echo "Weight decay:   $WEIGHT_DECAY"
echo "Benchmarks:     $BENCHMARKS"
echo "Device:         $DEVICE (CUDA required)"
echo "Host owner:     ${HOST_UID}:${HOST_GID} (applied to $OUTPUT_MOUNT on exit)"
echo "============================================================"

# Results must survive the container. Refuse to start when $OUTPUT_MOUNT is not
# a bind mount / volume, otherwise every checkpoint and metric file disappears
# with the container filesystem. Set REQUIRE_OUTPUT_MOUNT=0 to bypass.
if [[ "$REQUIRE_OUTPUT_MOUNT" == "1" ]]; then
    if ! mountpoint -q "$OUTPUT_MOUNT" 2>/dev/null && ! grep -qE "[[:space:]]${OUTPUT_MOUNT}[[:space:]]" /proc/self/mountinfo /proc/mounts 2>/dev/null; then
        echo "Error: '$OUTPUT_MOUNT' is not a mounted volume, so results would be lost" >&2
        echo "       when the container is removed. Start the container with" >&2
        echo "       -v /path/on/host:/output (run_full_pipeline.sh does this)," >&2
        echo "       or set REQUIRE_OUTPUT_MOUNT=0 to run anyway." >&2
        exit 3
    fi
fi

case "$OUTPUT_DIR" in
    "$OUTPUT_MOUNT"/*|"$OUTPUT_MOUNT") ;;
    *)
        echo "Warning: OUTPUT_DIR='$OUTPUT_DIR' is outside '$OUTPUT_MOUNT'; results will" >&2
        echo "         stay inside the container instead of on the host." >&2
        ;;
esac

mkdir -p "$OUTPUT_DIR"

# Hand the results back to the host user no matter how the run ends, so the
# files are readable outside Docker without sudo.
handoff_output() {
    local status=$?
    if [[ "$HOST_UID" != "0" || "$HOST_GID" != "0" ]]; then
        chown -R "${HOST_UID}:${HOST_GID}" "$OUTPUT_MOUNT" 2>/dev/null || \
            echo "Warning: could not chown $OUTPUT_MOUNT to ${HOST_UID}:${HOST_GID}" >&2
    fi
    chmod -R u+rwX,go+rX "$OUTPUT_MOUNT" 2>/dev/null || true
    return "$status"
}
trap handoff_output EXIT

# Preflight the requested device. This pipeline is GPU-only: any non-CUDA
# DEVICE is rejected here instead of silently falling back to the CPU.
DEVICE="$DEVICE" python - <<'PY'
import os
import torch

spec = os.environ["DEVICE"]
if spec == "auto":
    spec = "cuda"
    print("DEVICE=auto resolved to 'cuda' (this pipeline is CUDA-only)")

if not spec.startswith("cuda"):
    raise SystemExit(
        f"DEVICE='{spec}' is not supported: this pipeline is CUDA-only.\n"
        "  - Set DEVICE=cuda (or cuda:N) and run on a machine with an NVIDIA GPU."
    )

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA was requested but is unavailable inside the container.\n"
        "  - Start the container with `--gpus all` on a Linux host that has an\n"
        "    NVIDIA driver plus nvidia-container-toolkit installed.\n"
        "  - Docker Desktop on macOS cannot pass a GPU through, so this pipeline\n"
        "    cannot run there.\n"
        f"  - torch {torch.__version__} reports CUDA build: {torch.version.cuda}"
    )

index = int(spec.split(":", 1)[1]) if ":" in spec else 0
if index >= torch.cuda.device_count():
    raise SystemExit(
        f"DEVICE='{spec}' requests GPU index {index} but only "
        f"{torch.cuda.device_count()} device(s) are visible"
    )
props = torch.cuda.get_device_properties(index)
print(f"CUDA ready: {torch.cuda.get_device_name(index)} ({props.total_memory / 1024**3:.1f} GiB)")
PY

echo
# Flags differ between the real multi-marginal runner and the older pairwise
# one, so ask the selected script what it accepts instead of hard-coding them.
echo "Detecting flags supported by $EXPERIMENT_SCRIPT..."
SCRIPT_HELP="$(python "$EXPERIMENT_SCRIPT" --help 2>/dev/null || true)"
[[ -n "$SCRIPT_HELP" ]] || { echo "Error: '$EXPERIMENT_SCRIPT --help' produced no output" >&2; exit 4; }

EXTRA_ARGS=()
add_supported_flag() {
    local flag="$1"
    if grep -q -- "$flag" <<< "$SCRIPT_HELP"; then
        EXTRA_ARGS+=("$flag")
        if [[ "$#" -ge 2 ]]; then
            EXTRA_ARGS+=("$2")
        fi
    else
        echo "  note: $EXPERIMENT_SCRIPT does not support $flag; skipping it"
    fi
    # Never let a skipped flag look like a failure to `set -e`.
    return 0
}
add_supported_flag --require-validated-data
add_supported_flag --probe-epochs "$PROBE_EPOCHS"
add_supported_flag --ot-marginals "$OT_MARGINALS"
add_supported_flag --ot-solver-mode "$OT_SOLVER_MODE"
echo "  extra args: ${EXTRA_ARGS[*]}"

echo
echo "[1/2] Downloading or verifying ImageNet cache..."
python "$DOWNLOADER_SCRIPT" \
    --cache-dir "$DATA_CACHE" \
    --splits "$IMAGENET_SPLITS" \
    --check-samples all

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

    # `set -o pipefail` makes the tee pipeline carry the trainer's exit code,
    # and `if !` keeps `set -e` from killing the script before the message.
    if ! python "$EXPERIMENT_SCRIPT" \
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
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        --benchmark "$BENCHMARKS" \
        --selection-metric mean_top1 \
        --save-every 0 \
        --resume auto \
        --output-dir "$optimizer_output" \
        2>&1 | tee -a "$OUTPUT_DIR/${optimizer}_train.log"
    then
        echo "Optimizer run '$optimizer' failed; see $OUTPUT_DIR/${optimizer}_train.log" >&2
        exit 1
    fi
done

echo
echo "Collecting result files for the host..."
{
    echo "run_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "architecture=$ARCHITECTURE"
    echo "method=$METHOD"
    echo "epochs=$EPOCHS"
    echo "probe_epochs=$PROBE_EPOCHS"
    echo "batch_size=$BATCH_SIZE"
    echo "experiment_script=$EXPERIMENT_SCRIPT"
    echo "ot_marginals=$OT_MARGINALS"
    echo "ot_solver_mode=$OT_SOLVER_MODE"
    echo "ot_grad=$OT_GRAD"
    echo "optimizers=$OPTIMIZERS"
    echo "sgd_lr=$LEARNING_RATE"
    echo "adam_lr=$ADAM_LR"
    echo "momentum=$MOMENTUM"
    echo "weight_decay=$WEIGHT_DECAY"
    echo "benchmarks=$BENCHMARKS"
    echo "device=$DEVICE"
    echo "output_dir=$OUTPUT_DIR"
} > "$OUTPUT_DIR/run_manifest.txt"

find "$OUTPUT_DIR" -maxdepth 3 \
    \( -name "summary.json" -o -name "comparison.csv" -o -name "results.json" \
       -o -name "history.jsonl" -o -name "run_manifest.txt" -o -name "*_train.log" \) \
    -print | sort | tee "$OUTPUT_DIR/result_files.txt"

echo
echo "All optimizer runs completed successfully."
echo "Results are on the host under the directory bind-mounted at $OUTPUT_MOUNT."
