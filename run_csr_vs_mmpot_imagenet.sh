#!/usr/bin/env bash
set -Eeuo pipefail

# Convenient launcher for csr_vs_mmpot_imagenet.py.
#
# Usage:
#   ./run_csr_vs_mmpot_imagenet.sh /path/to/imagenet
#   DATA_ROOT=/path/to/imagenet ./run_csr_vs_mmpot_imagenet.sh --epochs 3
#   DATA_BACKEND=hf DATA_ROOT=/data/huggingface ./run_csr_vs_mmpot_imagenet.sh
#
# Any arguments after DATA_ROOT are forwarded to the Python program and can
# override the defaults below. Missing dependencies are installed by default;
# set INSTALL_DEPS=0 to require an already-prepared environment.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${DATA_ROOT:-}" && $# -gt 0 && "$1" != -* ]]; then
    DATA_ROOT="$1"
    shift
fi

if [[ -z "${DATA_ROOT:-}" ]]; then
    echo "Usage: $0 /path/to/imagenet [experiment arguments...]" >&2
    echo "   or: DATA_ROOT=/path/to/imagenet $0 [experiment arguments...]" >&2
    exit 2
fi

DATA_BACKEND="${DATA_BACKEND:-imagefolder}"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/runs/csr_mmpot/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/runs/csr_mmpot}"
WEIGHTS_CACHE="${WEIGHTS_CACHE:-$SCRIPT_DIR/weights}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
FAISS_GPU="${FAISS_GPU:-1}"

case "$DATA_BACKEND" in
    imagefolder|hf) ;;
    *)
        echo "Error: DATA_BACKEND must be 'imagefolder' or 'hf', not '$DATA_BACKEND'." >&2
        exit 2
        ;;
esac

required_modules=(torch torchvision numpy faiss PIL)
if [[ "$DATA_BACKEND" == "hf" ]]; then
    required_modules+=(datasets huggingface_hub)
fi

missing_modules=()
for module in "${required_modules[@]}"; do
    if ! "$PYTHON_BIN" -c "import ${module}" >/dev/null 2>&1; then
        missing_modules+=("$module")
    fi
done

case "$FAISS_GPU" in
    1)
        if ! "$PYTHON_BIN" -c "import faiss; raise SystemExit(0 if hasattr(faiss, 'StandardGpuResources') else 1)" >/dev/null 2>&1; then
            missing_modules+=(faiss)
        fi
        ;;
    0) ;;
    *)
        echo "Error: FAISS_GPU must be 0 or 1" >&2
        exit 2
        ;;
esac

if (( ${#missing_modules[@]} > 0 )); then
    if [[ "$FAISS_GPU" == "1" ]] && printf '%s\n' "${missing_modules[@]}" | grep -qx faiss; then
        echo "Error: CUDA FAISS is missing from this image." >&2
        echo "Rebuild it with run_full_pipeline.sh; GX10/ARM64 FAISS must be compiled at image build time." >&2
        exit 1
    fi
    if [[ "$INSTALL_DEPS" != "1" ]]; then
        echo "Error: missing Python modules: ${missing_modules[*]}" >&2
        echo "Install requirements.txt or rerun with INSTALL_DEPS=1." >&2
        exit 1
    fi
    echo "Installing missing Python dependencies: ${missing_modules[*]}"
    if (( ${#missing_modules[@]} == 1 )) && [[ "${missing_modules[0]}" == "faiss" ]]; then
        # Avoid reinstalling the large, platform-specific PyTorch stack when
        # an existing research environment only lacks the FAISS evaluator.
        "$PYTHON_BIN" -m pip install faiss-cpu
    else
        "$PYTHON_BIN" -m pip install --requirement "$SCRIPT_DIR/requirements.txt"
    fi

    for module in "${required_modules[@]}"; do
        "$PYTHON_BIN" -c "import ${module}" || {
            echo "Error: '$module' is unavailable after dependency installation." >&2
            exit 1
        }
    done
fi

if [[ "$FAISS_GPU" == "1" ]]; then
    "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'" || {
        echo "Error: FAISS_GPU=1 requires a CUDA-capable PyTorch runtime and visible NVIDIA GPU." >&2
        exit 1
    }
    "$PYTHON_BIN" -c "import faiss; assert hasattr(faiss, 'StandardGpuResources'), 'FAISS is CPU-only'" || {
        echo "Error: CUDA-enabled FAISS is required; rebuild the project image." >&2
        exit 1
    }
fi

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$WEIGHTS_CACHE"

command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/csr_vs_mmpot_imagenet.py"
    --data-root "$DATA_ROOT"
    --data-backend "$DATA_BACKEND"
    --cache-dir "$CACHE_DIR"
    --output-dir "$OUTPUT_DIR"
    --weights-cache "$WEIGHTS_CACHE"
)
if [[ "$FAISS_GPU" == "1" ]]; then
    command+=(--faiss-gpu)
else
    command+=(--no-faiss-gpu)
fi
command+=("$@")

echo "Running CSR vs MMPOT ImageNet experiment"
echo "  backend: $DATA_BACKEND"
echo "  data:    $DATA_ROOT"
echo "  cache:   $CACHE_DIR"
echo "  output:  $OUTPUT_DIR"
printf '  command:'
printf ' %q' "${command[@]}"
printf '\n'

exec "${command[@]}"
