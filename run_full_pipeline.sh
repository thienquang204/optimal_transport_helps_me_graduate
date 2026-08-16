#!/usr/bin/env bash
set -Eeuo pipefail

# Host-side A-Z runner. It builds the image with sudo, keeps ImageNet in a
# Docker named volume, writes every checkpoint and metric file to the host
# directory $OUTPUT_ROOT (bind-mounted at /output inside the container), and
# starts the full download/train/benchmark pipeline in the foreground by
# default. This is intended for tmux, where all output stays visible.
#
# Results always land outside Docker: $OUTPUT_ROOT is created with sudo when
# needed, the container chowns /output back to the calling user on exit, and
# the runner re-applies ownership afterwards so nothing is left root-owned.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-start}"
IMAGE_NAME="${IMAGE_NAME:-matryoshka-mmpot:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-matryoshka-pipeline}"
DATA_VOLUME="${DATA_VOLUME:-imagenet-data}"
ENV_FILE="${ENV_FILE:-.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/runs}"
RESTART_LIMIT="${RESTART_LIMIT:-20}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"

die() {
    echo "Error: $*" >&2
    exit 1
}

docker_sudo() {
    sudo docker "$@"
}

container_exists() {
    docker_sudo container inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

container_running() {
    [[ "$(docker_sudo container inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == "true" ]]
}

show_status() {
    if ! container_exists; then
        echo "Container '$CONTAINER_NAME' does not exist."
        return 1
    fi
    docker_sudo ps -a --filter "name=^/${CONTAINER_NAME}$"
    docker_sudo container inspect \
        --format 'status={{.State.Status}} exit_code={{.State.ExitCode}} started={{.State.StartedAt}} finished={{.State.FinishedAt}} restart_count={{.RestartCount}}' \
        "$CONTAINER_NAME"
}

# Create $ENV_FILE from .env.example on the first run, filling in HF_TOKEN from
# the environment when it is exported, so `start` needs no manual setup step.
bootstrap_env_file() {
    if [[ -f "$ENV_FILE" ]]; then
        return 0
    fi
    [[ -f .env.example ]] || die "$ENV_FILE does not exist and .env.example is missing"
    echo "Creating $ENV_FILE from .env.example..."
    cp .env.example "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    if [[ -n "${HF_TOKEN:-}" ]]; then
        # Rewrite in place without leaking the token onto the command line.
        HF_TOKEN="$HF_TOKEN" ENV_FILE="$ENV_FILE" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["ENV_FILE"])
token = os.environ["HF_TOKEN"]
lines = [
    f"HF_TOKEN={token}" if line.startswith("HF_TOKEN=") else line
    for line in path.read_text().splitlines()
]
path.write_text("\n".join(lines) + "\n")
PY
        echo "HF_TOKEN copied from the environment into $ENV_FILE."
    else
        die "$ENV_FILE was created; put your Hugging Face token in HF_TOKEN and rerun"
    fi
}

# Create the host results directory and make sure the calling user owns it,
# falling back to sudo when the parent directory is not user-writable.
prepare_output_root() {
    if [[ ! -d "$OUTPUT_ROOT" ]]; then
        mkdir -p "$OUTPUT_ROOT" 2>/dev/null || {
            echo "Creating $OUTPUT_ROOT with sudo..."
            sudo mkdir -p "$OUTPUT_ROOT" || die "could not create $OUTPUT_ROOT"
        }
    fi
    if [[ ! -w "$OUTPUT_ROOT" ]]; then
        echo "Taking ownership of $OUTPUT_ROOT with sudo..."
        sudo chown "${HOST_UID}:${HOST_GID}" "$OUTPUT_ROOT" || die "could not chown $OUTPUT_ROOT"
    fi
    OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
}

# Anything the container wrote as root is handed back to the calling user, so
# the results are usable outside Docker without sudo. Safe to run repeatedly.
claim_results() {
    [[ -d "$OUTPUT_ROOT" ]] || return 0
    if find "$OUTPUT_ROOT" ! -user "$HOST_UID" -print -quit 2>/dev/null | grep -q .; then
        echo "Reclaiming root-owned result files in $OUTPUT_ROOT with sudo..."
        sudo chown -R "${HOST_UID}:${HOST_GID}" "$OUTPUT_ROOT" || \
            echo "Warning: chown of $OUTPUT_ROOT failed; use sudo to read the results" >&2
    fi
}

# Print the MRL vs MMPOT tables on the host. summarize_results.py is stdlib
# only, so no torch/venv is needed here; fall back to the copy the container
# already wrote when this machine has no python3 at all.
show_table() {
    local metric="${1:-top1,top5}"
    if command -v python3 >/dev/null 2>&1 && [[ -f "$SCRIPT_DIR/summarize_results.py" ]]; then
        python3 "$SCRIPT_DIR/summarize_results.py" "$OUTPUT_ROOT" --metric "$metric" || true
        return 0
    fi
    local cached
    cached="$(find "$OUTPUT_ROOT" -name results_table.txt -print -quit 2>/dev/null || true)"
    if [[ -n "$cached" ]]; then
        echo "(python3 unavailable; showing the table the container saved)"
        cat "$cached"
    else
        echo "(no python3 and no saved results_table.txt; skipping the table)"
    fi
}

show_results() {
    [[ -d "$OUTPUT_ROOT" ]] || die "$OUTPUT_ROOT does not exist yet"
    claim_results
    show_table
    echo
    echo "Host results directory: $OUTPUT_ROOT"
    echo
    local found=0
    while IFS= read -r path; do
        found=1
        printf '  %s\n' "$path"
    done < <(find "$OUTPUT_ROOT" -maxdepth 4 \
        \( -name "summary.json" -o -name "comparison.csv" -o -name "results.json" \
           -o -name "run_manifest.txt" -o -name "results_table.txt" \
           -o -name "*_train.log" -o -name "best.pt" \) \
        2>/dev/null | sort)
    if [[ "$found" -eq 0 ]]; then
        echo "  (no result files yet)"
    fi
    echo
    echo "Disk usage: $(du -sh "$OUTPUT_ROOT" 2>/dev/null | cut -f1)"
    echo
    echo "Copy to your machine with:"
    echo "  rsync -avhP --info=progress2 $(id -un)@$(hostname -s 2>/dev/null || hostname):${OUTPUT_ROOT}/ ./runs/"
}

preflight() {
    command -v sudo >/dev/null 2>&1 || die "sudo is not installed"
    command -v docker >/dev/null 2>&1 || die "docker is not installed"
    sudo -v
    docker_sudo info >/dev/null 2>&1 || die "Docker daemon is not running"
    bootstrap_env_file
    grep -q '^HF_TOKEN=hf_' "$ENV_FILE" || die "$ENV_FILE does not contain a valid-looking HF_TOKEN"
    docker_sudo info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia || \
        echo "Warning: no 'nvidia' Docker runtime detected. This pipeline is GPU-only and will abort without one." >&2
    prepare_output_root
}

build_image() {
    echo "[1/3] Building Docker image with sudo..."
    sudo env BUILDX_GIT_INFO=false docker build \
        --pull \
        -t "$IMAGE_NAME" \
        .
}

create_volume() {
    echo "[2/3] Preparing Docker dataset volume '$DATA_VOLUME'..."
    docker_sudo volume inspect "$DATA_VOLUME" >/dev/null 2>&1 || \
        docker_sudo volume create "$DATA_VOLUME" >/dev/null
}

stop_legacy_downloader() {
    if docker_sudo container inspect imagenet-downloader >/dev/null 2>&1; then
        echo "Stopping the old standalone downloader to avoid concurrent writes to '$DATA_VOLUME'..."
        if [[ "$(docker_sudo container inspect --format '{{.State.Running}}' imagenet-downloader)" == "true" ]]; then
            docker_sudo stop --time 30 imagenet-downloader >/dev/null
        fi
        docker_sudo rm imagenet-downloader >/dev/null
        echo "Its cached dataset remains intact in '$DATA_VOLUME'."
    fi
}

start_pipeline() {
    local run_mode="${1:-foreground}"
    preflight

    if container_running; then
        echo "Pipeline is already running."
        show_status
        echo "Follow it with: $0 logs"
        return
    fi

    stop_legacy_downloader
    build_image
    create_volume

    if container_exists; then
        local exit_code
        exit_code="$(docker_sudo container inspect --format '{{.State.ExitCode}}' "$CONTAINER_NAME")"
        if [[ "$exit_code" == "0" ]]; then
            echo "The previous full pipeline completed successfully."
            show_status
            echo "Use '$0 restart' only if you intentionally want to run it again."
            return
        fi
        echo "Removing failed container. Dataset and checkpoints are preserved."
        docker_sudo rm "$CONTAINER_NAME" >/dev/null
    fi

    if [[ "$run_mode" == "background" ]]; then
        echo "[3/3] Starting the full pipeline in the background..."
        docker_sudo run -d \
            --name "$CONTAINER_NAME" \
            --restart "on-failure:${RESTART_LIMIT}" \
            --gpus all \
            --shm-size=16g \
            --env-file "$ENV_FILE" \
            -e "HOST_UID=${HOST_UID}" \
            -e "HOST_GID=${HOST_GID}" \
            -v "${DATA_VOLUME}:/data" \
            -v "${OUTPUT_ROOT}:/output" \
            --entrypoint /bin/bash \
            "$IMAGE_NAME" \
            /app/container_pipeline.sh >/dev/null

        echo
        echo "Full pipeline started in the background."
        show_status
        echo "Logs:       $0 logs"
        echo "Results:    $0 results  (host directory $OUTPUT_ROOT)"
    else
        echo "[3/3] Entering the container and streaming the full pipeline output..."
        echo "In tmux, detach with Ctrl+B then D. Do not press Ctrl+C unless you want to stop it."
        echo "Dataset:     Docker volume '$DATA_VOLUME'"
        echo "Results root: $OUTPUT_ROOT on the host (see OUTPUT_DIR in .env/container log)"
        echo

        # Every trainer invocation uses --resume auto, so a crashed run picks up
        # from its last checkpoint. Retry automatically instead of asking the
        # user to babysit the terminal.
        local run_status=0 attempt=1
        while true; do
            if [[ "$attempt" -gt 1 ]]; then
                echo
                echo "Attempt ${attempt}/${MAX_ATTEMPTS}: resuming from the last checkpoint..."
                docker_sudo rm --force "$CONTAINER_NAME" >/dev/null 2>&1 || true
            fi

            run_status=0
            docker_sudo run -it \
                --name "$CONTAINER_NAME" \
                --gpus all \
                --shm-size=16g \
                --env-file "$ENV_FILE" \
                -e "HOST_UID=${HOST_UID}" \
                -e "HOST_GID=${HOST_GID}" \
                -v "${DATA_VOLUME}:/data" \
                -v "${OUTPUT_ROOT}:/output" \
                --entrypoint /bin/bash \
                "$IMAGE_NAME" \
                /app/container_pipeline.sh || run_status=$?

            # 3 is the "/output is not mounted" guard and 130 is Ctrl+C: both
            # are configuration/user decisions that a retry cannot fix.
            if [[ "$run_status" -eq 0 || "$run_status" -eq 3 || "$run_status" -eq 130 ]]; then
                break
            fi
            if [[ "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
                echo "Pipeline still failing after ${MAX_ATTEMPTS} attempt(s) (exit ${run_status})." >&2
                break
            fi
            attempt=$((attempt + 1))
        done

        echo
        show_results
        return "$run_status"
    fi
}

case "$ACTION" in
    start)
        start_pipeline foreground
        ;;
    background)
        start_pipeline background
        ;;
    status)
        sudo -v
        show_status
        ;;
    logs)
        sudo -v
        container_exists || die "container '$CONTAINER_NAME' does not exist"
        docker_sudo logs --follow --tail 100 "$CONTAINER_NAME"
        ;;
    stop)
        sudo -v
        if container_running; then
            docker_sudo stop --time 60 "$CONTAINER_NAME" >/dev/null
            echo "Pipeline stopped. Dataset and checkpoints were preserved."
        else
            echo "Pipeline is not running."
        fi
        claim_results
        ;;
    results)
        sudo -v
        show_results
        ;;
    table)
        # Tables only: no sudo, no chown, safe to run while training continues.
        show_table "${2:-all}"
        ;;
    fix-perms)
        sudo -v
        prepare_output_root
        sudo chown -R "${HOST_UID}:${HOST_GID}" "$OUTPUT_ROOT"
        echo "Ownership of $OUTPUT_ROOT set to ${HOST_UID}:${HOST_GID}."
        ;;
    restart)
        preflight
        if container_exists; then
            docker_sudo rm --force "$CONTAINER_NAME" >/dev/null
        fi
        start_pipeline foreground
        ;;
    *)
        echo "Usage: $0 {start|background|status|logs|stop|restart|results|table|fix-perms}" >&2
        echo >&2
        echo "  results     print the result tables, list the files, reclaim ownership" >&2
        echo "  table [m]   print only the MRL vs MMPOT tables (m: top1, all, ...)" >&2
        echo "  fix-perms   chown \$OUTPUT_ROOT back to the calling user (sudo)" >&2
        exit 2
        ;;
esac
