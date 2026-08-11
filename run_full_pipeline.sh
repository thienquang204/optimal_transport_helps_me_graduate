#!/usr/bin/env bash
set -Eeuo pipefail

# Host-side A-Z runner. It builds the image with sudo, keeps ImageNet in a
# Docker named volume, writes checkpoints to ./runs, and starts the full
# download/train/benchmark pipeline in the foreground by default. This is
# intended for tmux, where all download and training output stays visible.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-start}"
IMAGE_NAME="${IMAGE_NAME:-matryoshka-mmpot:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-matryoshka-pipeline}"
DATA_VOLUME="${DATA_VOLUME:-imagenet-data}"
ENV_FILE="${ENV_FILE:-.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/runs}"
RESTART_LIMIT="${RESTART_LIMIT:-20}"

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

preflight() {
    command -v sudo >/dev/null 2>&1 || die "sudo is not installed"
    command -v docker >/dev/null 2>&1 || die "docker is not installed"
    sudo -v
    docker_sudo info >/dev/null 2>&1 || die "Docker daemon is not running"
    [[ -f "$ENV_FILE" ]] || die "$ENV_FILE does not exist; create it from .env.example"
    grep -q '^HF_TOKEN=hf_' "$ENV_FILE" || die "$ENV_FILE does not contain a valid-looking HF_TOKEN"
    mkdir -p "$OUTPUT_ROOT"
    OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
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
            -v "${DATA_VOLUME}:/data" \
            -v "${OUTPUT_ROOT}:/output" \
            --entrypoint /bin/bash \
            "$IMAGE_NAME" \
            /app/container_pipeline.sh >/dev/null

        echo
        echo "Full pipeline started in the background."
        show_status
        echo "Logs:       $0 logs"
    else
        echo "[3/3] Entering the container and streaming the full pipeline output..."
        echo "In tmux, detach with Ctrl+B then D. Do not press Ctrl+C unless you want to stop it."
        echo "Dataset:     Docker volume '$DATA_VOLUME'"
        echo "Checkpoints root: $OUTPUT_ROOT (see OUTPUT_DIR in .env/container log)"
        echo
        docker_sudo run -it \
            --name "$CONTAINER_NAME" \
            --gpus all \
            --shm-size=16g \
            --env-file "$ENV_FILE" \
            -v "${DATA_VOLUME}:/data" \
            -v "${OUTPUT_ROOT}:/output" \
            --entrypoint /bin/bash \
            "$IMAGE_NAME" \
            /app/container_pipeline.sh
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
        ;;
    restart)
        preflight
        if container_exists; then
            docker_sudo rm --force "$CONTAINER_NAME" >/dev/null
        fi
        start_pipeline foreground
        ;;
    *)
        echo "Usage: $0 {start|background|status|logs|stop|restart}" >&2
        exit 2
        ;;
esac
