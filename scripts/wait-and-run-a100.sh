#!/usr/bin/env bash
set -euo pipefail

required=(DOWNLOAD_CONTAINER MODEL MODEL_REVISION IMAGE HF_CACHE VLLM_CACHE LOG_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 1
  fi
done

asset_timeout="${ASSET_TIMEOUT:-43200}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deadline=$((SECONDS + asset_timeout))

while [[ "$(docker inspect "$DOWNLOAD_CONTAINER" --format '{{.State.Running}}' 2>/dev/null || true)" == true ]]; do
  if ((SECONDS >= deadline)); then
    echo "timed out waiting for model download" >&2
    exit 1
  fi
  sleep 30
done

download_exit="$(docker inspect "$DOWNLOAD_CONTAINER" --format '{{.State.ExitCode}}' 2>/dev/null || echo missing)"
if [[ "$download_exit" != 0 ]]; then
  echo "model download failed: exit=$download_exit" >&2
  docker logs --tail 80 "$DOWNLOAD_CONTAINER" >&2 || true
  exit 1
fi

while ! docker image inspect "$IMAGE" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    echo "timed out waiting for container image" >&2
    exit 1
  fi
  sleep 30
done

"$repo_dir/scripts/run-a100-qualification.sh"
