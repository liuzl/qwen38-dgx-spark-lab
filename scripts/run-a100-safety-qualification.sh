#!/usr/bin/env bash
set -euo pipefail

required=(MODEL MODEL_REVISION IMAGE HF_CACHE VLLM_CACHE LOG_DIR SAFETY_TOOL_DIR SAFETY_RESULT_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 1
  fi
done

container="${CONTAINER:-qwen38-a100-native-lora}"
port="${PORT:-18103}"
adapter_model="${ADAPTER_MODEL_NAME:-qwen3.8-27b-uncensored}"
startup_timeout="${STARTUP_TIMEOUT:-600}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -e "$SAFETY_RESULT_DIR" ]]; then
  echo "SAFETY_RESULT_DIR already exists: $SAFETY_RESULT_DIR" >&2
  exit 1
fi

capture_and_stop() {
  docker logs "$container" >"$LOG_DIR/a100-native-lora-safety-server.log" 2>&1 || true
  if [[ "$(docker inspect "$container" --format '{{.State.Running}}' 2>/dev/null || true)" == true ]]; then
    docker stop --timeout 60 "$container" >/dev/null 2>&1 || true
  fi
}
trap capture_and_stop EXIT

"$repo_dir/scripts/serve-a100.sh"
deadline=$((SECONDS + startup_timeout))
while ((SECONDS < deadline)); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    break
  fi
  state="$(docker inspect "$container" --format '{{.State.Status}}' 2>/dev/null || true)"
  if [[ "$state" == exited || "$state" == dead ]]; then
    echo "server exited before health succeeded" >&2
    exit 1
  fi
  sleep 15
done
curl -fsS "http://127.0.0.1:$port/health" >/dev/null

python3 "$SAFETY_TOOL_DIR/qwen38-fixed-strongreject.py" \
  --vendor "$SAFETY_TOOL_DIR/bench_strongreject_refusal.py" \
  --base "http://127.0.0.1:$port" \
  --model "$adapter_model" \
  --results-dir "$SAFETY_RESULT_DIR" \
  --max-tokens 2048 \
  --concurrency 4 \
  --retries 2 \
  --request-timeout 600

docker run --rm --entrypoint python3 \
  -e HF_HOME=/hf-cache \
  -v "$HF_CACHE:/hf-cache" \
  -v "$SAFETY_TOOL_DIR:/tools:ro" \
  -v "$SAFETY_RESULT_DIR:/results" \
  "$IMAGE" \
  /tools/score_refusal_classifier.py /results \
  --device cpu \
  --batch-size 8 \
  --cache-dir /hf-cache/refusal-classifier

echo "safety qualification completed"
