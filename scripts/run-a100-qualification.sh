#!/usr/bin/env bash
set -euo pipefail

required=(MODEL MODEL_REVISION IMAGE HF_CACHE VLLM_CACHE LOG_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 1
  fi
done

container="${CONTAINER:-qwen38-a100-fp8}"
port="${PORT:-18103}"
model_name="${SERVED_MODEL_NAME:-qwen3.8-27b}"
adapter_model_name="${ADAPTER_MODEL_NAME:-}"
run_id="${RUN_ID:-a100-fp8-eager-ar}"
startup_timeout="${STARTUP_TIMEOUT:-1800}"
telemetry_duration="${TELEMETRY_DURATION:-300}"
canary_max_tokens="${CANARY_MAX_TOKENS:-256}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 1
fi
if [[ "${RESTART_POLICY:-no}" != no ]]; then
  echo "qualification requires RESTART_POLICY=no" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
server_log="$LOG_DIR/${run_id}-server.log"
status_log="$LOG_DIR/${run_id}-container-status.json"
telemetry_pid=""

capture_server_state() {
  docker logs "$container" >"$server_log" 2>&1 || true
  if [[ -n "$telemetry_pid" ]]; then
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  if [[ "$(docker inspect "$container" --format '{{.State.Running}}' 2>/dev/null || true)" == true ]]; then
    docker stop --timeout 60 "$container" >/dev/null 2>&1 || true
  fi
  docker inspect "$container" >"$status_log" 2>/dev/null || true
}
trap capture_server_state EXIT

python3 "$repo_dir/scripts/capture-nvidia-environment.py" \
  --image "$IMAGE" \
  --model "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --output "$LOG_DIR/${run_id}-environment.prelaunch.json"

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
if ! curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
  echo "server did not become healthy within ${startup_timeout}s" >&2
  exit 1
fi

python3 "$repo_dir/scripts/capture-nvidia-environment.py" \
  --image "$IMAGE" \
  --model "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --output "$LOG_DIR/${run_id}-environment.json"

python3 "$repo_dir/scripts/capture-nvidia-telemetry.py" \
  --duration "$telemetry_duration" \
  --interval 1 \
  --label "$run_id" \
  --container "$container" \
  --output "$LOG_DIR/${run_id}-telemetry.json" &
telemetry_pid=$!

api_model_args=(--model "$model_name")
if [[ -n "$adapter_model_name" ]]; then
  api_model_args+=(--model "$adapter_model_name")
fi
docker exec "$container" python3 /lab/scripts/validate-apis.py \
  --base-url "http://127.0.0.1:$port" \
  "${api_model_args[@]}" \
  >"$LOG_DIR/${run_id}-api-smoke.json"

run_model_qualification() {
  local role="$1" selected_model="$2" suffix=""
  if [[ -n "$adapter_model_name" ]]; then
    suffix="-$role"
  fi
  docker exec "$container" python3 /lab/scripts/capture-greedy-canaries.py \
    --base-url "http://127.0.0.1:$port" \
    --model "$selected_model" \
    --label "$run_id$suffix" \
    --max-tokens "$canary_max_tokens" \
    --output "/run-logs/${run_id}${suffix}-canaries.raw.json"

  docker exec "$container" python3 /lab/scripts/benchmark-openai.py \
    --base-url "http://127.0.0.1:$port" \
    --model "$selected_model" \
    --label "$run_id$suffix" \
    --output "/run-logs/${run_id}${suffix}-benchmark.json" \
    --quiet

  docker exec "$container" python3 /lab/scripts/validate-stability.py \
    --base-url "http://127.0.0.1:$port" \
    --model "$selected_model" \
    --requests 64 \
    --concurrency 4 \
    >"$LOG_DIR/${run_id}${suffix}-stability.json"
}

run_model_qualification base "$model_name"
if [[ -n "$adapter_model_name" ]]; then
  docker exec "$container" python3 /lab/scripts/validate-cache-isolation.py \
    --base-url "http://127.0.0.1:$port" \
    --base-model "$model_name" \
    --adapter-model "$adapter_model_name" \
    >"$LOG_DIR/${run_id}-cache-isolation.json"
  run_model_qualification adapter "$adapter_model_name"
fi

curl -fsS "http://127.0.0.1:$port/metrics" \
  >"$LOG_DIR/${run_id}-metrics.prom"

wait "$telemetry_pid"
telemetry_pid=""
docker stop --timeout 60 "$container" >/dev/null
echo "qualification completed: $run_id"
