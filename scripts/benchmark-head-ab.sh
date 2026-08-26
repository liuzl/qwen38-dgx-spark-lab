#!/usr/bin/env bash
set -euo pipefail

# Controlled packed-FP4 lm_head vs dense-BF16 lm_head comparison.
# Both arms reuse the same image, DFlash2 drafter, LoRA, KV settings, warmup,
# and benchmark workload. The original FP4 service is restored on exit.

required=(FP4_MODEL_DIR BF16_MODEL_DIR DRAFT_DIR ADAPTER_DIR CACHE_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 1
  fi
  if [[ ! -e "${!name}" ]]; then
    echo "$name does not exist: ${!name}" >&2
    exit 1
  fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
container="${CONTAINER:-qwen38-vllm-native-lora}"
port="${PORT:-18102}"
base_model="${BASE_MODEL_NAME:-qwen3.8-27b}"
repeats="${AB_REPEATS:-3}"
startup_timeout="${STARTUP_TIMEOUT_SECONDS:-900}"
run_id="${AB_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
result_subdir="${AB_RESULT_SUBDIR:-prob-k7-native-lora/head-ab-${run_id}}"
result_dir="/vllm-cache/${result_subdir}"
host_result_dir="${CACHE_DIR}/${result_subdir}"
restore_model_dir="${RESTORE_MODEL_DIR:-$FP4_MODEL_DIR}"
restoring=0

case "$repeats" in
  ''|*[!0-9]*) echo "AB_REPEATS must be a positive integer" >&2; exit 1 ;;
esac
(( repeats > 0 )) || { echo "AB_REPEATS must be greater than zero" >&2; exit 1; }

mkdir -p "$host_result_dir"

wait_ready() {
  local deadline=$((SECONDS + startup_timeout))
  until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "server did not become healthy within ${startup_timeout}s" >&2
      docker logs --tail 200 "$container" >&2 || true
      return 1
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$container"; then
      echo "container exited before becoming healthy" >&2
      docker logs --tail 200 "$container" >&2 || true
      return 1
    fi
    sleep 5
  done
}

warmup() {
  local i
  for i in 1 2; do
    curl -fsS "http://127.0.0.1:${port}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${base_model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a compact Python LRU cache with type hints.\"}],\"max_tokens\":256,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
      >/dev/null
  done
}

launch() {
  local model_dir="$1"
  MODEL_DIR="$model_dir" "$script_dir/serve-native-lora.sh"
  wait_ready
  warmup
}

restore() {
  local status=$?
  if (( restoring == 0 )); then
    restoring=1
    echo "restoring service with MODEL_DIR=${restore_model_dir}" >&2
    MODEL_DIR="$restore_model_dir" "$script_dir/serve-native-lora.sh" >&2 || true
    wait_ready >&2 || true
  fi
  return "$status"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
  echo "run_id=${run_id}"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "image=${IMAGE:-qwen38-vllm-dflash2:lab}"
  echo "fp4_model_dir=${FP4_MODEL_DIR}"
  echo "bf16_model_dir=${BF16_MODEL_DIR}"
  echo "draft_dir=${DRAFT_DIR}"
  echo "adapter_dir=${ADAPTER_DIR}"
  echo "repeats=${repeats}"
  echo "result_dir=${result_dir}"
} >"${host_result_dir}/manifest.txt"

docker image inspect "${IMAGE:-qwen38-vllm-dflash2:lab}" \
  --format 'image_id={{.Id}} image_created={{.Created}}' \
  >>"${host_result_dir}/manifest.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
  >>"${host_result_dir}/manifest.txt" 2>/dev/null || true

for checkpoint in "$FP4_MODEL_DIR" "$BF16_MODEL_DIR"; do
  for artifact in config.json model.safetensors.index.json model-*-of-*.safetensors; do
    for file in "$checkpoint"/$artifact; do
      [[ -f "$file" ]] && sha256sum "$file"
    done
  done
done >"${host_result_dir}/checkpoint-sha256.txt"

for arm in fp4-head bf16-head; do
  if [[ "$arm" == "fp4-head" ]]; then
    model_dir="$FP4_MODEL_DIR"
  else
    model_dir="$BF16_MODEL_DIR"
  fi

  echo "starting ${arm}: ${model_dir}"
  launch "$model_dir"
  docker inspect "$container" --format '{{json .Config.Cmd}}' \
    >"${host_result_dir}/${arm}-server-command.json"

  for ((rep = 1; rep <= repeats; rep++)); do
    echo "benchmarking ${arm}, repetition ${rep}/${repeats}"
    RESULT_DIR="$result_dir" RESULT_TAG="${arm}-r${rep}" \
      "$script_dir/benchmark.sh"
  done
done

echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >>"${host_result_dir}/manifest.txt"
echo "A/B results: ${host_result_dir}"
