#!/usr/bin/env bash
set -euo pipefail

required=(MODEL MODEL_REVISION IMAGE HF_CACHE VLLM_CACHE LOG_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 1
  fi
done

if [[ "$MODEL_REVISION" == REPLACE_* || "$IMAGE" == *REPLACE_* ]]; then
  echo "MODEL_REVISION and IMAGE must be pinned before launch" >&2
  exit 1
fi

container="${CONTAINER:-qwen38-a100-fp8}"
port="${PORT:-18103}"
served_model_name="${SERVED_MODEL_NAME:-qwen3.8-27b}"
adapter_dir="${ADAPTER_DIR:-}"
adapter_model_name="${ADAPTER_MODEL_NAME:-qwen3.8-27b-uncensored}"
dtype="${DTYPE:-auto}"
kv_cache_dtype="${KV_CACHE_DTYPE:-auto}"
linear_backend="${LINEAR_BACKEND:-auto}"
enforce_eager="${ENFORCE_EAGER:-1}"
language_model_only="${LANGUAGE_MODEL_ONLY:-1}"
enable_prefix_caching="${ENABLE_PREFIX_CACHING:-1}"
enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL:-1}"
kv_cache_bytes="${KV_CACHE_BYTES:-}"
restart_policy="${RESTART_POLICY:-no}"
speculative_config="${SPECULATIVE_CONFIG:-}"
max_model_len="${MAX_MODEL_LEN:-32768}"
max_num_seqs="${MAX_NUM_SEQS:-8}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-8192}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
docker_api_version="${DOCKER_API_VERSION:-}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$HF_CACHE" "$VLLM_CACHE" "$LOG_DIR"

docker_args=(docker)
mount_args=()
if [[ -n "$docker_api_version" ]]; then
  export DOCKER_API_VERSION="$docker_api_version"
fi
if [[ -n "$adapter_dir" ]]; then
  if [[ ! -d "$adapter_dir" ]]; then
    echo "ADAPTER_DIR does not exist: $adapter_dir" >&2
    exit 1
  fi
  mount_args+=(-v "$adapter_dir:/adapter:ro")
fi

engine_args=(
  "$MODEL"
  --revision "$MODEL_REVISION"
  --served-model-name "$served_model_name"
  --host 127.0.0.1
  --port "$port"
  --dtype "$dtype"
  --kv-cache-dtype "$kv_cache_dtype"
  --linear-backend "$linear_backend"
  --max-model-len "$max_model_len"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$max_num_batched_tokens"
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --mm-encoder-tp-mode data
  --default-chat-template-kwargs '{"enable_thinking":false}'
)
if [[ "$enforce_eager" == 1 ]]; then
  engine_args+=(--enforce-eager)
elif [[ "$enforce_eager" != 0 ]]; then
  echo "ENFORCE_EAGER must be 0 or 1" >&2
  exit 1
fi
if [[ "$language_model_only" == 1 ]]; then
  engine_args+=(--language-model-only)
elif [[ "$language_model_only" != 0 ]]; then
  echo "LANGUAGE_MODEL_ONLY must be 0 or 1" >&2
  exit 1
fi
if [[ "$enable_prefix_caching" == 1 ]]; then
  engine_args+=(--enable-prefix-caching)
elif [[ "$enable_prefix_caching" != 0 ]]; then
  echo "ENABLE_PREFIX_CACHING must be 0 or 1" >&2
  exit 1
fi
if [[ "$enable_chunked_prefill" == 1 ]]; then
  engine_args+=(--enable-chunked-prefill)
elif [[ "$enable_chunked_prefill" != 0 ]]; then
  echo "ENABLE_CHUNKED_PREFILL must be 0 or 1" >&2
  exit 1
fi
if [[ -n "$kv_cache_bytes" ]]; then
  engine_args+=(--kv-cache-memory-bytes "$kv_cache_bytes")
fi
if [[ -n "$speculative_config" ]]; then
  engine_args+=(--speculative-config "$speculative_config")
fi
if [[ -n "$adapter_dir" ]]; then
  engine_args+=(
    --enable-lora
    --max-loras 1
    --max-lora-rank 1
    --lora-dtype bfloat16
    --lora-modules "$adapter_model_name=/adapter"
  )
fi

"${docker_args[@]}" rm -f "$container" >/dev/null 2>&1 || true
"${docker_args[@]}" run -d \
  --name "$container" \
  --gpus all \
  --ipc=host \
  --network host \
  --restart "$restart_policy" \
  --health-cmd "curl -fsS http://127.0.0.1:$port/health >/dev/null || exit 1" \
  --health-interval 30s \
  --health-timeout 5s \
  --health-retries 3 \
  --health-start-period 15m \
  -e HF_HOME=/hf-cache \
  -e VLLM_CACHE_ROOT=/vllm-cache \
  -v "$HF_CACHE:/hf-cache" \
  -v "$VLLM_CACHE:/vllm-cache" \
  -v "$LOG_DIR:/run-logs" \
  -v "$repo_dir:/lab:ro" \
  "${mount_args[@]}" \
  "$IMAGE" \
  "${engine_args[@]}" \
  >"$LOG_DIR/${container}.launch.log"

echo "started $container on 127.0.0.1:$port"
