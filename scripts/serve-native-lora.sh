#!/usr/bin/env bash
set -euo pipefail

required=(MODEL_DIR DRAFT_DIR ADAPTER_DIR CACHE_DIR)
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

image="${IMAGE:-qwen38-vllm-dflash2:lab}"
container="${CONTAINER:-qwen38-vllm-native-lora}"
port="${PORT:-18102}"
max_model_len="${MAX_MODEL_LEN:-131072}"
kv_cache_bytes="${KV_CACHE_BYTES:-17179869184}"
max_num_seqs="${MAX_NUM_SEQS:-10}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-16384}"
base_model_name="${BASE_MODEL_NAME:-qwen3.8-27b}"
adapter_model_name="${ADAPTER_MODEL_NAME:-qwen3.8-27b-uncensored}"

mkdir -p "$CACHE_DIR/flashinfer" "$CACHE_DIR/prob-k7-native-lora"

docker rm -f "$container" >/dev/null 2>&1 || true
docker run -d \
  --name "$container" \
  --gpus all \
  --ipc=host \
  --network host \
  --restart unless-stopped \
  --health-cmd "curl -fsS http://127.0.0.1:$port/health >/dev/null || exit 1" \
  --health-interval 30s \
  --health-timeout 5s \
  --health-retries 3 \
  --health-start-period 8m \
  -e VLLM_CACHE_ROOT=/vllm-cache/prob-k7-native-lora \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v "$MODEL_DIR:/model:ro" \
  -v "$DRAFT_DIR:/draft:ro" \
  -v "$ADAPTER_DIR:/adapter:ro" \
  -v "$CACHE_DIR:/vllm-cache" \
  -v "$CACHE_DIR/flashinfer:/root/.cache/flashinfer" \
  "$image" \
  --model /model \
  --served-model-name "$base_model_name" \
  --host 0.0.0.0 \
  --port "$port" \
  --max-model-len "$max_model_len" \
  --gpu-memory-utilization 0.50 \
  --kv-cache-memory-bytes "$kv_cache_bytes" \
  --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens "$max_num_batched_tokens" \
  --enable-prefix-caching \
  --prefix-cache-retention-interval 1648 \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8_e4m3 \
  --no-enable-flashinfer-autotune \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 1 \
  --lora-dtype bfloat16 \
  --lora-modules "$adapter_model_name=/adapter" \
  --speculative-config \
    '{"method":"dflash","model":"/draft","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"draft_sample_method":"probabilistic"}'

echo "started $container on :$port"
