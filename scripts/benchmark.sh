#!/usr/bin/env bash
set -euo pipefail

container="${CONTAINER:-qwen38-vllm-native-lora}"
port="${PORT:-18102}"
base_model="${BASE_MODEL_NAME:-qwen3.8-27b}"
adapter_model="${ADAPTER_MODEL_NAME:-qwen3.8-27b-uncensored}"
result_dir="${RESULT_DIR:-/vllm-cache/prob-k7-native-lora/evidence}"
result_tag="${RESULT_TAG:-}"

if [[ -n "$result_tag" && ! "$result_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RESULT_TAG may contain only letters, digits, dot, underscore, and dash" >&2
  exit 1
fi

result_prefix=""
[[ -n "$result_tag" ]] && result_prefix="${result_tag}-"

docker exec "$container" mkdir -p "$result_dir"

bench() {
  local model="$1" concurrency="$2" input="$3" output="$4" prompts="$5" file="$6"
  docker exec "$container" vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:$port" \
    --endpoint /v1/chat/completions \
    --model /model \
    --served-model-name "$model" \
    --tokenizer /model \
    --dataset-name random \
    --random-input-len "$input" \
    --random-output-len "$output" \
    --random-range-ratio 0.0 \
    --num-prompts "$prompts" \
    --max-concurrency "$concurrency" \
    --seed 101 \
    --ignore-eos \
    --temperature 0 \
    --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
    --save-result --save-detailed \
    --result-dir "$result_dir" \
    --result-filename "$file"
}

bench "$base_model" 1 512 2048 1 "${result_prefix}base-c1.json"
bench "$adapter_model" 1 512 2048 1 "${result_prefix}adapter-c1.json"
bench "$base_model" 8 1024 256 32 "${result_prefix}base-c8.json"
bench "$adapter_model" 8 1024 256 32 "${result_prefix}adapter-c8.json"
