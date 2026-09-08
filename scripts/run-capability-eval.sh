#!/usr/bin/env bash
# Capability evaluation of one served alias through lm-evaluation-harness,
# driven from the CPU-only qwen38-lm-eval container against a running vLLM
# endpoint. Used to compare the FP8 and INT8 targets, base and adapter aliases,
# with the same harness, prompts, and few-shot settings.
#
# Stage 1 (completions endpoint, no chat template):
#   loglikelihood multiple choice: arc_challenge, hellaswag, winogrande,
#   truthfulqa_mc2, mmlu; generative: gsm8k (5-shot, strict/flexible match)
# Stage 2 (chat completions with the served chat template, thinking off):
#   ifeval (instruction following)
#
# usage: run-capability-eval.sh <label> <served_model> [base_url]
#   base_url defaults to production (http://127.0.0.1:18103)
# env: EVAL_TASKS, CHAT_TASKS, LIMIT (debug subsample), NUM_CONCURRENT
set -euo pipefail
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.43}"
label="${1:?label}"
model="${2:?served model name}"
base_url="${3:-http://127.0.0.1:18103}"
EVAL_CONTAINER="${EVAL_CONTAINER:-qwen38-lm-eval}"
EVAL_TASKS="${EVAL_TASKS:-arc_challenge,hellaswag,winogrande,truthfulqa_mc2,mmlu,gsm8k}"
CHAT_TASKS="${CHAT_TASKS:-ifeval}"
NUM_CONCURRENT="${NUM_CONCURRENT:-32}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.8-27B-FP8}"
LIMIT="${LIMIT:-}"
[[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad label" >&2; exit 1; }

limit_arg=()
[[ -n "$LIMIT" ]] && limit_arg=(--limit "$LIMIT")
out="/eval/results/$label"

echo "[$(date -u +%FT%TZ)] eval $label model=$model url=$base_url"
curl -fsS "$base_url/health" >/dev/null || { echo "endpoint not healthy: $base_url" >&2; exit 1; }
curl -fsS "$base_url/v1/models" | python3 -c '
import json,sys; ids=[m["id"] for m in json.load(sys.stdin)["data"]]
assert sys.argv[1] in ids, f"{sys.argv[1]} not served: {ids}"' "$model"

echo "[$(date -u +%FT%TZ)] stage 1: $EVAL_TASKS"
docker exec "$EVAL_CONTAINER" lm_eval \
  --model local-completions \
  --model_args "model=$model,base_url=$base_url/v1/completions,num_concurrent=$NUM_CONCURRENT,max_retries=3,tokenized_requests=False,tokenizer=$TOKENIZER,tokenizer_backend=huggingface" \
  --tasks "$EVAL_TASKS" \
  --batch_size "$NUM_CONCURRENT" \
  --output_path "$out/completions" \
  --log_samples \
  "${limit_arg[@]}" 2>&1 | grep -v -E 'Warning|it/s\]' | tail -40

if [[ -n "$CHAT_TASKS" ]]; then
  echo "[$(date -u +%FT%TZ)] stage 2: $CHAT_TASKS"
  docker exec "$EVAL_CONTAINER" lm_eval \
    --model local-chat-completions \
    --model_args "model=$model,base_url=$base_url/v1/chat/completions,num_concurrent=$NUM_CONCURRENT,max_retries=3,tokenized_requests=False,tokenizer=$TOKENIZER,tokenizer_backend=huggingface" \
    --tasks "$CHAT_TASKS" \
    --apply_chat_template \
    --batch_size "$NUM_CONCURRENT" \
    --output_path "$out/chat" \
    --log_samples \
    "${limit_arg[@]}" 2>&1 | grep -v -E 'Warning|it/s\]' | tail -30
fi
echo "[$(date -u +%FT%TZ)] EVAL DONE $label"
