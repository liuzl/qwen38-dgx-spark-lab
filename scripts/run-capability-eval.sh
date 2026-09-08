#!/usr/bin/env bash
# Capability evaluation of one served alias through lm-evaluation-harness,
# driven from the CPU-only qwen38-lm-eval container against a running vLLM
# endpoint. Used to compare the FP8 and INT8 targets, base and adapter aliases,
# with the same harness, prompts, and few-shot settings.
#
# Stage 1 (completions endpoint, no chat template), Open-LLM-Leaderboard-v1
# few-shot settings. These also keep every multiple-choice prompt above the
# 241-255-token band in which vLLM 0.28.0 returns NaN prompt logprobs on this
# serving profile (see docs/a100-performance-tuning.md):
#   mmlu 5-shot, arc_challenge 25-shot, hellaswag 10-shot, winogrande 5-shot,
#   truthfulqa_mc2 0-shot, gsm8k 5-shot (generative, strict/flexible match)
# Stage 2 (chat completions with the served chat template, thinking off):
#   ifeval (instruction following)
#
# All requests go through a local logging proxy; any non-200 response is
# counted and the run is marked INVALID if the count is not zero.
#
# usage: run-capability-eval.sh <label> <served_model> [base_url]
#   base_url defaults to production (http://127.0.0.1:18103)
# env: STAGE1 ("task:fewshot task:fewshot ..."), CHAT_TASKS, LIMIT (debug
#      subsample), NUM_CONCURRENT, PROXY_PORT
set -euo pipefail
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.43}"
label="${1:?label}"
model="${2:?served model name}"
base_url="${3:-http://127.0.0.1:18103}"
EVAL_CONTAINER="${EVAL_CONTAINER:-qwen38-lm-eval}"
EVAL_DIR="${EVAL_DIR:-/databank/zliu/qwen38-a100/eval}"
STAGE1="${STAGE1:-mmlu:5 arc_challenge:25 hellaswag:10 winogrande:5 truthfulqa_mc2:0 gsm8k:5}"
CHAT_TASKS="${CHAT_TASKS:-ifeval}"
NUM_CONCURRENT="${NUM_CONCURRENT:-32}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.8-27B-FP8}"
LIMIT="${LIMIT:-}"
PROXY_PORT="${PROXY_PORT:-18199}"
[[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad label" >&2; exit 1; }

limit_arg=()
[[ -n "$LIMIT" ]] && limit_arg=(--limit "$LIMIT")
out="/eval/results/$label"
proxy_log="$EVAL_DIR/results/$label.non200.jsonl"
mkdir -p "$EVAL_DIR/results"
rm -f "$proxy_log"

echo "[$(date -u +%FT%TZ)] eval $label model=$model url=$base_url"
curl -fsS "$base_url/health" >/dev/null || { echo "endpoint not healthy: $base_url" >&2; exit 1; }
curl -fsS "$base_url/v1/models" | python3 -c '
import json,sys; ids=[m["id"] for m in json.load(sys.stdin)["data"]]
assert sys.argv[1] in ids, f"{sys.argv[1]} not served: {ids}"' "$model"

# logging proxy inside the eval container (same network namespace: --network host)
docker exec "$EVAL_CONTAINER" pkill -f "logproxy.py .* $PROXY_PORT " 2>/dev/null || true
docker exec -d "$EVAL_CONTAINER" python3 /eval/logproxy.py "$base_url" "$PROXY_PORT" "/eval/results/$label.non200.jsonl"
sleep 2
curl -fsS "http://127.0.0.1:$PROXY_PORT/health" >/dev/null || { echo "proxy failed to start" >&2; exit 1; }
purl="http://127.0.0.1:$PROXY_PORT"

for spec in $STAGE1; do
  task="${spec%%:*}"; shots="${spec##*:}"
  echo "[$(date -u +%FT%TZ)] stage 1: $task ($shots-shot)"
  docker exec "$EVAL_CONTAINER" lm_eval \
    --model local-completions \
    --model_args "model=$model,base_url=$purl/v1/completions,num_concurrent=$NUM_CONCURRENT,max_retries=3,tokenized_requests=False,tokenizer=$TOKENIZER,tokenizer_backend=huggingface" \
    --tasks "$task" --num_fewshot "$shots" \
    --batch_size "$NUM_CONCURRENT" \
    --output_path "$out/$task" \
    --log_samples \
    "${limit_arg[@]}" 2>&1 | grep -E '^\||Error|error|Traceback' | grep -v -E '^\|-|Tasks' || true
  echo "  non-200 so far: $(wc -l <"$proxy_log" 2>/dev/null || echo 0)"
done

if [[ -n "$CHAT_TASKS" ]]; then
  echo "[$(date -u +%FT%TZ)] stage 2: $CHAT_TASKS (chat template)"
  docker exec "$EVAL_CONTAINER" lm_eval \
    --model local-chat-completions \
    --model_args "model=$model,base_url=$purl/v1/chat/completions,num_concurrent=$NUM_CONCURRENT,max_retries=3,tokenized_requests=False,tokenizer=$TOKENIZER,tokenizer_backend=huggingface" \
    --tasks "$CHAT_TASKS" \
    --apply_chat_template \
    --batch_size "$NUM_CONCURRENT" \
    --output_path "$out/chat" \
    --log_samples \
    "${limit_arg[@]}" 2>&1 | grep -E '^\||Error|error|Traceback' | grep -v -E '^\|-|Tasks' || true
fi

docker exec "$EVAL_CONTAINER" pkill -f "logproxy.py .* $PROXY_PORT " 2>/dev/null || true
bad=$(wc -l <"$proxy_log" 2>/dev/null || echo 0)
if (( bad > 0 )); then
  echo "[$(date -u +%FT%TZ)] EVAL INVALID $label: $bad non-200 responses (see $proxy_log)"
  exit 2
fi
echo "[$(date -u +%FT%TZ)] EVAL DONE $label"
