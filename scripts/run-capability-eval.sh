#!/usr/bin/env bash
# Capability evaluation of one served alias through lm-evaluation-harness,
# driven from the CPU-only qwen38-lm-eval container against a running vLLM
# endpoint. Used to compare the FP8 and INT8 targets, base and adapter aliases,
# with the same harness, prompts, and few-shot settings.
#
# Budget: about 12-15 minutes per alias on the A100 profile. This is a
# regression smoke across quantization/adapter arms, not a leaderboard run:
# every arm sees the same deterministic subset (lm-eval --limit takes the
# first N documents per task), so differences between arms are comparable,
# but absolute numbers carry roughly +-1 point of sampling error.
#
# Stage 1 (completions endpoint, no chat template):
#   mmlu 0-shot, 25 questions per subject (1,425 questions, 5.7k prompts)
#   truthfulqa_mc2 0-shot, 200 questions
#   0-shot for multiple choice is deliberate: vLLM computes prompt logprobs with
#   full-vocabulary logits per prompt token and cannot use the prefix cache, so
#   few-shot contexts multiply cost 5-25x (MMLU 5-shot measured at ~25 h). Full
#   MMLU 0-shot alone took 60 min. Prompts of 241-255 tokens hit the vLLM
#   0.28.0 NaN prompt-logprob band; eval-logproxy.py pads those and reports it.
#   Dropped as low-signal for this purpose: hellaswag (40k prompts), arc
#   (0-shot completion scoring is distorted on this chat model), winogrande.
# Stage 2 (chat completions with the served chat template, thinking off):
#   gsm8k 5-shot as multi-turn chat, 300 questions (strict/flexible match)
#   ifeval 0-shot, 200 prompts (instruction following)
#   gsm8k must go through the chat template: on the raw completions path the
#   server's enable_thinking=false default does not apply, the model emits a
#   <think> block and exhausts the 256-token generation cap before the answer
#   (FP8 base measured 68% that way; 88 of 96 misses had no final answer).
#
# Task specs are task:fewshot[:limit]; LIMIT overrides every limit (debugging).
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
STAGE1="${STAGE1:-mmlu:0:25 truthfulqa_mc2:0:200}"
CHAT_TASKS="${CHAT_TASKS:-gsm8k:5:300 ifeval:0:200}"
NUM_CONCURRENT="${NUM_CONCURRENT:-16}"
BATCH="${BATCH:-8}"          # prompts per completions request
TIMEOUT="${TIMEOUT:-600}"    # seconds per request
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.8-27B-FP8}"
LIMIT="${LIMIT:-}"
PROXY_PORT="${PROXY_PORT:-18199}"
[[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad label" >&2; exit 1; }

out="/eval/results/$label"
proxy_log="$EVAL_DIR/results/$label.non200.jsonl"
mkdir -p "$EVAL_DIR/results"
rm -f "$proxy_log" "${proxy_log/.non200.jsonl/.padded.jsonl}"

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
  IFS=: read -r task shots tlimit <<<"$spec"
  limit_arg=(); [[ -n "${LIMIT:-$tlimit}" ]] && limit_arg=(--limit "${LIMIT:-$tlimit}")
  echo "[$(date -u +%FT%TZ)] stage 1: $task ($shots-shot, limit ${LIMIT:-${tlimit:-none}})"
  docker exec "$EVAL_CONTAINER" lm_eval \
    --model local-completions \
    --model_args "model=$model,base_url=$purl/v1/completions,num_concurrent=$NUM_CONCURRENT,max_retries=3,timeout=$TIMEOUT,tokenized_requests=False,tokenizer=$TOKENIZER,tokenizer_backend=huggingface" \
    --tasks "$task" --num_fewshot "$shots" \
    --batch_size "$BATCH" \
    --output_path "$out/$task" \
    --log_samples \
    "${limit_arg[@]}" 2>&1 | grep -E '^\||Error|error|Traceback' | grep -v -E '^\|-|Tasks' || true
  echo "  non-200 so far: $(wc -l <"$proxy_log" 2>/dev/null || echo 0)"
done

for spec in $CHAT_TASKS; do
  IFS=: read -r task shots tlimit <<<"$spec"
  limit_arg=(); [[ -n "${LIMIT:-$tlimit}" ]] && limit_arg=(--limit "${LIMIT:-$tlimit}")
  fewshot_args=(--num_fewshot "$shots"); (( shots > 0 )) && fewshot_args+=(--fewshot_as_multiturn)
  echo "[$(date -u +%FT%TZ)] stage 2: $task ($shots-shot chat, limit ${LIMIT:-${tlimit:-none}})"
  docker exec "$EVAL_CONTAINER" lm_eval \
    --model local-chat-completions \
    --model_args "model=$model,base_url=$purl/v1/chat/completions,num_concurrent=$NUM_CONCURRENT,max_retries=3,timeout=$TIMEOUT,tokenized_requests=False,tokenizer=$TOKENIZER,tokenizer_backend=huggingface" \
    --tasks "$task" "${fewshot_args[@]}" \
    --apply_chat_template \
    --batch_size "$BATCH" \
    --output_path "$out/chat-$task" \
    --log_samples \
    "${limit_arg[@]}" 2>&1 | grep -E '^\||Error|error|Traceback' | grep -v -E '^\|-|Tasks' || true
  echo "  non-200 so far: $(wc -l <"$proxy_log" 2>/dev/null || echo 0)"
done

docker exec "$EVAL_CONTAINER" pkill -f "logproxy.py .* $PROXY_PORT " 2>/dev/null || true
padded=$(wc -l <"${proxy_log/.non200.jsonl/.padded.jsonl}" 2>/dev/null || echo 0)
echo "[$(date -u +%FT%TZ)] padded NaN-band requests: $padded"
bad=$(wc -l <"$proxy_log" 2>/dev/null || echo 0)
if (( bad > 0 )); then
  echo "[$(date -u +%FT%TZ)] EVAL INVALID $label: $bad non-200 responses (see $proxy_log)"
  exit 2
fi
echo "[$(date -u +%FT%TZ)] EVAL DONE $label"
