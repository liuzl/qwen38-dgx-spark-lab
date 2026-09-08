#!/usr/bin/env bash
# Phase 2 (quality gates) of docs/a100-prefill-int8-plan.md for the base alias.
# Stops production for the maintenance window, starts the INT8 W8A8 candidate
# with the production profile (MTP K7, no adapter), then runs on the candidate:
#   1. API smoke: Chat Completions, Responses, Anthropic Messages, forced tool
#   2. four greedy semantic canaries, summarised and compared with the FP8 K7
#      production capture
#   3. image inputs: OCR through all three protocols, remote URL denied,
#      five-image request denied
#   4. 64 requests at C32 stability
# Production is restored on exit. No response text leaves /run-logs.
#
# usage: qualify-a100-int8-candidate.sh            # RUN_ID defaults to date-stamped
set -euo pipefail
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.43}"
BASE="${A100_BASE:-/databank/zliu/qwen38-a100}"
LAB="$BASE/lab"
LOGS="$BASE/logs"
PROD="${PROD_CONTAINER:-qwen38-a100-native-lora}"
PROD_PORT="${PROD_PORT:-18103}"
CAND="${CAND_CONTAINER:-qwen38-a100-int8-qual}"
CAND_PORT="${CAND_PORT:-18105}"
RUN_ID="${RUN_ID:-int8-phase2-$(date -u +%Y%m%d)}"
INT8_MODEL="${INT8_MODEL:-Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8}"
INT8_REVISION="${INT8_REVISION:-2df4e3b00d4b865d59a7de0dc286fb18fd455a1e}"
SPEC="${SPEC:-{\"method\":\"mtp\",\"num_speculative_tokens\":7\}}"
MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"
IMAGE_FIXTURE="${IMAGE_FIXTURE:-$LOGS/perf-image.png}"   # renders the digits 7429
REFERENCE_CANARIES="${REFERENCE_CANARIES:-$LOGS/perf-k7-final-qwen3.8-27b-canaries.raw.json}"
KERNEL_GATE="${KERNEL_GATE:-CutlassInt8ScaledMM}"

wait_health() { # port, max seconds, container
  local waited=0
  until curl -fsS "http://127.0.0.1:$1/health" >/dev/null 2>&1; do
    sleep 10
    waited=$((waited + 10))
    if (( waited >= $2 )); then
      echo "health timeout on port $1" >&2
      return 1
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$3"; then
      echo "container $3 is not running" >&2
      return 1
    fi
  done
}

restore() {
  echo "[$(date -u +%FT%TZ)] restoring $PROD"
  docker logs "$CAND" >"$LOGS/$RUN_ID-server.log" 2>&1 || true
  docker rm -f "$CAND" >/dev/null 2>&1 || true
  docker start "$PROD" >/dev/null
  wait_health "$PROD_PORT" 1500 "$PROD" && echo "[$(date -u +%FT%TZ)] $PROD healthy"
}
trap restore EXIT

[[ -f "$IMAGE_FIXTURE" ]] || { echo "missing image fixture $IMAGE_FIXTURE" >&2; exit 1; }
[[ -f "$REFERENCE_CANARIES" ]] || { echo "missing reference canaries $REFERENCE_CANARIES" >&2; exit 1; }

echo "[$(date -u +%FT%TZ)] stopping $PROD for maintenance window"
docker stop "$PROD" >/dev/null

echo "[$(date -u +%FT%TZ)] starting candidate $CAND model=$INT8_MODEL@${INT8_REVISION:0:8} spec=$SPEC"
(
  set -a
  # shellcheck disable=SC1091
  source "$BASE/.multimodal.env"
  export MODEL="$INT8_MODEL" MODEL_REVISION="$INT8_REVISION" ADAPTER_DIR= DTYPE=bfloat16
  set +a
  CONTAINER=$CAND PORT=$CAND_PORT RESTART_POLICY=no SPECULATIVE_CONFIG="$SPEC" \
    bash "$LAB/scripts/serve-a100.sh"
)
t0=$(date +%s)
wait_health "$CAND_PORT" 1500 "$CAND"
echo "[$(date -u +%FT%TZ)] candidate healthy after $(( $(date +%s) - t0 ))s"
docker logs "$CAND" 2>&1 | grep -E 'Selected .*Kernel|GDN prefill kernel|FlashAttention version' | sed 's/^/  kernel: /' || true
if ! docker logs "$CAND" 2>&1 | grep -E "$KERNEL_GATE" >/dev/null; then
  echo "kernel gate FAILED: /$KERNEL_GATE/ not in server log; aborting" >&2
  exit 1
fi
echo "kernel gate passed"

url="http://127.0.0.1:$CAND_PORT"
status=0

echo "[$(date -u +%FT%TZ)] 1/4 API smoke"
if docker exec "$CAND" python3 /lab/scripts/validate-apis.py --base-url "$url" --model "$MODEL_NAME" \
    >"$LOGS/$RUN_ID-api-smoke.json"; then
  echo "api smoke: passed"
else
  echo "api smoke: FAILED"; status=1
fi

echo "[$(date -u +%FT%TZ)] 2/4 greedy canaries"
docker exec "$CAND" python3 /lab/scripts/capture-greedy-canaries.py \
  --base-url "$url" --model "$MODEL_NAME" --label "$RUN_ID" --max-tokens 256 \
  --output "/run-logs/$RUN_ID-canaries.raw.json"
python3 "$LAB/scripts/summarize-greedy-canaries.py" "$LOGS/$RUN_ID-canaries.raw.json" \
  --output "$LOGS/$RUN_ID-canaries.json"
if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["all_passed"] else 1)' \
    "$LOGS/$RUN_ID-canaries.json"; then
  echo "canaries: all passed"
else
  echo "canaries: FAILED"; status=1
  python3 -c 'import json,sys; [print("  ",c["name"],c["failures"]) for c in json.load(open(sys.argv[1]))["canaries"] if not c["passed"]]' "$LOGS/$RUN_ID-canaries.json"
fi
python3 "$LAB/scripts/compare-greedy-canaries.py" "$REFERENCE_CANARIES" "$LOGS/$RUN_ID-canaries.raw.json" \
  --output "$LOGS/$RUN_ID-canaries-vs-fp8-k7.json" || echo "canary comparison: could not run"

echo "[$(date -u +%FT%TZ)] 3/4 image inputs (direct, base alias)"
if python3 "$LAB/scripts/validate-multimodal.py" --direct --base-url "$url" \
    --image "$IMAGE_FIXTURE" --models "$MODEL_NAME" \
    --remote-url-probe "http://127.0.0.1:$CAND_PORT/health" \
    | tee "$LOGS/$RUN_ID-multimodal.txt"; then
  echo "image inputs: passed"
else
  echo "image inputs: FAILED"; status=1
fi

echo "[$(date -u +%FT%TZ)] 4/4 stability 64 @ C32"
if docker exec "$CAND" python3 /lab/scripts/validate-stability.py \
    --base-url "$url" --model "$MODEL_NAME" --requests 64 --concurrency 32 \
    >"$LOGS/$RUN_ID-stability.json"; then
  echo "stability: passed"
else
  echo "stability: FAILED"; status=1
fi

curl -fsS "$url/metrics" >"$LOGS/$RUN_ID-metrics.prom" || true
if (( status == 0 )); then
  echo "PHASE2 BASE GATES: PASSED ($RUN_ID)"
else
  echo "PHASE2 BASE GATES: FAILED ($RUN_ID)"
fi
exit $status
