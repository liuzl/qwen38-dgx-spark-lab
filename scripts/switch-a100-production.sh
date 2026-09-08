#!/usr/bin/env bash
# Switch the A100 production service to a new target checkpoint + adapter with
# automatic rollback. Used on 2026-09-08 to move from the official FP8 (Marlin
# W8A16) checkpoint to the SmoothQuant INT8 W8A8 checkpoint.
#
#   1. back up .multimodal.env, rewrite MODEL / MODEL_REVISION / ADAPTER_DIR / DTYPE
#   2. stop the current production container and RENAME it (kept for rollback)
#   3. launch the new production container from the updated env (same name,
#      same restart policy, same ports) and wait for health
#   4. gate: kernel line, API smoke (both aliases), canaries (both), image
#      protocols + guardrails (both), prefix-cache isolation, 64 @ C32 (both),
#      short performance replay (1K and 16K C1, 1 repetition)
#   5. any failure in 3-4 -> stop new container, restore env backup, rename the
#      old container back and start it (rollback), exit non-zero
#
# usage: switch-a100-production.sh   (env: NEW_MODEL NEW_REVISION NEW_ADAPTER_DIR
#        NEW_DTYPE KERNEL_GATE ROLLBACK_SUFFIX)
set -euo pipefail
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.43}"
BASE="${A100_BASE:-/databank/zliu/qwen38-a100}"
LAB="$BASE/lab"; LOGS="$BASE/logs"
ENV_FILE="$BASE/.multimodal.env"
NEW_MODEL="${NEW_MODEL:-Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8}"
NEW_REVISION="${NEW_REVISION:-2df4e3b00d4b865d59a7de0dc286fb18fd455a1e}"
NEW_ADAPTER_DIR="${NEW_ADAPTER_DIR:-$BASE/artifacts/adapter-int8-2df4e3b0-v2}"
NEW_DTYPE="${NEW_DTYPE:-bfloat16}"
KERNEL_GATE="${KERNEL_GATE:-CutlassInt8ScaledMM}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ROLLBACK_SUFFIX="${ROLLBACK_SUFFIX:-fp8-pre-int8-$STAMP}"
RUN_ID="prod-switch-$STAMP"
IMAGE_FIXTURE="${IMAGE_FIXTURE:-$LOGS/perf-image.png}"

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
PROD="$CONTAINER"; PORT_="$PORT"; BASE_ALIAS="$SERVED_MODEL_NAME"; ADAPTER_ALIAS="$ADAPTER_MODEL_NAME"
OLD_NAME="$PROD-$ROLLBACK_SUFFIX"
url="http://127.0.0.1:$PORT_"

wait_health() { # port, max seconds, container
  local waited=0
  until curl -fsS "http://127.0.0.1:$1/health" >/dev/null 2>&1; do
    sleep 10; waited=$((waited + 10))
    (( waited >= $2 )) && { echo "health timeout on port $1" >&2; return 1; }
    docker ps --format '{{.Names}}' | grep -qx "$3" || { echo "container $3 is not running" >&2; return 1; }
  done
}

switched=0
rollback() {
  echo "[$(date -u +%FT%TZ)] ROLLBACK: restoring previous production"
  docker logs "$PROD" >"$LOGS/$RUN_ID-failed-server.log" 2>&1 || true
  docker rm -f "$PROD" >/dev/null 2>&1 || true
  cp "$ENV_FILE.before-$RUN_ID" "$ENV_FILE"
  docker rename "$OLD_NAME" "$PROD"
  docker update --restart unless-stopped "$PROD" >/dev/null
  docker start "$PROD" >/dev/null
  wait_health "$PORT_" 1500 "$PROD" && echo "[$(date -u +%FT%TZ)] previous production healthy again"
  echo "PRODUCTION SWITCH: ROLLED BACK ($RUN_ID)"
}
on_exit() {
  local rc=$?
  if (( rc != 0 )) && (( switched == 1 )); then rollback; fi
}
trap on_exit EXIT

[[ -f "$NEW_ADAPTER_DIR/adapter_model.safetensors" ]] || { echo "missing adapter $NEW_ADAPTER_DIR" >&2; exit 1; }
[[ -d "$HF_CACHE/hub/models--${NEW_MODEL//\//--}/snapshots/$NEW_REVISION" ]] || { echo "missing snapshot for $NEW_MODEL@$NEW_REVISION" >&2; exit 1; }
[[ -f "$IMAGE_FIXTURE" ]] || { echo "missing image fixture" >&2; exit 1; }

echo "[$(date -u +%FT%TZ)] $RUN_ID: $MODEL@${MODEL_REVISION:0:8} -> $NEW_MODEL@${NEW_REVISION:0:8}, adapter $(basename "$NEW_ADAPTER_DIR")"
cp "$ENV_FILE" "$ENV_FILE.before-$RUN_ID"
python3 - "$ENV_FILE" "$NEW_MODEL" "$NEW_REVISION" "$NEW_ADAPTER_DIR" "$NEW_DTYPE" <<'PY'
import re, sys
path, model, rev, adapter, dtype = sys.argv[1:]
text = open(path).read()
for key, val in (("MODEL", model), ("MODEL_REVISION", rev), ("ADAPTER_DIR", adapter), ("DTYPE", dtype)):
    text, n = re.subn(rf"^{key}=.*$", f"{key}={val}", text, flags=re.M)
    assert n == 1, f"{key} not found exactly once"
open(path, "w").write(text)
PY
grep -E '^(MODEL|MODEL_REVISION|ADAPTER_DIR|DTYPE)=' "$ENV_FILE" | sed 's/^/  env: /'

echo "[$(date -u +%FT%TZ)] stopping and renaming $PROD -> $OLD_NAME"
docker stop "$PROD" >/dev/null
docker rename "$PROD" "$OLD_NAME"
docker update --restart no "$OLD_NAME" >/dev/null
switched=1

echo "[$(date -u +%FT%TZ)] launching new production"
( set -a; source "$ENV_FILE"; set +a; bash "$LAB/scripts/serve-a100.sh" )
t0=$(date +%s)
wait_health "$PORT_" 1500 "$PROD"
echo "[$(date -u +%FT%TZ)] new production healthy after $(( $(date +%s) - t0 ))s"
docker logs "$PROD" 2>&1 | grep -E 'Selected .*Kernel|GDN prefill kernel|FlashAttention version|GPU KV cache size' | sed 's/^/  kernel: /' || true
docker logs "$PROD" 2>&1 | grep -E "$KERNEL_GATE" >/dev/null || { echo "kernel gate FAILED" >&2; exit 1; }
echo "kernel gate passed"

echo "[$(date -u +%FT%TZ)] gate: API smoke"
docker exec "$PROD" python3 /lab/scripts/validate-apis.py --base-url "$url" --model "$BASE_ALIAS" --model "$ADAPTER_ALIAS" >"$LOGS/$RUN_ID-api-smoke.json"
echo "api smoke: passed"
for alias in "$BASE_ALIAS" "$ADAPTER_ALIAS"; do
  echo "[$(date -u +%FT%TZ)] gate: canaries ($alias)"
  docker exec "$PROD" python3 /lab/scripts/capture-greedy-canaries.py --base-url "$url" --model "$alias" --label "$RUN_ID-$alias" --max-tokens 512 --output "/run-logs/$RUN_ID-$alias-canaries.raw.json"
  python3 "$LAB/scripts/summarize-greedy-canaries.py" "$LOGS/$RUN_ID-$alias-canaries.raw.json" --output "$LOGS/$RUN_ID-$alias-canaries.json"
  python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["all_passed"] else 1)' "$LOGS/$RUN_ID-$alias-canaries.json" || { echo "canaries ($alias) FAILED" >&2; exit 1; }
  echo "canaries ($alias): passed"
done
echo "[$(date -u +%FT%TZ)] gate: image inputs"
python3 "$LAB/scripts/validate-multimodal.py" --direct --base-url "$url" --image "$IMAGE_FIXTURE" --models "$BASE_ALIAS" "$ADAPTER_ALIAS" --remote-url-probe "http://127.0.0.1:$PORT_/health" | tee "$LOGS/$RUN_ID-multimodal.txt"
echo "[$(date -u +%FT%TZ)] gate: prefix-cache isolation"
docker exec "$PROD" python3 /lab/scripts/validate-cache-isolation.py --base-url "$url" --base-model "$BASE_ALIAS" --adapter-model "$ADAPTER_ALIAS" >"$LOGS/$RUN_ID-cache-isolation.json"
echo "cache isolation: passed"
for alias in "$BASE_ALIAS" "$ADAPTER_ALIAS"; do
  echo "[$(date -u +%FT%TZ)] gate: stability 64 @ C32 ($alias)"
  docker exec "$PROD" python3 /lab/scripts/validate-stability.py --base-url "$url" --model "$alias" --requests 64 --concurrency 32 >"$LOGS/$RUN_ID-$alias-stability.json"
  echo "stability ($alias): passed"
done
echo "[$(date -u +%FT%TZ)] performance replay (1K/16K C1, 1 repetition)"
"${PYTHON:-/databank/zliu/miniconda3/bin/python3}" "$LAB/scripts/benchmark-serving-matrix.py" \
  --base-url "$url" --output "$LOGS/$RUN_ID-replay.json" --image "$IMAGE_FIXTURE" --label "$RUN_ID" \
  --models "$BASE_ALIAS,$ADAPTER_ALIAS" --cases "text_1k_c1:19:1,text_16k_c1:307:1" --repetitions 1 >/dev/null
python3 - "$LOGS/$RUN_ID-replay.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))["rows"]
for r in rows:
    print(f"  replay {r['model']:26} {r['case']['name']:12} ttft={r['ttft_s_mean']:.3f}s decode={r['request_decode_tok_s_mean']:.1f} tok/s failures={len(r['failures'])}")
PY
switched=0   # past the point of rollback: success
echo "PRODUCTION SWITCH: DONE ($RUN_ID). Previous container kept as $OLD_NAME (restart=no). Env backup: $ENV_FILE.before-$RUN_ID"
