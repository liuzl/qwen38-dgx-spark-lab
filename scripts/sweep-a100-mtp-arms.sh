#!/usr/bin/env bash
# Controlled MTP arms on the A100 host. Stops the production container for the
# maintenance window, runs each arm in a fresh candidate process with the
# production env plus an overridden SPECULATIVE_CONFIG, then restores production.
#
# usage: sweep-a100-mtp-arms.sh <label> '<speculative_config_json>' [<label> '<json>' ...]
set -euo pipefail
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.43}"
BASE="${A100_BASE:-/databank/zliu/qwen38-a100}"
LAB="$BASE/lab"
LOGS="$BASE/logs"
PROD="${PROD_CONTAINER:-qwen38-a100-native-lora}"
PROD_PORT="${PROD_PORT:-18103}"
CAND="${CAND_CONTAINER:-qwen38-a100-perf}"
CAND_PORT="${CAND_PORT:-18105}"
PYTHON="${PYTHON:-/databank/zliu/miniconda3/bin/python3}"
CASES="${CASES:-text_1k_c1:19:1,text_1k_c8:19:8,text_1k_c16:19:16,text_1k_c32:19:32}"

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
  docker rm -f "$CAND" >/dev/null 2>&1 || true
  docker start "$PROD" >/dev/null
  wait_health "$PROD_PORT" 1500 "$PROD" && echo "[$(date -u +%FT%TZ)] $PROD healthy"
}
trap restore EXIT

echo "[$(date -u +%FT%TZ)] stopping $PROD for maintenance window"
docker stop "$PROD" >/dev/null

while (( $# >= 2 )); do
  label=$1
  spec=$2
  shift 2
  echo "[$(date -u +%FT%TZ)] arm $label spec=$spec extra_env=${EXTRA_DOCKER_ENV:-}"
  set -a
  # shellcheck disable=SC1091
  source "$BASE/.multimodal.env"
  set +a
  CONTAINER=$CAND PORT=$CAND_PORT RESTART_POLICY=no SPECULATIVE_CONFIG="$spec" \
    bash "$LAB/scripts/serve-a100.sh"
  t0=$(date +%s)
  if ! wait_health "$CAND_PORT" 1500 "$CAND"; then
    docker logs "$CAND" >"$LOGS/perf-$label-server.log" 2>&1 || true
    docker rm -f "$CAND" >/dev/null 2>&1 || true
    echo "arm $label failed to start"
    continue
  fi
  echo "[$(date -u +%FT%TZ)] arm $label healthy after $(( $(date +%s) - t0 ))s"
  curl -s "http://127.0.0.1:$CAND_PORT/v1/models" \
    | python3 -c 'import json,sys;print([m["id"] for m in json.load(sys.stdin)["data"]])'
  "$PYTHON" "$LAB/scripts/benchmark-serving-matrix.py" \
    --base-url "http://127.0.0.1:$CAND_PORT" \
    --output "$LOGS/perf-$label.json" \
    --image "$LOGS/perf-image.png" \
    --label "$label" \
    --cases "$CASES" \
    || echo "arm $label benchmark FAILED"
  docker logs "$CAND" >"$LOGS/perf-$label-server.log" 2>&1
  docker rm -f "$CAND" >/dev/null
  echo "[$(date -u +%FT%TZ)] arm $label done"
done
echo "SWEEP DONE"
