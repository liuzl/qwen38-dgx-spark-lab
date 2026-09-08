#!/usr/bin/env bash
# Controlled serving arms on the A100 host. Stops the production container for
# the maintenance window, runs each arm in a fresh candidate process with the
# production env plus per-arm overrides, then restores production.
#
# usage: sweep-a100-mtp-arms.sh <label> '<speculative_config_json>' [KEY=VAL ...] \
#                               [<label> '<json>' [KEY=VAL ...] ...]
#
# Each arm is a label, a speculative config, and zero or more UPPERCASE
# KEY=VALUE overrides applied on top of $BASE/.multimodal.env before the
# candidate starts. Labels must not look like KEY=VALUE. An empty value clears
# a variable, for example ADAPTER_DIR= to serve without the LoRA alias.
#
# Environment:
#   CASES   benchmark shapes, name:repeats:concurrency (see benchmark-serving-matrix.py)
#   MODELS  served model names to benchmark; defaults to base + adapter alias
#   KERNEL_GATE  regex the server log must match after health, else the arm is
#                labelled "kernel gate FAILED" (default: none). May also be
#                given per arm as a KERNEL_GATE=regex override.
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
MODELS="${MODELS:-qwen3.8-27b,qwen3.8-27b-uncensored}"
KERNEL_GATE="${KERNEL_GATE:-}"

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
  overrides=()
  while (( $# >= 1 )) && [[ $1 =~ ^[A-Z_][A-Z0-9_]*= ]]; do
    overrides+=("$1")
    shift
  done
  arm_gate="$KERNEL_GATE"
  for pair in "${overrides[@]+"${overrides[@]}"}"; do
    [[ $pair == KERNEL_GATE=* ]] && arm_gate="${pair#KERNEL_GATE=}"
  done
  echo "[$(date -u +%FT%TZ)] arm $label spec=$spec overrides=${overrides[*]:-} extra_env=${EXTRA_DOCKER_ENV:-}"
  (
    set -a
    # shellcheck disable=SC1091
    source "$BASE/.multimodal.env"
    for pair in "${overrides[@]+"${overrides[@]}"}"; do
      export "${pair?}"
    done
    set +a
    CONTAINER=$CAND PORT=$CAND_PORT RESTART_POLICY=no SPECULATIVE_CONFIG="$spec" \
      bash "$LAB/scripts/serve-a100.sh"
  )
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
  # Record kernel routing so an arm cannot inherit a label it did not earn.
  docker logs "$CAND" 2>&1 \
    | grep -E 'Selected .*Kernel|ScaledMM|Marlin|GDN prefill kernel|attention backend|FlashAttention version|cudagraph_mode' \
    | sed 's/^/  kernel: /' || true
  # No grep -q: under pipefail an early exit would SIGPIPE docker logs and
  # report a false gate failure even when the kernel line is present.
  if [[ -n "$arm_gate" ]] && ! docker logs "$CAND" 2>&1 | grep -E "$arm_gate" >/dev/null; then
    echo "arm $label kernel gate FAILED: log does not match /$arm_gate/"
  fi
  "$PYTHON" "$LAB/scripts/benchmark-serving-matrix.py" \
    --base-url "http://127.0.0.1:$CAND_PORT" \
    --output "$LOGS/perf-$label.json" \
    --image "$LOGS/perf-image.png" \
    --label "$label" \
    --models "$MODELS" \
    --cases "$CASES" \
    || echo "arm $label benchmark FAILED"
  docker logs "$CAND" >"$LOGS/perf-$label-server.log" 2>&1
  docker rm -f "$CAND" >/dev/null
  echo "[$(date -u +%FT%TZ)] arm $label done"
done
echo "SWEEP DONE"
