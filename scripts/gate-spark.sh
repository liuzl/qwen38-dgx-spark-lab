#!/usr/bin/env bash
# Functional + performance gates for a running Qwen3.8 native-LoRA vLLM server.
#
# usage: IMAGE_FIXTURE=/path/ocr.png gate-spark.sh <label> [base_url] [reference_label]
#   label            results go to $RESULTS_DIR/gate-<label>/ (default ./results)
#   base_url         default http://127.0.0.1:18102
#   reference_label  earlier gate run to compare canaries against (optional)
#   IMAGE_FIXTURE    PNG that renders the digits 7429 (validate-multimodal.py)
#
# Hard gates (exit 1 if any fails):
#   1 API smoke: Chat Completions, Responses, Messages, forced tool, per alias
#   2 greedy semantic canaries (4), per alias
#   3 image OCR through all three protocols, per alias
#   4 base/adapter prefix-cache isolation
#   5 fresh long-prompt prefix reuse >= 50% (validate-prefix-reuse.py)
#   6 stability: 64 marker-echo requests at C8, per alias
# Informational (recorded, not gating):
#   - canary text equivalence vs reference (hashes only)
#   - remote-URL / image-count guardrail sub-checks of validate-multimodal.py;
#     they only pass on servers that configure those limits
#   - perf: 1K C1 and 16K C1 (base x2, adapter 16K x2); engine facts from logs
# Raw canary captures contain response text; keep RESULTS_DIR out of git.
set -uo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="${1:?label required}"
URL="${2:-http://127.0.0.1:18102}"
REF="${3:-}"
: "${IMAGE_FIXTURE:?set IMAGE_FIXTURE to a PNG rendering 7429}"
BASE_MODEL="${BASE_MODEL_NAME:-qwen3.8-27b}"
ADAPTER_MODEL="${ADAPTER_MODEL_NAME:-qwen3.8-27b-uncensored}"
CONTAINER="${CONTAINER:-qwen38-vllm-native-lora}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/results}"
OUT="$RESULTS_DIR/gate-$LABEL"
mkdir -p "$OUT"
aliases=("$BASE_MODEL" "$ADAPTER_MODEL")
status=0
declare -A gate
log() { echo "[$(date -u +%FT%TZ)] $*"; }
mark() { gate[$1]=$2; [[ $2 == passed ]] || status=1; log "$1: $2"; }
py() { python3 "$SCRIPTS/$1" "${@:2}"; }

curl -fsS "$URL/health" >/dev/null || { echo "server not healthy at $URL" >&2; exit 2; }
docker logs "$CONTAINER" 2>&1 | grep -E "Initializing a V1 LLM engine|GDN prefill kernel|GPU KV cache size|kv cache group sizes|no KV cache group" \
  | tail -5 | cut -c1-220 >"$OUT/engine-facts.txt"

log "1/6 API smoke"
api_args=(); for a in "${aliases[@]}"; do api_args+=(--model "$a"); done
if py validate-apis.py --base-url "$URL" "${api_args[@]}" >"$OUT/api-smoke.json" 2>"$OUT/api-smoke.err"; then
  mark api passed; else mark api FAILED; fi

log "2/6 greedy canaries"
canary_ok=passed
for a in "${aliases[@]}"; do
  raw="$OUT/$a-canaries.raw.json"
  py capture-greedy-canaries.py --base-url "$URL" --model "$a" --label "$LABEL-$a" --max-tokens 512 --output "$raw" \
    >/dev/null 2>"$OUT/$a-canaries.err" || canary_ok=FAILED
  py summarize-greedy-canaries.py "$raw" --output "$OUT/$a-canaries.json" >/dev/null 2>&1 || canary_ok=FAILED
  python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["all_passed"] else 1)' "$OUT/$a-canaries.json" 2>/dev/null \
    || canary_ok=FAILED
  if [[ -n "$REF" && -f "$RESULTS_DIR/gate-$REF/$a-canaries.raw.json" ]]; then
    py compare-greedy-canaries.py "$RESULTS_DIR/gate-$REF/$a-canaries.raw.json" "$raw" \
      --output "$OUT/$a-canaries-vs-$REF.json" >/dev/null 2>&1 || true
  fi
done
mark canaries "$canary_ok"

log "3/6 image OCR"
py validate-multimodal.py --direct --base-url "$URL" --image "$IMAGE_FIXTURE" --models "${aliases[@]}" \
  --remote-url-probe "$URL/health" >"$OUT/multimodal.txt" 2>&1 || true
ocr_ok=$(grep -c -E "^ok: .* image$" "$OUT/multimodal.txt")
if [[ "$ocr_ok" -ge $((3 * ${#aliases[@]})) ]]; then mark image passed; else mark image "FAILED ($ocr_ok OCR ok)"; fi

log "4/6 prefix-cache isolation"
if py validate-cache-isolation.py --base-url "$URL" --base-model "$BASE_MODEL" --adapter-model "$ADAPTER_MODEL" \
    >"$OUT/cache-isolation.json" 2>&1; then mark isolation passed; else mark isolation FAILED; fi

log "5/6 prefix reuse"
if py validate-prefix-reuse.py --base-url "$URL" --model "$BASE_MODEL" >"$OUT/prefix-reuse.json" 2>&1; then
  mark prefix_reuse passed; else mark prefix_reuse FAILED; fi

log "6/6 stability 64 @ C8"
stab_ok=passed
for a in "${aliases[@]}"; do
  py validate-stability.py --base-url "$URL" --model "$a" --requests 64 --concurrency 8 \
    >"$OUT/$a-stability.json" 2>&1 || stab_ok=FAILED
done
mark stability "$stab_ok"

log "perf (informational)"
py benchmark-serving-matrix.py --base-url "$URL" --output "$OUT/perf-base.json" --image "$IMAGE_FIXTURE" \
  --label "$LABEL" --repetitions 2 --models "$BASE_MODEL" --cases text_1k_c1:19:1,text_16k_c1:307:1 \
  >"$OUT/perf-base.log" 2>&1 || log "perf base: incomplete"
py benchmark-serving-matrix.py --base-url "$URL" --output "$OUT/perf-adapter.json" --image "$IMAGE_FIXTURE" \
  --label "$LABEL" --repetitions 2 --models "$ADAPTER_MODEL" --cases text_16k_c1:307:1 \
  >"$OUT/perf-adapter.log" 2>&1 || log "perf adapter: incomplete"

python3 - "$OUT" "$status" "${gate[api]}" "${gate[canaries]}" "${gate[image]}" "${gate[isolation]}" \
  "${gate[prefix_reuse]}" "${gate[stability]}" <<'EOF'
import json, statistics, sys
from pathlib import Path
out = Path(sys.argv[1])
names = ["api", "canaries", "image", "isolation", "prefix_reuse", "stability"]
summary = {"gates": dict(zip(names, sys.argv[3:])), "all_passed": sys.argv[2] == "0",
           "engine_facts": (out / "engine-facts.txt").read_text().splitlines()}
perf = {}
for name in ("perf-base.json", "perf-adapter.json"):
    try:
        rows = json.loads((out / name).read_text())["rows"]
    except Exception:
        continue
    for r in rows:
        perf.setdefault(f'{r["model"]} {r["case"]["name"]}', []).append(
            (r["request_decode_tok_s_mean"], r["ttft_s_mean"]))
summary["perf_median"] = {
    k: {"decode_tok_s": round(statistics.median(x[0] for x in v), 1),
        "ttft_s": round(statistics.median(x[1] for x in v), 2), "n": len(v)}
    for k, v in perf.items()}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
EOF
exit "$status"
