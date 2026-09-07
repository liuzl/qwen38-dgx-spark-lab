#!/usr/bin/env bash
# Phase 0 gate for docs/a100-prefill-int8-plan.md: confirm the pinned INT8
# snapshot is complete in the HF cache and its quantization config matches the
# plan before any maintenance window is scheduled.
set -euo pipefail
HF_CACHE="${HF_CACHE:-/databank/zliu/qwen38-a100/cache/huggingface}"
INT8_MODEL="${INT8_MODEL:-Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8}"
INT8_REVISION="${INT8_REVISION:-2df4e3b00d4b865d59a7de0dc286fb18fd455a1e}"
EXPECTED_FILES="${EXPECTED_FILES:-49}"

repo_dir="$HF_CACHE/hub/models--${INT8_MODEL//\//--}"
snap="$repo_dir/snapshots/$INT8_REVISION"
fail=0
[[ -d "$snap" ]] || { echo "missing snapshot $snap"; exit 1; }

incomplete=$(find "$repo_dir/blobs" -name '*.incomplete' | wc -l)
files=$(find -L "$snap" -type f | wc -l)
echo "snapshot: $snap"
echo "files: $files (expected $EXPECTED_FILES)  incomplete blobs: $incomplete"
echo "size: $(du -shL "$snap" | cut -f1)"
(( incomplete == 0 )) || fail=1
(( files == EXPECTED_FILES )) || fail=1
[[ -f "$snap/model-mtp.safetensors" ]] && echo "mtp head: present" || { echo "mtp head: MISSING"; fail=1; }

python3 - "$snap/config.json" <<'PY' || fail=1
import json, sys
cfg = json.load(open(sys.argv[1]))
q = cfg.get("quantization_config", {})
groups = q.get("config_groups", {})
w8a8 = [g for g in groups.values() if g.get("input_activations")]
w8a16 = [g for g in groups.values() if not g.get("input_activations")]
ok = True
def check(cond, msg):
    global ok
    print(("ok   " if cond else "FAIL ") + msg)
    ok = ok and cond
check(cfg.get("architectures") == ["Qwen3_5ForConditionalGeneration"], "architecture Qwen3_5ForConditionalGeneration")
check(q.get("format") == "mixed-precision", "compressed-tensors mixed-precision format")
check(len(w8a8) == 1 and w8a8[0]["input_activations"]["num_bits"] == 8, "one W8A8 group with INT8 activations")
check(len(w8a16) == 1 and w8a16[0]["format"] == "pack-quantized", "one W8A16 pack-quantized group")
check(any("mtp" in i for i in q.get("ignore", [])), "mtp.* kept in higher precision")
sys.exit(0 if ok else 1)
PY

if (( fail )); then echo "INT8 SNAPSHOT GATE: FAILED"; exit 1; fi
echo "INT8 SNAPSHOT GATE: PASSED"
