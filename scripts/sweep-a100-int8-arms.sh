#!/usr/bin/env bash
# Phase 1 of docs/a100-prefill-int8-plan.md: base-alias prefill arms comparing
# the production FP8 (Marlin W8A16) service against the pinned SmoothQuant
# W8A8 INT8 checkpoint on CUTLASS INT8 Tensor Cores. Requires a maintenance
# window; production is stopped and restored by sweep-a100-mtp-arms.sh.
#
# The INT8 snapshot must already be in the HF cache; the candidate runs with
# HF_HUB_OFFLINE=1 like production. Verify with check-a100-int8-snapshot.sh.
#
# usage: sweep-a100-int8-arms.sh            # control + INT8 K7 + INT8 K3
#        ARMS=int8-k7 sweep-a100-int8-arms.sh   # any subset, space-separated
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INT8_MODEL="${INT8_MODEL:-Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8}"
INT8_REVISION="${INT8_REVISION:-2df4e3b00d4b865d59a7de0dc286fb18fd455a1e}"
K7='{"method":"mtp","num_speculative_tokens":7}'
K3='{"method":"mtp","num_speculative_tokens":3}'

# Shapes from the plan: short C1, long C1 (TTFT), C8, image C1, C32 aggregate.
export CASES="${CASES:-text_1k_c1:19:1,text_16k_c1:307:1,text_1k_c8:19:8,image_1k_c1:19:1,text_1k_c32:19:32}"
# Base alias only: the FP8-derived rank-1 adapter does not transfer to INT8.
export MODELS="${MODELS:-qwen3.8-27b}"

# INT8 arms drop the adapter, pin bf16 activations (author: fp16 breaks
# speculative acceptance on this hybrid target), and must route the W8A8
# groups to CUTLASS INT8. Everything else inherits the production env.
int8_overrides=(
  "MODEL=$INT8_MODEL"
  "MODEL_REVISION=$INT8_REVISION"
  "ADAPTER_DIR="
  "DTYPE=bfloat16"
  "KERNEL_GATE=CutlassInt8ScaledMM"
)
# The control must still be the Marlin FP8 path, base alias only.
fp8_overrides=(
  "ADAPTER_DIR="
  "KERNEL_GATE=MarlinFP8ScaledMMLinearKernel"
)

args=()
for arm in ${ARMS:-fp8-k7 int8-k7 int8-k3}; do
  case "$arm" in
    fp8-k7) args+=("$arm" "$K7" "${fp8_overrides[@]}") ;;
    int8-k7) args+=("$arm" "$K7" "${int8_overrides[@]}") ;;
    int8-k3) args+=("$arm" "$K3" "${int8_overrides[@]}") ;;
    *) echo "unknown arm: $arm (fp8-k7|int8-k7|int8-k3)" >&2; exit 1 ;;
  esac
done

exec bash "$here/sweep-a100-mtp-arms.sh" "${args[@]}"
