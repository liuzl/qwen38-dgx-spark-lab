#!/usr/bin/env bash
# Run from the same source/ after build.sh has finished. Short synthetic test only.
set -euo pipefail
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.3/bin:$PATH"
cmake -S . -B build-v100 -DSTRATA_PARITY_PROMPT_ATTN=ON
cmake --build build-v100 --target qsa_prompt_attn_parity -j 4
# Physical GPU 1 only. The test compares FP16/INT8 attention to an FP64 reference.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  timeout 180 build-v100/qsa_prompt_attn_parity 32768 128 2
