#!/usr/bin/env bash
# Vision projector only; leaves the verified text weights and download.status untouched.
set -euo pipefail
cd "${1:?experiment root required}"
F=models/mmproj-Qwen3.8-Flash-Next-BF16.gguf
SIZE=907543008
SHA=b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0
printf "running\n" > evidence/mmproj.status
trap "printf \"failed-or-interrupted\n\" > evidence/mmproj.status" EXIT
aria2c --input-file=evidence/mmproj-download.aria2 --dir=models \
  --continue=true --check-integrity=true --file-allocation=none \
  --split=8 --max-connection-per-server=8 --max-tries=12 --retry-wait=15 \
  --connect-timeout=30 --timeout=60 --summary-interval=300 \
  --console-log-level=warn --download-result=full
[ "$(stat -c %s "$F")" = "$SIZE" ] || { echo "Wrong size: $F"; exit 1; }
[ "$(sha256sum "$F" | cut -d" " -f1)" = "$SHA" ] || { echo "Wrong SHA256: $F"; exit 1; }
printf "{\n  \"repo\": \"ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF\",\n  \"revision\": \"ed59f92082b1e93c0e96d60a8b11aab089b52f09\",\n  \"file\": \"%s\",\n  \"bytes\": %s,\n  \"sha256\": \"%s\"\n}\n" "$(basename $F)" "$SIZE" "$SHA" > evidence/mmproj-verified.json
printf "verified\n" > evidence/mmproj.status
trap - EXIT
