#!/usr/bin/env bash
# RunPod autostart bootstrap for Qwen3.8-27B NVFP4 + DFlash2 on vLLM v0.28.0.
#
# Used as the template env BOOTSTRAP with image vllm/vllm-openai:v0.28.0,
# entrypoint `bash -c` and start command `eval "$BOOTSTRAP"`. Required env:
#   LAB_COMMIT         commit of this repo to fetch docker/patch_vllm_dflash_v028.py from
#   PATCH_SHA256       expected sha256 of that patch (bootstrap stops on mismatch)
#   FINGERPRINT_VALUE  deployment ID returned as system_fingerprint
# Optional: KV_BYTES (default 12 GiB, the MIG 48GB setting). Serves the base
# alias only on 127.0.0.1:18102; the private adapter is not downloaded.
# Progress goes to /workspace/timeline.log. First measured 2026-09-29:
# order -> healthy 10.5 min on PRO 6000 MIG 48GB.
set -u
W=/workspace
mkdir -p $W/run $W/models $W/cache/flashinfer
exec > >(tee -a $W/bootstrap.log) 2>&1
ts() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $W/timeline.log; }
ts bootstrap-start

# sshd in the background so weights start downloading immediately.
(
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server >/dev/null
  mkdir -p /root/.ssh /run/sshd && chmod 700 /root/.ssh
  printf '%s\n' "${PUBLIC_KEY:-}" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
  /usr/sbin/sshd && ts sshd-up
) &

python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo, rev, d in [("RadixArk/Qwen3.8-27B-NVFP4", "319f741cce68d7914884900c138a1fbb70a42f30", "RadixArk-Qwen3.8-27B-NVFP4"),
                     ("z-lab/Qwen3.8-27B-DFlash2", "50307d4c4cde6860d4eee73e2547cd786fe8e8a4", "Qwen3.8-27B-DFlash2")]:
    snapshot_download(repo, revision=rev, local_dir=f"/workspace/models/{d}", max_workers=16)
    print("downloaded", repo, rev, flush=True)
PY
ts weights-ready

# Pinned DFlash2 compile-cache fix from the public lab repo; fail closed on hash mismatch.
python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "https://raw.githubusercontent.com/liuzl/qwen38-dgx-spark-lab/${LAB_COMMIT}/docker/patch_vllm_dflash_v028.py" $W/run/patch.py
echo "${PATCH_SHA256}  $W/run/patch.py" | sha256sum -c - || { ts patch-hash-mismatch; sleep infinity; }
SITE=$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')
python3 $W/run/patch.py --site "$SITE" && python3 -m py_compile "$SITE/config/speculative.py" && ts patched

ln -sfn $W/cache/flashinfer /root/.cache/flashinfer
export VLLM_MARLIN_USE_ATOMIC_ADD=1 VLLM_PREFIX_CACHE_RETENTION_INTERVAL=1648 VLLM_CACHE_ROOT=$W/cache/prob-k7
nohup vllm serve $W/models/RadixArk-Qwen3.8-27B-NVFP4 --served-model-name qwen3.8-27b --host 127.0.0.1 --port 18102 \
  --max-model-len 131072 --gpu-memory-utilization 0.95 --kv-cache-memory-bytes ${KV_BYTES:-12884901888} \
  --max-num-seqs 10 --max-num-batched-tokens 16384 --enable-prefix-caching --enable-chunked-prefill \
  --kv-cache-dtype fp8_e4m3 --no-enable-flashinfer-autotune --trust-remote-code --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml --enable-auto-tool-choice --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-prompt-tokens-details --fingerprint-mode custom --fingerprint-value "${FINGERPRINT_VALUE}" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$W/models/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":1,\"draft_sample_method\":\"probabilistic\"}" \
  > $W/run/vllm.log 2>&1 &
VLLM_PID=$!
ts vllm-started
until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18102/health', timeout=3)" 2>/dev/null; do
  kill -0 $VLLM_PID 2>/dev/null || { ts vllm-exited; sleep infinity; }
  sleep 5
done
ts vllm-healthy
sleep infinity
