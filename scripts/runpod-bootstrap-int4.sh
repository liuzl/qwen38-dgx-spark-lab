#!/usr/bin/env bash
# RunPod autostart bootstrap for Qwen3.8-27B RedHat INT4 (W4A16) + MTP on vLLM v0.29.0,
# for Ampere 48 GB cards (RTX A6000, A40) that cannot run the NVFP4 checkpoint.
#
# Used as the template env BOOTSTRAP with image vllm/vllm-openai:v0.29.0,
# entrypoint `bash -c` and start command `eval "$BOOTSTRAP"`. Required env:
#   FINGERPRINT_VALUE  deployment ID returned as system_fingerprint
# Optional: KV_BYTES (default 16 GiB; bf16 KV, since Ampere has no FP8 KV).
# Optional adapter (serves qwen3.8-27b-uncensored next to the base alias):
#   ARTIFACT_URL       URL of the adapter tar (tokenflow tf-artifacts Worker)
#   ARTIFACT_TOKEN     bearer token, e.g. "{{ RUNPOD_SECRET_tf_artifacts_token }}"
#   ADAPTER_TAR_SHA256 / ADAPTER_SHA256  expected hashes (tar, adapter_model.safetensors)
# The adapter for this checkpoint is the rank-1 adapter with lora_B scaled by 0.85
# (adapter_model.safetensors sha256 7dc18d34...), not the NVFP4 one: at 1.0 it costs
# 3.7 Global-MMLU-Lite points on INT4, at 0.85 1.2 points with refusals still removed.
# Without ARTIFACT_TOKEN only the base alias is served. Listens on 127.0.0.1:18102.
# Optional data-plane tunnel (tokenflow ADR-0009 §2), started once vLLM is healthy:
#   TUNNEL_TOKEN       Cloudflare Tunnel token, e.g. "{{ RUNPOD_SECRET_tf_runpod_tunnel_token }}"
#   CLOUDFLARED_URL / CLOUDFLARED_SHA256  pinned static cloudflared binary and its hash
# Progress goes to /workspace/timeline.log.
#
# 2026-10-12 measurements (A6000, oplog logs/changes/2026-10-12-runpod-qwen38-27b-int4*.md):
# 131072 context, 16 images, KV 16 GiB = 209,715 tokens; code 1K single stream 126 tok/s.
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
for repo, rev, d in [("RedHatAI/Qwen3.8-27B-INT4", "91bd022d5b49442a868bc35008f6c21e1860edfa", "RedHatAI-Qwen3.8-27B-INT4")]:
    snapshot_download(repo, revision=rev, local_dir=f"/workspace/models/{d}", max_workers=16)
    print("downloaded", repo, rev, flush=True)
PY
ts weights-ready

lora_args=()
if [[ -n "${ARTIFACT_TOKEN:-}" ]]; then
  python3 - <<'PY' || { ts adapter-failed; sleep infinity; }
import hashlib, os, tarfile, urllib.request
req = urllib.request.Request(os.environ["ARTIFACT_URL"], headers={"Authorization": "Bearer " + os.environ["ARTIFACT_TOKEN"], "User-Agent": "tokenflow-bootstrap/1"})
data = urllib.request.urlopen(req, timeout=300).read()
assert hashlib.sha256(data).hexdigest() == os.environ["ADAPTER_TAR_SHA256"], "adapter tar hash mismatch"
open("/workspace/adapter.tar", "wb").write(data)
with tarfile.open("/workspace/adapter.tar") as t:
    t.extractall("/workspace/models", filter="data")
got = hashlib.sha256(open("/workspace/models/adapter/adapter_model.safetensors", "rb").read()).hexdigest()
assert got == os.environ["ADAPTER_SHA256"], "adapter weights hash mismatch"
os.remove("/workspace/adapter.tar")
print("adapter ready", got[:12], flush=True)
PY
  lora_args=(--enable-lora --max-loras 1 --max-lora-rank 1 --lora-dtype bfloat16
             --lora-modules qwen3.8-27b-uncensored=$W/models/adapter)
  ts adapter-ready
fi

ln -sfn $W/cache/flashinfer /root/.cache/flashinfer
export VLLM_MARLIN_USE_ATOMIC_ADD=1 VLLM_PREFIX_CACHE_RETENTION_INTERVAL=1648 VLLM_CACHE_ROOT=$W/cache/mtp-k7
nohup vllm serve $W/models/RedHatAI-Qwen3.8-27B-INT4 --served-model-name qwen3.8-27b --host 127.0.0.1 --port 18102 \
  --max-model-len 131072 --gpu-memory-utilization 0.95 --kv-cache-memory-bytes ${KV_BYTES:-17179869184} \
  --max-num-seqs 10 --max-num-batched-tokens 16384 --enable-prefix-caching --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":16,"video":0}' --mm-processor-kwargs '{"max_pixels":1048576}' --mm-processor-cache-gb 0 \
  --allowed-media-domains media.invalid \
  --kv-cache-dtype bfloat16 --dtype bfloat16 --no-enable-flashinfer-autotune --trust-remote-code --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml --enable-auto-tool-choice --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-prompt-tokens-details --enable-force-include-usage --fingerprint-mode custom --fingerprint-value "${FINGERPRINT_VALUE}" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":7}' \
  "${lora_args[@]}" \
  > $W/run/vllm.log 2>&1 &
VLLM_PID=$!
ts vllm-started
until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18102/health', timeout=3)" 2>/dev/null; do
  kill -0 $VLLM_PID 2>/dev/null || { ts vllm-exited; sleep infinity; }
  sleep 5
done
ts vllm-healthy

if [[ -n "${TUNNEL_TOKEN:-}" ]]; then
  python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$CLOUDFLARED_URL" $W/run/cloudflared
  echo "${CLOUDFLARED_SHA256}  $W/run/cloudflared" | sha256sum -c - || { ts cloudflared-hash-mismatch; sleep infinity; }
  chmod 700 $W/run/cloudflared
  ( umask 077; printf '%s' "$TUNNEL_TOKEN" > $W/run/tunnel-token )
  nohup $W/run/cloudflared tunnel --no-autoupdate --metrics 127.0.0.1:20241 run --token-file $W/run/tunnel-token > $W/run/cloudflared.log 2>&1 &
  ts tunnel-started
fi
sleep infinity
