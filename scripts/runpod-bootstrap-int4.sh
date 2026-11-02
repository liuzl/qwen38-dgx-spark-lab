#!/usr/bin/env bash
# RunPod autostart bootstrap for Qwen3.8-27B RedHat INT4 (W4A16) + MTP on vLLM v0.29.0,
# for Ampere 48 GB cards (RTX A6000, A40) that cannot run the NVFP4 checkpoint.
#
# Used as the template env BOOTSTRAP with image vllm/vllm-openai:v0.29.0,
# entrypoint `bash -c` and start command `eval "$BOOTSTRAP"`. Required env:
#   FINGERPRINT_VALUE  deployment ID returned as system_fingerprint
# Optional: KV_BYTES (default 16 GiB; bf16 KV, since Ampere has no FP8 KV).
# Optional adapter (serves qwen3.8-27b-uncensored next to the base alias):
#   ARTIFACT_URL       URL of the adapter tar (authenticated artifact endpoint)
#   ARTIFACT_TOKEN     bearer token supplied through a deployment secret
#   ADAPTER_TAR_SHA256 / ADAPTER_SHA256  expected hashes (tar, adapter_model.safetensors)
# The adapter for this checkpoint is the rank-1 adapter with lora_B scaled by 0.85
# (adapter_model.safetensors sha256 7dc18d34...), not the NVFP4 one: at 1.0 it costs
# 3.7 Global-MMLU-Lite points on INT4, at 0.85 1.2 points with refusals still removed.
# Optional second adapter (serves qwen3.8-27b-cute; needs the first adapter's ARTIFACT_TOKEN):
#   CUTE_ARTIFACT_URL  URL of the tar holding adapter-cute/ (same artifact endpoint and token)
#   CUTE_ADAPTER_TAR_SHA256 / CUTE_ADAPTER_SHA256  expected hashes (tar, adapter_model.safetensors)
# It is an exact cat merge (rank 17) of wesjos/Qwen3.8-27B-CuteHuman-LoRA@edd19281 with the 0.85 adapter
# above (sha256 f2119d8f...); serving it raises --max-loras to 2 and --max-lora-rank to 32.
# Without ARTIFACT_TOKEN only the base alias is served. Listens on 127.0.0.1:18102.
# Optional data-plane tunnel, started once vLLM is healthy:
#   TUNNEL_TOKEN       Cloudflare Tunnel token supplied through a deployment secret
#   CLOUDFLARED_URL / CLOUDFLARED_SHA256  pinned static cloudflared binary and its hash
# Optional compile cache: download a published cache for this exact runtime and GPU setup.
#   COMPILE_CACHE_BASE  authenticated artifact URL prefix (uses ARTIFACT_TOKEN)
# The key hashes the vLLM/torch versions, GPU architecture and serve arguments.
# It is written to $RUNTIME_ROOT/cache/compile-cache-key. Nodes only download caches
# and verify their SHA-256; publishing is a separate operator action.
# DATA_ROOT defaults to /workspace; RUNTIME_ROOT defaults to /var/log/model-serving.
# Logs and runtime caches use local disk; use DATA_ROOT=/opt/model-data for volume-free nodes.
#
set -u
export DATA_ROOT=${DATA_ROOT:-/workspace}
W=$DATA_ROOT
export RUNTIME_ROOT=${RUNTIME_ROOT:-/var/log/model-serving}
R=$RUNTIME_ROOT
mkdir -p "$R/run" "$R/cache/flashinfer"
exec > >(tee -a "$R/bootstrap.log") 2>&1
ts() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$R/timeline.log"; }
ts bootstrap-start

# sshd in the background so weights start downloading immediately.
(
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server >/dev/null
  mkdir -p /root/.ssh /run/sshd && chmod 700 /root/.ssh
  printf '%s\n' "${PUBLIC_KEY:-}" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
  /usr/sbin/sshd && ts sshd-up
) &

mkdir -p "$W/models"
# The serve arguments, decided by the env alone, so the compile cache key is known before anything downloads.
lora_args=()
if [[ -n "${ARTIFACT_TOKEN:-}" ]]; then
  lora_args=(--enable-lora --max-loras 1 --max-lora-rank 1 --lora-dtype bfloat16
             --lora-modules qwen3.8-27b-uncensored=$W/models/adapter)
  [[ -n "${CUTE_ARTIFACT_URL:-}" ]] && lora_args=(--enable-lora --max-loras 2 --max-lora-rank 32 --lora-dtype bfloat16
             --lora-modules qwen3.8-27b-uncensored=$W/models/adapter qwen3.8-27b-cute=$W/models/adapter-cute)
fi
serve_args=($W/models/RedHatAI-Qwen3.8-27B-INT4 --served-model-name qwen3.8-27b --host 127.0.0.1 --port 18102
  --max-model-len 131072 --gpu-memory-utilization 0.95 --kv-cache-memory-bytes ${KV_BYTES:-17179869184}
  --max-num-seqs 10 --max-num-batched-tokens 16384 --enable-prefix-caching --enable-chunked-prefill
  --limit-mm-per-prompt '{"image":16,"video":0}' --mm-processor-kwargs '{"max_pixels":1048576}' --mm-processor-cache-gb 0
  --allowed-media-domains media.invalid
  --kv-cache-dtype bfloat16 --dtype bfloat16 --no-enable-flashinfer-autotune --trust-remote-code --reasoning-parser qwen3
  --tool-call-parser qwen3_xml --enable-auto-tool-choice --default-chat-template-kwargs '{"enable_thinking":false}'
  --enable-prompt-tokens-details --enable-force-include-usage
  --speculative-config '{"method":"mtp","num_speculative_tokens":7}'
  "${lora_args[@]}")
export VLLM_MARLIN_USE_ATOMIC_ADD=1 VLLM_PREFIX_CACHE_RETENTION_INTERVAL=1648 VLLM_CACHE_ROOT=$R/cache/mtp-k7
# Keep Triton cache local as well; optionally seed it from published artifacts.
export TRITON_CACHE_DIR=$R/cache/triton
mkdir -p $TRITON_CACHE_DIR
CC_KEY=$(printf '%s\n' "$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"), m.version("torch"))')" \
  "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" "${serve_args[@]}" | sha256sum | cut -c1-16)
echo "$CC_KEY" > $R/cache/compile-cache-key
ts "compile-cache-key $CC_KEY"
cc_fetch=""
if [[ -n "${COMPILE_CACHE_BASE:-}" && -n "${ARTIFACT_TOKEN:-}" && ! -d $VLLM_CACHE_ROOT/torch_compile_cache ]]; then
  ( CC_URL="$COMPILE_CACHE_BASE/$CC_KEY.tar.gz" python3 - <<'PY' && ts compile-cache-seeded || ts compile-cache-miss
import hashlib, os, tarfile, urllib.error, urllib.request
h = {"Authorization": "Bearer " + os.environ["ARTIFACT_TOKEN"], "User-Agent": "model-bootstrap/1"}
url = os.environ["CC_URL"]
try:
    want = urllib.request.urlopen(urllib.request.Request(url + ".sha256", headers=h), timeout=30).read().decode().split()[0]
except urllib.error.HTTPError as e:
    raise SystemExit(f"no compile cache published for this setup ({e.code})")
digest, path = hashlib.sha256(), os.path.join(os.environ["RUNTIME_ROOT"], "compile-cache.tar.gz")
with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600) as r, open(path, "wb") as f:
    for block in iter(lambda: r.read(1 << 20), b""):
        digest.update(block)
        f.write(block)
if digest.hexdigest() != want:
    os.remove(path)
    raise SystemExit("compile cache hash mismatch")
os.makedirs(os.environ["VLLM_CACHE_ROOT"], exist_ok=True)
with tarfile.open(path) as t:
    t.extractall(os.environ["VLLM_CACHE_ROOT"], filter="data")
os.remove(path)
print("compile cache ready", want[:12], flush=True)
PY
  ) &
  cc_fetch=$!
fi

python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo, rev, d in [("RedHatAI/Qwen3.8-27B-INT4", "91bd022d5b49442a868bc35008f6c21e1860edfa", "RedHatAI-Qwen3.8-27B-INT4")]:
    snapshot_download(repo, revision=rev, local_dir=os.path.join(os.environ["DATA_ROOT"], "models", d), max_workers=16)
    print("downloaded", repo, rev, flush=True)
PY
ts weights-ready

if [[ -n "${ARTIFACT_TOKEN:-}" ]]; then
  fetch_adapter() {  # fetch_adapter <url> <tar sha256> <weights sha256> <dir under models/>
  ARTIFACT_URL=$1 ADAPTER_TAR_SHA256=$2 ADAPTER_SHA256=$3 ADAPTER_DIR=$4 python3 - <<'PY' || { ts adapter-failed; sleep infinity; }
import hashlib, os, tarfile, urllib.request
req = urllib.request.Request(os.environ["ARTIFACT_URL"], headers={"Authorization": "Bearer " + os.environ["ARTIFACT_TOKEN"], "User-Agent": "model-bootstrap/1"})
data = urllib.request.urlopen(req, timeout=300).read()
assert hashlib.sha256(data).hexdigest() == os.environ["ADAPTER_TAR_SHA256"], "adapter tar hash mismatch"
open(os.path.join(os.environ["DATA_ROOT"], "adapter.tar"), "wb").write(data)
with tarfile.open(os.path.join(os.environ["DATA_ROOT"], "adapter.tar")) as t:
    t.extractall(os.path.join(os.environ["DATA_ROOT"], "models"), filter="data")
got = hashlib.sha256(open(os.path.join(os.environ["DATA_ROOT"], "models", os.environ["ADAPTER_DIR"], "adapter_model.safetensors"), "rb").read()).hexdigest()
assert got == os.environ["ADAPTER_SHA256"], "adapter weights hash mismatch"
os.remove(os.path.join(os.environ["DATA_ROOT"], "adapter.tar"))
print("adapter ready", os.environ["ADAPTER_DIR"], got[:12], flush=True)
PY
  }
  fetch_adapter "$ARTIFACT_URL" "$ADAPTER_TAR_SHA256" "$ADAPTER_SHA256" adapter
  if [[ -n "${CUTE_ARTIFACT_URL:-}" ]]; then
    fetch_adapter "$CUTE_ARTIFACT_URL" "$CUTE_ADAPTER_TAR_SHA256" "$CUTE_ADAPTER_SHA256" adapter-cute
  fi
  ts adapter-ready
fi

ln -sfn $R/cache/flashinfer /root/.cache/flashinfer
[[ -n "$cc_fetch" ]] && wait "$cc_fetch"
nohup vllm serve "${serve_args[@]}" --fingerprint-mode custom --fingerprint-value "${FINGERPRINT_VALUE}" \
  > $R/run/vllm.log 2>&1 &
VLLM_PID=$!
ts vllm-started
until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18102/health', timeout=3)" 2>/dev/null; do
  kill -0 $VLLM_PID 2>/dev/null || { ts vllm-exited; sleep infinity; }
  sleep 5
done
ts vllm-healthy

if [[ -n "${TUNNEL_TOKEN:-}" ]]; then
  python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$CLOUDFLARED_URL" $R/run/cloudflared
  echo "${CLOUDFLARED_SHA256}  $R/run/cloudflared" | sha256sum -c - || { ts cloudflared-hash-mismatch; sleep infinity; }
  chmod 700 $R/run/cloudflared
  ( umask 077; printf '%s' "$TUNNEL_TOKEN" > $R/run/tunnel-token )
  nohup $R/run/cloudflared tunnel --no-autoupdate --metrics 127.0.0.1:20241 run --token-file $R/run/tunnel-token > $R/run/cloudflared.log 2>&1 &
  ts tunnel-started
fi
sleep infinity
