#!/usr/bin/env bash
# Reproduce: vLLM v0.28.0 returns wrong prompt_logprobs (garbage top-1, ~-3.6e37,
# NaN -> HTTP 400 for 65-71 token prompts) when MTP speculative decoding runs with
# CUDA graphs, on the A100 service INT8 W8A8 checkpoint. Generation is unaffected.
#
# RunPod template env BOOTSTRAP; image vllm/vllm-openai:v0.28.0-cu129-ubuntu2404,
# entrypoint `bash -c`, start command `eval "$BOOTSTRAP"`, A100 80GB, 80 GB volume.
# Runs six configs in sequence and appends one JSON line per config to
# /workspace/results.jsonl. 2026-09-29 on A100 SXM: A, C, E wrong; B (no MTP),
# D (--enforce-eager) and F (cudagraph_mode NONE) correct.
set -u
W=/workspace
mkdir -p $W/run $W/hf
exec > >(tee -a $W/bootstrap.log) 2>&1
ts() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $W/timeline.log; }
ts bootstrap-start
(
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server >/dev/null
  mkdir -p /root/.ssh /run/sshd && chmod 700 /root/.ssh
  printf '%s\n' "${PUBLIC_KEY:-}" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
  /usr/sbin/sshd && ts sshd-up
) &

export HF_HOME=$W/hf
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8', revision='2df4e3b00d4b865d59a7de0dc286fb18fd455a1e', max_workers=16)"
ts weights-ready
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$W/vllm-cache

cat > $W/run/probe.py <<'PY'
import json, math, sys, urllib.request, urllib.error
B = "http://127.0.0.1:18103"
def post(body):
    req = urllib.request.Request(B + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        return {"error": e.read()[:200].decode()}
text = "The quick brown fox jumps over the lazy dog. The capital of France is Paris, and the capital of Japan is Tokyo."
r = post({"model": "qwen3.8-27b", "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
if "error" in r:
    ranks, minlp = None, None
else:
    ents = [list(e.values())[0] for e in r["choices"][0]["prompt_logprobs"][1:]]
    ranks = [e["rank"] for e in ents]
    minlp = min(e["logprob"] for e in ents)
ids = list(range(1000, 1066))  # 66 arbitrary token ids
n66 = post({"model": "qwen3.8-27b", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
g = post({"model": "qwen3.8-27b", "prompt": text[:70], "max_tokens": 1, "temperature": 0, "logprobs": 3})
print(json.dumps({
    "config": sys.argv[1],
    "prompt_ranks": ranks,
    "prompt_rank1_share": None if ranks is None else round(sum(x == 1 for x in ranks) / len(ranks), 3),
    "prompt_min_logprob": minlp,
    "len66": "error: " + n66["error"][:90] if "error" in n66 else "ok",
    "gen_top": None if "error" in g else g["choices"][0]["logprobs"]["top_logprobs"][0],
}))
PY

BASE_ARGS=(Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8 --revision 2df4e3b00d4b865d59a7de0dc286fb18fd455a1e
  --served-model-name qwen3.8-27b --host 127.0.0.1 --port 18103 --dtype bfloat16 --kv-cache-dtype auto --linear-backend auto
  --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 32 --max-num-batched-tokens 16384
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --mm-encoder-tp-mode data
  --default-chat-template-kwargs '{"enable_thinking":false}' --limit-mm-per-prompt '{"image":4,"video":0}'
  --mm-processor-cache-gb 0 --mm-processor-kwargs '{"max_pixels":1048576}' --allowed-media-domains media.invalid
  --enable-prefix-caching --enable-chunked-prefill --enable-prompt-tokens-details
  --kv-cache-memory-bytes 25769803776
  --enable-force-include-usage --enable-per-request-metrics)
CG_DEFAULT=(--compilation-config '{"max_cudagraph_capture_size": 256}')
CG_PIECEWISE=(--compilation-config '{"max_cudagraph_capture_size": 256, "cudagraph_mode": "PIECEWISE"}')
CG_NONE=(--compilation-config '{"max_cudagraph_capture_size": 256, "cudagraph_mode": "NONE"}')
MTP=(--speculative-config '{"method":"mtp","num_speculative_tokens":7}')
LORA=(--enable-lora --max-loras 1 --max-lora-rank 1 --lora-dtype bfloat16)

run_config() {
  local name=$1; shift
  ts "config $name start"
  vllm serve "$@" > $W/run/vllm-$name.log 2>&1 < /dev/null &
  local pid=$!
  until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18103/health', timeout=3)" 2>/dev/null; do
    kill -0 $pid 2>/dev/null || { ts "config $name exited"; return; }
    sleep 5
  done
  ts "config $name healthy"
  python3 $W/run/probe.py "$name" | tee -a $W/results.jsonl
  kill $pid; wait $pid 2>/dev/null
  sleep 10
  ts "config $name done"
}

run_config A-a100-service "${BASE_ARGS[@]}" "${CG_DEFAULT[@]}" "${MTP[@]}" "${LORA[@]}"
run_config B-no-mtp "${BASE_ARGS[@]}" "${CG_DEFAULT[@]}" "${LORA[@]}"
run_config C-no-lora "${BASE_ARGS[@]}" "${CG_DEFAULT[@]}" "${MTP[@]}"
run_config D-eager "${BASE_ARGS[@]}" "${CG_DEFAULT[@]}" "${MTP[@]}" "${LORA[@]}" --enforce-eager
run_config E-piecewise "${BASE_ARGS[@]}" "${CG_PIECEWISE[@]}" "${MTP[@]}" "${LORA[@]}"
run_config F-no-cudagraph "${BASE_ARGS[@]}" "${CG_NONE[@]}" "${MTP[@]}" "${LORA[@]}"
ts all-done
sleep infinity
