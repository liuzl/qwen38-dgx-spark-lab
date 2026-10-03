#!/usr/bin/env bash
# Verify vLLM PR #53520 (prompt_logprobs corruption from CUDA-graph buffer reuse by the
# padded MTP drafter) on A100 + INT8 W8A8, and split V1 vs V2 model runner.
# Same probe/flags as runpod-repro-mtp-prompt-logprobs.sh (2026-09-29).
# ROLE=v028 (image v0.28.0): C unpatched, C patched, A patched, C unpatched + V2 runner.
# ROLE=v029 (image v0.29.0): C default runner, C forced V1.
# Each config appends one JSON line (with detected runner) to /workspace/results.jsonl.
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


PR_COMMIT=7af173ae2dc4da50541b414d9b89037d48343d20
SITE=$(python3 -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))")
RUNNER_PY=$SITE/vllm/v1/worker/gpu_model_runner.py
cp "$RUNNER_PY" $W/run/gpu_model_runner.py.orig
ts "vllm $(python3 -c 'import vllm;print(vllm.__version__)') site=$SITE"
apply_patch() {
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq patch >/dev/null 2>&1
  curl -fsSL https://github.com/vllm-project/vllm/commit/$PR_COMMIT.diff -o $W/run/pr.diff || { ts "diff download failed"; return 1; }
  python3 - $W/run/pr.diff $W/run/pr-vllm.diff <<'PY2'
import sys,re
s=open(sys.argv[1]).read()
parts=re.split(r'(?=^diff --git )', s, flags=re.M)
open(sys.argv[2],'w').write(''.join(p for p in parts if p.startswith('diff --git a/vllm/')))
PY2
  patch -p1 -d "$SITE" < $W/run/pr-vllm.diff && grep -q _get_bookkeeping_hidden_states "$RUNNER_PY" && ts "patch applied ($PR_COMMIT)"
}
unpatch() { cp $W/run/gpu_model_runner.py.orig "$RUNNER_PY"; find "$SITE/vllm/v1/worker" -name "gpu_model_runner*.pyc" -delete; ts "patch reverted"; }

run_config() {
  local name=$1; shift
  ts "config $name start"
  env "${EXTRA_ENV[@]}" vllm serve "$@" > $W/run/vllm-$name.log 2>&1 < /dev/null &
  local pid=$!
  until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18103/health', timeout=3)" 2>/dev/null; do
    kill -0 $pid 2>/dev/null || { ts "config $name exited"; echo "{\"config\": \"$name\", \"error\": \"server exited\"}" >> $W/results.jsonl; return; }
    sleep 5
  done
  ts "config $name healthy"
  local runner=V1
  grep -q "Using V2 Model Runner" $W/run/vllm-$name.log && runner=V2
  python3 $W/run/probe.py "$name" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); d['runner']=sys.argv[1]; print(json.dumps(d))" "$runner" | tee -a $W/results.jsonl
  kill $pid; wait $pid 2>/dev/null
  sleep 10
  ts "config $name done"
}
EXTRA_ENV=(VLLM_DUMMY=1)
case "${ROLE:-v028}" in
  v028)
    run_config C-unpatched "${BASE_ARGS[@]}" "${CG_DEFAULT[@]}" "${MTP[@]}"
    if apply_patch; then
      run_config C-pr53520 "${BASE_ARGS[@]}" "${CG_DEFAULT[@]}" "${MTP[@]}"
      run_config A-pr53520 "${BASE_ARGS[@]}" "${CG_DEFAULT[@]}" "${MTP[@]}" "${LORA[@]}"
    else
      echo '{"config": "patch", "error": "patch failed"}' >> $W/results.jsonl
    fi
    unpatch
    EXTRA_ENV=(VLLM_USE_V2_MODEL_RUNNER=1)
    run_config C-unpatched-V2 "${BASE_ARGS[@]}" "${CG_DEFAULT[@]}" "${MTP[@]}"
    ;;
  v029)
    run_config C-default "${BASE_ARGS[@]}" "${CG_DEFAULT[@]}" "${MTP[@]}"
    EXTRA_ENV=(VLLM_USE_V2_MODEL_RUNNER=0)
    run_config C-forced-V1 "${BASE_ARGS[@]}" "${CG_DEFAULT[@]}" "${MTP[@]}"
    ;;
esac
ts all-done
sleep infinity
