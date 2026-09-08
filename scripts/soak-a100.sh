#!/usr/bin/env bash
# 24-hour soak of the A100 production service. Every INTERVAL seconds, for each
# alias: one short greedy chat, one forced tool call (argument must be
# Singapore), and every 6th round one image OCR (digits 7429) plus a burst of 8
# concurrent short requests. Records status, latency, server metrics, GPU
# memory and container restart count as JSONL. Never stores response text
# beyond a pass/fail flag. Summarise with soak-a100-summary.py.
#
# usage: soak-a100.sh <label> [duration_seconds=86400] [interval_seconds=300]
set -uo pipefail
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.43}"
label="${1:?label}"; duration="${2:-86400}"; interval="${3:-300}"
BASE="${A100_BASE:-/databank/zliu/qwen38-a100}"
PROD="${PROD_CONTAINER:-qwen38-a100-native-lora}"
URL="${URL:-http://127.0.0.1:18103}"
IMAGE="${IMAGE_FIXTURE:-$BASE/logs/perf-image.png}"
OUT="$BASE/logs/soak-$label.jsonl"
ALIASES="${ALIASES:-qwen3.8-27b qwen3.8-27b-uncensored}"
start=$(date +%s); round=0
echo "[$(date -u +%FT%TZ)] soak $label start: ${duration}s, every ${interval}s, out=$OUT"
while (( $(date +%s) - start < duration )); do
  round=$((round + 1))
  python3 - "$URL" "$IMAGE" "$OUT" "$round" "$PROD" $ALIASES <<'PY'
import base64, json, subprocess, sys, time, urllib.request, urllib.error, concurrent.futures as cf
url, image, out, rnd, prod, *aliases = sys.argv[1:]
rnd = int(rnd)
def post(path, body, timeout=180):
    t0 = time.time()
    rq = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return r.status, json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, None, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return 0, {"error": type(e).__name__}, time.time() - t0
schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "round": rnd, "aliases": {}}
for alias in aliases:
    a = {}
    st, r, dt = post("/v1/chat/completions", {"model": alias, "max_tokens": 64, "temperature": 0,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}]})
    txt = (r or {}).get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else ""
    a["chat"] = {"status": st, "latency_s": round(dt, 3), "ok": st == 200 and "OK" in txt}
    st, r, dt = post("/v1/chat/completions", {"model": alias, "max_tokens": 64, "temperature": 0,
        "messages": [{"role": "user", "content": "Use the weather tool for Singapore."}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather lookup", "parameters": schema}}],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
    args = ""
    if st == 200:
        tc = r["choices"][0]["message"].get("tool_calls") or []
        args = tc[0]["function"]["arguments"] if tc else ""
    a["tool"] = {"status": st, "latency_s": round(dt, 3), "ok": st == 200 and "Singapore" in args}
    if rnd % 6 == 1:
        b64 = base64.b64encode(open(image, "rb").read()).decode()
        st, r, dt = post("/v1/chat/completions", {"model": alias, "max_tokens": 16, "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Read the four digits in this image. Reply with the digits only."},
                                                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]})
        txt = (r or {}).get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else ""
        a["image"] = {"status": st, "latency_s": round(dt, 3), "ok": st == 200 and "7429" in txt}
        def burst(i):
            return post("/v1/chat/completions", {"model": alias, "max_tokens": 32, "temperature": 0,
                "messages": [{"role": "user", "content": f"soak-{rnd}-{i}: name one prime number greater than {i + 10}."}]})
        t0 = time.time()
        with cf.ThreadPoolExecutor(8) as ex:
            res = list(ex.map(burst, range(8)))
        a["burst8"] = {"ok": sum(1 for s, _, _ in res if s == 200), "wall_s": round(time.time() - t0, 3)}
    rec["aliases"][alias] = a
try:
    m = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    def metric(name):
        for line in m.splitlines():
            if line.startswith("vllm:" + name + "{"):
                return float(line.split()[-1])
        return None
    rec["metrics"] = {k: metric(k) for k in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "num_preemptions_total")}
except Exception as e:  # noqa: BLE001
    rec["metrics"] = {"error": type(e).__name__}
try:
    rec["gpu_mem_mib"] = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()[0])
    insp = json.loads(subprocess.check_output(["docker", "inspect", prod]).decode())[0]
    rec["container"] = {"restart_count": insp["RestartCount"], "started_at": insp["State"]["StartedAt"][:19], "health": insp["State"].get("Health", {}).get("Status")}
except Exception as e:  # noqa: BLE001
    rec["container"] = {"error": type(e).__name__}
rec["all_ok"] = all(v.get("ok", True) if isinstance(v, dict) and "ok" in v and k != "burst8" else True
                    for a in rec["aliases"].values() for k, v in a.items()) and all(
                    a.get("burst8", {"ok": 8})["ok"] == 8 for a in rec["aliases"].values())
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"[{rec['ts']}] round {rnd} all_ok={rec['all_ok']} gpu={rec.get('gpu_mem_mib')} restarts={rec.get('container', {}).get('restart_count')}", flush=True)
PY
  sleep "$interval"
done
echo "[$(date -u +%FT%TZ)] SOAK DONE $label rounds=$round"
