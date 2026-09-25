#!/usr/bin/env python3
"""Send one unique ~11K-token prompt twice and report prefix-cache hits.

Complements validate-cache-isolation.py: that proves identity boundaries with a
fixed prompt; this checks that a fresh long prompt is actually reused, which
speculative decoding on hybrid GDN models can silently break (vLLM #54360).

Exits non-zero when the repeat hits fewer than --min-ratio of the prompt.
Only token counts and timings are printed; no response text is kept.
"""

import argparse
import json
import sys
import time
import urllib.request


def counters(base: str) -> tuple[float, float]:
    text = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()

    def total(name: str) -> float:
        return sum(float(line.split()[-1]) for line in text.splitlines() if line.startswith(name + "{"))

    return total("vllm:prefix_cache_queries_total"), total("vllm:prefix_cache_hits_total")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--min-ratio", type=float, default=0.5)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    prompt = f"apc-probe-{time.time_ns()} " + " ".join(
        f"line {i}: the quick brown fox jumps over the lazy dog." for i in range(700)
    )
    body = json.dumps(
        {"model": args.model, "messages": [{"role": "user", "content": prompt + "\nReply OK."}], "max_tokens": 4}
    ).encode()
    rows = []
    for attempt in (1, 2):
        q0, h0 = counters(base)
        start = time.time()
        request = urllib.request.Request(base + "/v1/chat/completions", body, {"content-type": "application/json"})
        response = json.load(urllib.request.urlopen(request, timeout=300))
        q1, h1 = counters(base)
        rows.append(
            {
                "attempt": attempt,
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "seconds": round(time.time() - start, 3),
                "queries": q1 - q0,
                "hits": h1 - h0,
            }
        )
    ratio = rows[1]["hits"] / rows[1]["prompt_tokens"]
    result = {"model": args.model, "rows": rows, "repeat_hit_ratio": round(ratio, 4), "passed": ratio >= args.min_ratio}
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
