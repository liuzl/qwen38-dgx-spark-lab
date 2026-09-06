#!/usr/bin/env python3
"""Repeated text/image matrix using the shared streaming benchmark counters."""

import argparse
import base64
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path


def counters(base_url):
    with urllib.request.urlopen(base_url + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    names = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    return {
        name: sum(
            float(line.split()[-1])
            for line in lines
            if line.startswith("vllm:spec_decode_" + name + "_total{")
        )
        for name in names
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "shared_bench", Path(__file__).with_name("benchmark-openai.py")
    )
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    original = bench.prompt_for
    data = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()
    rows = []
    for model in ("qwen3.8-27b", "qwen3.8-27b-uncensored"):
        for repetition in range(args.repetitions):
            for name, count, concurrency, image in (
                ("text_1k_c1", 19, 1, False),
                ("text_1k_c8", 19, 8, False),
                ("text_16k_c1", 307, 1, False),
                ("image_1k_c1", 19, 1, True),
            ):

                def prompt(
                    case,
                    request_index,
                    warmup=False,
                    model=model,
                    repetition=repetition,
                    name=name,
                    image=image,
                ):
                    messages = original(case, request_index, warmup)
                    # Identical corpus across arms, distinct prefixes per repetition.
                    # Start each arm in a fresh process to avoid cross-arm cache reuse.
                    messages[0]["content"] = (
                        f"matrix-v1:{model}:{repetition}:{name}\n"
                        + messages[0]["content"]
                    )
                    if image:
                        messages[1]["content"] = [
                            {"type": "text", "text": messages[1]["content"]},
                            {"type": "image_url", "image_url": {"url": data}},
                        ]
                    return messages

                bench.prompt_for = prompt
                options = argparse.Namespace(
                    base_url=args.base_url,
                    api_key=None,
                    model=model,
                    max_tokens=256,
                    timeout=180,
                )
                before = counters(args.base_url)
                result = bench.run_case(
                    options, bench.Case(name, count, concurrency, concurrency)
                )
                after = counters(args.base_url)
                result["speculative_counter_delta_including_warmup"] = {
                    k: after[k] - before[k] for k in before
                }
                if result["failures"]:
                    raise RuntimeError("benchmark requests failed")
                if any(r["completion_tokens"] != 256 for r in result["requests"]):
                    raise RuntimeError("output length mismatch")
                result.update(model=model, repetition=repetition)
                rows.append(result)
                args.output.write_text(
                    json.dumps({"label": args.label, "rows": rows}, indent=2)
                )
                print(
                    json.dumps(
                        {
                            k: result[k]
                            for k in (
                                "model",
                                "repetition",
                                "case",
                                "aggregate_output_tok_s",
                                "request_decode_tok_s_mean",
                                "ttft_s_mean",
                            )
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
