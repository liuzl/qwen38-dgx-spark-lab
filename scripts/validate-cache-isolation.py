#!/usr/bin/env python3
"""Prove same-LoRA prefix reuse and cross-LoRA cache isolation."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request


def metric(base_url: str) -> float:
    with urllib.request.urlopen(
        base_url.rstrip("/") + "/metrics", timeout=10
    ) as response:
        text = response.read().decode()
    match = re.search(
        r"^vllm:prefix_cache_hits_total\{[^\n]+\} ([0-9.e+]+)$", text, re.M
    )
    if match is None:
        raise RuntimeError("prefix_cache_hits_total metric not found")
    return float(match.group(1))


def call(base_url: str, model: str, prompt: str) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-model", required=True)
    args = parser.parse_args()

    prompt = ("native-lora-cache-isolation unique-prefix-block " * 400) + "\nReply ok."
    steps = [("initial", metric(args.base_url))]
    for label, model in (
        ("base_first", args.base_model),
        ("base_second", args.base_model),
        ("adapter_first", args.adapter_model),
        ("adapter_second", args.adapter_model),
    ):
        call(args.base_url, model, prompt)
        steps.append((label, metric(args.base_url)))

    deltas = [steps[index][1] - steps[index - 1][1] for index in range(1, len(steps))]
    result = {"steps": steps, "deltas": deltas}
    print(json.dumps(result, indent=2))

    if deltas[0] != 0 or deltas[2] != 0:
        raise RuntimeError("first request crossed a cache identity boundary")
    if deltas[1] <= 0 or deltas[3] <= 0:
        raise RuntimeError("same-model repeated prefix did not hit cache")


if __name__ == "__main__":
    main()
