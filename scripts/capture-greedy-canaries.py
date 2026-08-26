#!/usr/bin/env python3
"""Capture deterministic responses for runtime equivalence comparisons.

The output contains response text and therefore belongs under benchmarks/raw,
which is intentionally gitignored. Compare two captures before publishing only
hashes and mismatch locations in a qualification result.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

CANARIES = (
    (
        "code",
        "Write a Python function that merges two sorted integer lists. Output "
        "only code and include three assert statements.",
    ),
    (
        "chinese_reasoning",
        "一个水箱先装入总容量的三分之一，又装入24升后达到总容量的五分之三。"
        "求水箱容量，并简要写出计算过程。",
    ),
    (
        "multilingual",
        "Translate 'The cache is valid only for an identical prefix.' into "
        "Chinese and Japanese, one language per line.",
    ),
    (
        "structured",
        "Return one compact JSON object with keys name, primes, and valid. Set "
        "name to canary, primes to the first five prime numbers, and valid to true.",
    ),
)


def post(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib_request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib_request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    rows = []
    for name, prompt in CANARIES:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        response = post(args.base_url, payload, args.timeout)
        choice = response["choices"][0]
        rows.append(
            {
                "name": name,
                "content": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage") or {},
                "elapsed_s": round(time.perf_counter() - started, 4),
            }
        )

    result = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "label": args.label,
        "model": args.model,
        "protocol": {
            "temperature": 0,
            "seed": 42,
            "max_tokens": args.max_tokens,
            "enable_thinking": False,
        },
        "canaries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
