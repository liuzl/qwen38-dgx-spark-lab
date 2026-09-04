#!/usr/bin/env python3
"""Run a bounded, response-checking stability smoke test."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from datetime import UTC, datetime
from typing import Any
from urllib import request as urllib_request


def one_request(
    base_url: str, model: str, request_index: int, timeout: float
) -> dict[str, Any]:
    marker = f"stability-ok-{request_index}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Reply exactly: {marker}"}],
        "max_tokens": 32,
        "temperature": 0,
        "seed": request_index,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib_request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib_request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    text = result["choices"][0]["message"].get("content") or ""
    usage = result.get("usage") or {}
    if text.strip() != marker:
        raise RuntimeError("exact response mismatch")
    return {
        "request_index": request_index,
        "elapsed_s": round(elapsed, 4),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        parser.error("requests and concurrency must be positive")

    rows = []
    failures = []
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                one_request,
                args.base_url,
                args.model,
                request_index,
                args.timeout,
            ): request_index
            for request_index in range(args.requests)
        }
        for future in concurrent.futures.as_completed(futures):
            request_index = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # preserve every failure without response text
                failures.append(
                    {"request_index": request_index, "error": type(exc).__name__}
                )
    elapsed = time.perf_counter() - started
    latencies = [row["elapsed_s"] for row in rows]
    result = {
        "schema_version": 1,
        "created_at": started_at.isoformat(),
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_s": round(elapsed, 4),
        "successes": len(rows),
        "failures": failures,
        "request_rate": round(len(rows) / elapsed, 4) if elapsed else None,
        "latency_s_mean": round(statistics.fmean(latencies), 4) if rows else None,
        "latency_s_max": round(max(latencies), 4) if rows else None,
        "usage_reported_for_all": all(
            row["prompt_tokens"] is not None and row["completion_tokens"] is not None
            for row in rows
        ),
    }
    print(json.dumps(result, indent=2))
    if failures or len(rows) != args.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
