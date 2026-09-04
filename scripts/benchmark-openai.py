#!/usr/bin/env python3
"""Run the same streaming decode qualification against OpenAI-compatible APIs.

The harness deliberately owns the prompt corpus on the client side so unlike
runtime-specific benchmark commands it can send identical requests to vLLM,
oMLX, and other servers. It stores metrics and finish reasons, never response
text or credentials.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

CODE_UNIT = """def merge_sorted(left, right):
    out = []
    while left and right:
        out.append(left.pop(0) if left[0] <= right[0] else right.pop(0))
    return out + left + right

"""


@dataclass(frozen=True)
class Case:
    name: str
    repeats: int
    concurrency: int
    requests: int


DEFAULT_CASES = (
    Case("pp1k_tg256_c1", repeats=19, concurrency=1, requests=1),
    Case("pp1k_tg256_c4", repeats=19, concurrency=4, requests=8),
    Case("pp16k_tg256_c1", repeats=307, concurrency=1, requests=1),
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def prompt_for(case: Case, request_index: int, warmup: bool = False) -> list[dict]:
    phase = "warmup" if warmup else "run"
    marker = f"qualification-{case.name}-{request_index}-{phase}"
    system = (
        f"Unique request marker: {marker}.\n"
        "Study this Python corpus, then follow the final instruction.\n"
        + CODE_UNIT
        * case.repeats
    )
    user = (
        "Continue writing a deterministic Python test-data module. Keep emitting "
        "code and integer test cases without a conclusion until the token limit."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def stream_request(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    case: Case,
    request_index: int,
    max_tokens: int,
    timeout: float,
    warmup: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": prompt_for(case, request_index, warmup),
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 101 + request_index,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib_request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    finish_reason: str | None = None
    with urllib_request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if first_token_at is None and any(
                    delta.get(key) for key in ("content", "reasoning_content")
                ):
                    first_token_at = time.perf_counter()
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]

    finished = time.perf_counter()
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if prompt_tokens is None or completion_tokens is None:
        raise RuntimeError("stream ended without prompt/completion usage")
    total_s = finished - started
    decode_s = finished - first_token_at if first_token_at is not None else None
    decode_tokens = max(0, int(completion_tokens) - 1)
    itl_ms = (decode_s / decode_tokens * 1000) if decode_s and decode_tokens else None
    return {
        "request_index": request_index,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cached_tokens": int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        ),
        "ttft_s": round((first_token_at or finished) - started, 4),
        "total_s": round(total_s, 4),
        "e2e_output_tok_s": round(completion_tokens / total_s, 4),
        "decode_tok_s": round(decode_tokens / decode_s, 4)
        if decode_s and decode_tokens
        else None,
        "itl_ms": round(itl_ms, 4) if itl_ms is not None else None,
        "finish_reason": finish_reason,
        "runtime_timings": timings or None,
    }


def run_case(args: argparse.Namespace, case: Case) -> dict[str, Any]:
    warmup_case = Case(
        f"{case.name}_warmup",
        repeats=case.repeats,
        concurrency=case.concurrency,
        requests=case.concurrency,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=warmup_case.concurrency
    ) as pool:
        warmups = [
            pool.submit(
                stream_request,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                case=warmup_case,
                request_index=index,
                max_tokens=8,
                timeout=args.timeout,
                warmup=True,
            )
            for index in range(warmup_case.requests)
        ]
        for future in warmups:
            future.result()

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=case.concurrency) as pool:
        futures = {
            pool.submit(
                stream_request,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                case=case,
                request_index=index,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            ): index
            for index in range(case.requests)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # report every request without hiding partial runs
                failures.append({"request_index": index, "error": str(exc)})
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: row["request_index"])
    completed = sum(row["completion_tokens"] for row in rows)
    total_times = [row["total_s"] for row in rows]
    ttfts = [row["ttft_s"] for row in rows]
    decode_rates = [
        row["decode_tok_s"] for row in rows if row["decode_tok_s"] is not None
    ]
    itls = [row["itl_ms"] for row in rows if row["itl_ms"] is not None]
    return {
        "case": asdict(case),
        "elapsed_s": round(elapsed, 4),
        "aggregate_output_tok_s": round(completed / elapsed, 4) if elapsed else None,
        "request_e2e_output_tok_s_mean": round(
            statistics.fmean(row["e2e_output_tok_s"] for row in rows), 4
        )
        if rows
        else None,
        "request_decode_tok_s_mean": round(statistics.fmean(decode_rates), 4)
        if decode_rates
        else None,
        "itl_ms_mean": round(statistics.fmean(itls), 4) if itls else None,
        "itl_ms_p95": round(percentile(itls, 0.95), 4) if itls else None,
        "ttft_s_mean": round(statistics.fmean(ttfts), 4) if ttfts else None,
        "ttft_s_p95": round(percentile(ttfts, 0.95), 4) if ttfts else None,
        "latency_s_mean": round(statistics.fmean(total_times), 4)
        if total_times
        else None,
        "latency_s_p95": round(percentile(total_times, 0.95), 4)
        if total_times
        else None,
        "requests": rows,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in DEFAULT_CASES],
        help="run only the named case; repeat to select multiple cases",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = (
        [case for case in DEFAULT_CASES if case.name in args.case]
        if args.case
        else DEFAULT_CASES
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "label": args.label,
        "model": args.model,
        "endpoint": "/v1/chat/completions",
        "protocol": {
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": True,
            "thinking": False,
            "response_text_stored": False,
            "unique_prefix_per_request": True,
        },
        "cases": [run_case(args, case) for case in cases],
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if not args.quiet:
        print(rendered, end="")


if __name__ == "__main__":
    main()
