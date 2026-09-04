#!/usr/bin/env python3
"""Summarize two benchmark-openai.py result files without raw response data."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def index_cases(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["case"]["name"]: case for case in result["cases"]}


def metrics(case: dict[str, Any]) -> dict[str, Any]:
    decode_rate = case["request_decode_tok_s_mean"]
    itl_ms = case.get("itl_ms_mean")
    if itl_ms is None and decode_rate:
        itl_ms = round(1000 / decode_rate, 4)
    return {
        "prompt_tokens": case["requests"][0]["prompt_tokens"],
        "aggregate_output_tok_s": case["aggregate_output_tok_s"],
        "request_decode_tok_s_mean": decode_rate,
        "itl_ms_mean": itl_ms,
        "ttft_s_mean": case["ttft_s_mean"],
        "latency_s_mean": case["latency_s_mean"],
        "failures": len(case["failures"]),
    }


def percent(numerator: float, denominator: float) -> float:
    return round((numerator / denominator - 1) * 100, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = load(args.reference)
    candidate = load(args.candidate)
    reference_cases = index_cases(reference)
    candidate_cases = index_cases(candidate)
    if reference_cases.keys() != candidate_cases.keys():
        raise RuntimeError("benchmark files contain different case sets")

    comparisons = []
    for name in reference_cases:
        ref = metrics(reference_cases[name])
        cand = metrics(candidate_cases[name])
        if ref["prompt_tokens"] != cand["prompt_tokens"]:
            raise RuntimeError(f"{name}: prompt token counts differ")
        comparisons.append(
            {
                "case": name,
                "reference": ref,
                "candidate": cand,
                "candidate_vs_reference_percent": {
                    "aggregate_output_tok_s": percent(
                        cand["aggregate_output_tok_s"],
                        ref["aggregate_output_tok_s"],
                    ),
                    "request_decode_tok_s_mean": percent(
                        cand["request_decode_tok_s_mean"],
                        ref["request_decode_tok_s_mean"],
                    ),
                    "itl_ms_mean": percent(cand["itl_ms_mean"], ref["itl_ms_mean"]),
                    "ttft_s_mean": percent(cand["ttft_s_mean"], ref["ttft_s_mean"]),
                    "latency_s_mean": percent(
                        cand["latency_s_mean"], ref["latency_s_mean"]
                    ),
                },
            }
        )

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "reference_label": reference["label"],
        "candidate_label": candidate["label"],
        "protocol": reference["protocol"],
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
