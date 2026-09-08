#!/usr/bin/env python3
"""Summarise a soak-a100.sh JSONL log: rounds, failures per check, latency percentiles,
GPU memory drift, container restarts. Prints JSON; stores no response text."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.log.read_text().splitlines() if line.strip()
    ]
    checks: dict[str, dict[str, list]] = defaultdict(
        lambda: {"n": 0, "fail": 0, "lat": []}
    )
    bursts: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "short": 0})
    gpu = [r["gpu_mem_mib"] for r in rows if isinstance(r.get("gpu_mem_mib"), int)]
    restarts = [r.get("container", {}).get("restart_count") for r in rows]
    for r in rows:
        for alias, a in r["aliases"].items():
            for name, v in a.items():
                if name == "burst8":
                    bursts[alias]["n"] += 1
                    bursts[alias]["short"] += int(v["ok"] != 8)
                    continue
                c = checks[f"{alias}/{name}"]
                c["n"] += 1
                c["fail"] += int(not v["ok"])
                c["lat"].append(v["latency_s"])
    summary = {
        "log": str(args.log),
        "rounds": len(rows),
        "first": rows[0]["ts"] if rows else None,
        "last": rows[-1]["ts"] if rows else None,
        "rounds_all_ok": sum(1 for r in rows if r.get("all_ok")),
        "checks": {
            k: {
                "n": v["n"],
                "failures": v["fail"],
                "latency_p50_s": pct(v["lat"], 0.5),
                "latency_p95_s": pct(v["lat"], 0.95),
                "latency_max_s": max(v["lat"]) if v["lat"] else None,
            }
            for k, v in sorted(checks.items())
        },
        "burst8_rounds_short_of_8": {k: v for k, v in bursts.items()},
        "gpu_mem_mib": {
            "first": gpu[0] if gpu else None,
            "last": gpu[-1] if gpu else None,
            "min": min(gpu) if gpu else None,
            "max": max(gpu) if gpu else None,
            "median": statistics.median(gpu) if gpu else None,
        },
        "container_restart_count": {
            "first": restarts[0] if restarts else None,
            "last": restarts[-1] if restarts else None,
        },
        "preemptions_last": rows[-1].get("metrics", {}).get("num_preemptions_total")
        if rows
        else None,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
