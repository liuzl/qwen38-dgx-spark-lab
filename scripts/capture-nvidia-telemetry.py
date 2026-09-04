#!/usr/bin/env python3
"""Sample NVIDIA GPU telemetry and write a compact qualification artifact."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUERY_FIELDS = (
    "timestamp",
    "index",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "power.draw",
    "temperature.gpu",
)
NUMERIC_FIELDS = {
    "memory.used": "memory_used_mib",
    "memory.total": "memory_total_mib",
    "utilization.gpu": "gpu_utilization_percent",
    "power.draw": "power_w",
    "temperature.gpu": "temperature_c",
}


def sample() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(QUERY_FIELDS):
            raise RuntimeError(f"unexpected nvidia-smi row with {len(values)} fields")
        raw = dict(zip(QUERY_FIELDS, values, strict=True))
        row: dict[str, Any] = {
            "timestamp": raw["timestamp"],
            "index": int(raw["index"]),
        }
        for source, target in NUMERIC_FIELDS.items():
            row[target] = float(raw[source])
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key in NUMERIC_FIELDS.values():
        values = [row[key] for row in rows]
        result[key] = {
            "min": min(values),
            "mean": round(statistics.fmean(values), 3),
            "max": max(values),
        }
    return result


def container_restarts(container: str | None) -> int | None:
    if not container:
        return None
    completed = subprocess.run(
        ["docker", "inspect", container, "--format", "{{.RestartCount}}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return int(completed.stdout.strip()) if completed.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--container")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        parser.error("duration and interval must be positive")

    # This host-side sampler intentionally supports the observed Python 3.10.
    started_at = datetime.now(timezone.utc)  # noqa: UP017
    started = time.monotonic()
    restart_count_before = container_restarts(args.container)
    rows = []
    errors = []
    while time.monotonic() - started < args.duration:
        try:
            rows.extend(sample())
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(type(exc).__name__)
        remaining = args.duration - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(min(args.interval, remaining))

    if not rows:
        raise RuntimeError("no NVIDIA telemetry samples collected")
    result = {
        "schema_version": 1,
        "created_at": started_at.isoformat(),
        "label": args.label,
        "requested_duration_s": args.duration,
        "observed_duration_s": round(time.monotonic() - started, 3),
        "interval_s": args.interval,
        "sample_count": len(rows),
        "summary": summarize(rows),
        "container_restart_count_before": restart_count_before,
        "container_restart_count_after": container_restarts(args.container),
        "sample_errors": errors,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
