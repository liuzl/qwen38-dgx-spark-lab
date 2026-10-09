#!/usr/bin/env python3
"""External watchdog. State is a private, pre-verified resource-ID whitelist.
Creation controller MUST verify platform terminateAfter before running workloads.
Stopping compute is not disk cleanup; platform termination is the final fallback.
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def command(action, pod):
    try:
        result = subprocess.run(
            ["runpodctl", "pod", action, pod],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.returncode, result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return -1, ""


def validate(cfg):
    if not cfg["pod_id"] or not cfg["pod_name"].startswith("strata-pilot-"):
        raise ValueError("Invalid experiment resource identity")
    for key in ("created_epoch", "hourly_rate", "platform_terminate_epoch"):
        if not math.isfinite(float(cfg[key])):
            raise ValueError("Invalid budget state")
    if not 0 < cfg["hourly_rate"] <= 2:
        raise ValueError("Unapproved hourly rate")
    if not cfg.get("platform_termination_verified") or not (
        cfg["created_epoch"]
        < cfg["platform_terminate_epoch"]
        <= cfg["created_epoch"] + 10800
    ):
        raise ValueError("Verified platform termination within 3h is required")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.state).read_text())
    validate(cfg)
    pod, name = cfg["pod_id"], cfg["pod_name"]
    # Freeze identity/deadlines: later state-file changes must not extend spending.
    stop_at = cfg["created_epoch"] + min(cfg.get("stop_seconds", 10500), 10500)
    monotonic_stop = time.monotonic() + max(0, stop_at - time.time())
    rate = cfg["hourly_rate"]
    failures = 0
    stopping = False
    while True:
        age = max(0, time.time() - cfg["created_epoch"])
        bound = age / 3600 * rate + 1
        stopping |= (
            time.time() >= stop_at
            or time.monotonic() >= monotonic_stop
            or bound >= 7
            or rate > 2
        )
        # Deadline is evaluated before any query; a failed GET must not disable STOP.
        if stopping:
            code, _ = command("stop", pod)
            emit("budget_stop_attempt", pod_id=pod, exit=code)
        code, raw = command("get", pod)
        try:
            info = json.loads(raw) if code == 0 else None
            if not isinstance(info, dict):
                raise ValueError("Invalid response")
            if info.get("name") != name:
                raise SystemExit("Pod identity mismatch; refusing further mutations")
            current = float(info.get("costPerHr") or rate)
            if not math.isfinite(current) or current < 0:
                raise ValueError("Invalid rate")
            rate = max(rate, current)
            failures = 0
            status = info.get("desiredStatus")
            emit(
                "sample",
                pod_id=pod,
                age_seconds=round(age),
                status=status,
                estimated_cost_with_reserve=round(bound, 3),
            )
            if status in ("EXITED", "STOPPED", "TERMINATED"):
                emit(
                    "compute_stopped_cleanup_required",
                    pod_id=pod,
                    platform_terminate_epoch=cfg["platform_terminate_epoch"],
                )
                return
        except (ValueError, TypeError, KeyError):
            failures += 1
            emit(
                "control_failure",
                pod_id=pod,
                failures=failures,
                platform_terminate_epoch=cfg["platform_terminate_epoch"],
            )
            # Fail closed after repeated control errors, even before budget expires.
            if failures >= 3:
                stopping = True
        time.sleep(5 if stopping else min(30, max(0, stop_at - time.time())))


if __name__ == "__main__":
    main()
