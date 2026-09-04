#!/usr/bin/env python3
"""Capture a sanitized NVIDIA qualification environment manifest.

The manifest deliberately excludes hostname, username, network addresses,
environment variables, credential locations, and absolute model/cache paths.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def command(*args: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def parse_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"ID", "VERSION_ID", "PRETTY_NAME"}:
            result[key.lower()] = value.strip('"')
    return result


def parse_gpus(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not result.get("ok"):
        return []
    fields = (
        "index",
        "name",
        "memory_total_mib",
        "compute_capability",
        "driver",
        "power_limit_w",
        "power_max_limit_w",
        "persistence_mode",
        "performance_state",
        "pcie_generation_current",
        "pcie_width_current",
    )
    rows = []
    for line in result["stdout"].splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values, strict=True)))
    return rows


def compute_process_summary() -> dict[str, Any]:
    result = command(
        "nvidia-smi",
        "--query-compute-apps=used_memory",
        "--format=csv,noheader,nounits",
    )
    if not result.get("ok"):
        return {"available": False, "process_count": None, "used_memory_mib": None}
    values = []
    for line in result["stdout"].splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return {
        "available": True,
        "process_count": len(values),
        "used_memory_mib": sum(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="container image reference to inspect")
    parser.add_argument("--model", help="public model ID, not a local path")
    parser.add_argument("--model-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.model and Path(args.model).is_absolute():
        parser.error("--model must be a public model ID, not an absolute path")

    gpu_query = command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap,driver_version,"
        "power.limit,power.max_limit,persistence_mode,pstate,"
        "pcie.link.gen.current,pcie.link.width.current",
        "--format=csv,noheader,nounits",
    )
    docker_version = command("docker", "version", "--format", "{{json .}}")
    image = None
    if args.image:
        image = command(
            "docker",
            "image",
            "inspect",
            args.image,
            "--format",
            "{{json .RepoDigests}}|{{.Id}}|{{.Created}}",
        )

    manifest = {
        "schema_version": 1,
        # This host-side probe intentionally supports the observed Python 3.10.
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "privacy": {
            "hostname_recorded": False,
            "username_recorded": False,
            "network_recorded": False,
            "absolute_paths_recorded": False,
        },
        "os": parse_os_release(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpus": parse_gpus(gpu_query),
        "compute_processes": compute_process_summary(),
        "nvidia_smi_error": None if gpu_query.get("ok") else gpu_query,
        "docker": docker_version,
        "image": image,
        "model": args.model,
        "model_revision": args.model_revision,
    }
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
