#!/usr/bin/env python3
"""Inspect real ELF/cubins in a built image without accessing a GPU."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    engine = Path("/opt/strata/engine")
    meta = json.loads((engine / "BUILD.json").read_text())
    if not {80, 89} <= set(meta.get("archs", [])):
        raise SystemExit("BUILD.json does not cover both target GPUs")
    rows = []
    for name in ("strata", "strata-vision"):
        path = engine / name
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit("Missing executable: " + name)
        header = subprocess.check_output(["readelf", "-h", str(path)], text=True)
        if "Advanced Micro Devices X86-64" not in header:
            raise SystemExit("Wrong CPU architecture: " + name)
        cubins = subprocess.check_output(
            ["/usr/local/cuda/bin/cuobjdump", "--list-elf", str(path)], text=True
        )
        archs = sorted({int(x) for x in re.findall(r"sm_(\d+)\.cubin", cubins)})
        if not {80, 89} <= set(archs):
            raise SystemExit("Missing target GPU cubins: " + name)
        linked = subprocess.check_output(["ldd", str(path)], text=True)
        # Host libcuda is supplied by the NVIDIA container runtime, never bundled.
        missing = [
            line.strip()
            for line in linked.splitlines()
            if "not found" in line and "libcuda.so.1 " not in line
        ]
        if missing:
            raise SystemExit("Missing runtime libraries: " + ", ".join(missing))
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        rows.append(
            dict(
                file=name,
                bytes=path.stat().st_size,
                sha256=digest,
                gpu_architectures=archs,
            )
        )
    print(json.dumps({"cpu": "x86_64", "binaries": rows, "build": meta}, indent=2))


if __name__ == "__main__":
    main()
