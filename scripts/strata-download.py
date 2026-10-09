#!/usr/bin/env python3
"""Download a pinned, hash-checked manifest without printing credentials."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--token-file", required=True)
    a = p.parse_args()
    try:
        from huggingface_hub import hf_hub_download

        token = Path(a.token_file).read_text().strip()
        # Remove the transport copy before any long-running network operation.
        Path(a.token_file).unlink()
        if not token.startswith("hf_"):
            raise SystemExit("Missing HF token")
        for row in json.loads(Path(a.manifest).read_text()):
            if not re.fullmatch(r"[0-9a-f]{40}", row["revision"]):
                raise ValueError("Manifest must pin a commit revision")
            expected = row["lfs"]["oid"]
            if len(expected) != 64 or not all(
                c in "0123456789abcdef" for c in expected
            ):
                raise ValueError("Invalid manifest hash")
            path = Path(
                hf_hub_download(
                    repo_id=row["repo"],
                    revision=row["revision"],
                    filename=row["file"],
                    local_dir=a.out,
                    token=token,
                )
            )
            if path.stat().st_size != row["bytes"]:
                raise ValueError("Size mismatch")
            digest = hashlib.sha256()
            with path.open("rb") as f:
                while data := f.read(8 * 1024 * 1024):
                    digest.update(data)
            if digest.hexdigest() != expected:
                raise SystemExit("SHA256 mismatch: " + row["file"])
            print(
                json.dumps(
                    {
                        "file": row["file"],
                        "bytes": row["bytes"],
                        "sha256": expected,
                        "verified": True,
                    }
                ),
                flush=True,
            )
    finally:
        Path(a.token_file).unlink(missing_ok=True)
    print("Manifest complete; temporary credential removed", flush=True)


if __name__ == "__main__":
    main()
