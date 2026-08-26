#!/usr/bin/env python3
"""Compare two raw deterministic canary captures without republishing text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def first_difference(left: str, right: str) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    left_rows = {row["name"]: row for row in left["canaries"]}
    right_rows = {row["name"]: row for row in right["canaries"]}
    if left_rows.keys() != right_rows.keys():
        raise RuntimeError("canary names differ")

    rows = []
    for name in left_rows:
        left_text = left_rows[name]["content"]
        right_text = right_rows[name]["content"]
        rows.append(
            {
                "name": name,
                "exact_match": left_text == right_text,
                "left_sha256": digest(left_text),
                "right_sha256": digest(right_text),
                "left_chars": len(left_text),
                "right_chars": len(right_text),
                "first_difference_char": first_difference(left_text, right_text),
            }
        )

    result = {
        "schema_version": 1,
        "left_label": left["label"],
        "right_label": right["label"],
        "all_exact_match": all(row["exact_match"] for row in rows),
        "canaries": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
