#!/usr/bin/env python3
"""Validate greedy canary constraints without republishing response text."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def strip_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:python|json)?\s*(.*?)\s*```\s*", text, re.S)
    return match.group(1) if match else text.strip()


def validate(name: str, text: str) -> tuple[bool, list[str]]:
    checks: list[tuple[str, bool]]
    if name == "code":
        code = strip_fence(text)
        try:
            tree = ast.parse(code)
            parsed = True
        except SyntaxError:
            tree = ast.parse("")
            parsed = False
        checks = [
            ("python_parses", parsed),
            (
                "has_function",
                any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)),
            ),
            (
                "three_asserts",
                sum(isinstance(node, ast.Assert) for node in ast.walk(tree)) >= 3,
            ),
        ]
    elif name == "chinese_reasoning":
        checks = [
            ("answer_90", bool(re.search(r"(?<!\d)90(?:\.0+)?(?!\d)", text))),
            ("has_chinese", bool(re.search(r"[\u4e00-\u9fff]", text))),
        ]
    elif name == "multilingual":
        checks = [
            ("has_chinese", bool(re.search(r"[\u4e00-\u9fff]", text))),
            ("has_japanese_kana", bool(re.search(r"[\u3040-\u30ff]", text))),
            ("multiple_lines", len([line for line in text.splitlines() if line]) >= 2),
        ]
    elif name == "structured":
        try:
            value = json.loads(strip_fence(text))
        except json.JSONDecodeError:
            value = None
        checks = [
            ("valid_json", isinstance(value, dict)),
            (
                "exact_keys",
                isinstance(value, dict) and set(value) == {"name", "primes", "valid"},
            ),
            ("name", isinstance(value, dict) and value.get("name") == "canary"),
            (
                "primes",
                isinstance(value, dict) and value.get("primes") == [2, 3, 5, 7, 11],
            ),
            ("valid", isinstance(value, dict) and value.get("valid") is True),
        ]
    else:
        return False, ["unknown_canary"]
    failures = [label for label, passed in checks if not passed]
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source: dict[str, Any] = json.loads(args.input.read_text())
    rows = []
    for row in source["canaries"]:
        text = row.get("content") or ""
        passed, failures = validate(row["name"], text)
        rows.append(
            {
                "name": row["name"],
                "passed": passed,
                "failures": failures,
                "sha256": digest(text),
                "characters": len(text),
                "finish_reason": row.get("finish_reason"),
                "usage": row.get("usage") or {},
            }
        )
    result = {
        "schema_version": 1,
        "source_label": source.get("label"),
        "all_passed": all(row["passed"] for row in rows),
        "canaries": rows,
        "response_text_included": False,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
