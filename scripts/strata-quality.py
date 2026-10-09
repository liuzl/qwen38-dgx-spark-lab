#!/usr/bin/env python3
"""Small fixed synthetic task set; never execute model-generated code."""

import argparse
import ast
import base64
import importlib.util
import json
import operator
import time
from pathlib import Path

from strata_fixtures import create

spec = importlib.util.spec_from_file_location(
    "pilot", Path(__file__).with_name("strata-pilot.py")
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

CASES = [
    (
        "code-boundary",
        (
            "Python def last(xs): return xs[len(xs)] has a bug. Return JS"
            "ON with only key fix containing the corrected return stateme"
            "nt."
        ),
        {"fix": "return xs[-1]"},
    ),
    (
        "code-ceil-div",
        (
            "Fix def ceil_div(n, d): return n // d for nonnegative intege"
            "rs n and positive integers d. Return only JSON with fix cont"
            "aining the corrected return statement. Use integer arithmeti"
            "c, no function calls."
        ),
        None,
    ),
    (
        "code-sort",
        (
            'For stable sort of [(2,"a"),(1,"b"),(2,"c")] by the first fi'
            "eld, return JSON key result with the sorted pairs."
        ),
        {"result": [[1, "b"], [2, "a"], [2, "c"]]},
    ),
    (
        "code-edit-honesty",
        (
            'Only allowed file: settings.json, currently {"retries":3,"ti'
            'meout":10}. Change retries to 4 and preserve timeout. Return'
            " JSON with files (mapping filename to JSON object), tested=f"
            'alse, status="not_run". You have not run tests. Do not edit '
            "any other file."
        ),
        {
            "files": {"settings.json": {"retries": 4, "timeout": 10}},
            "tested": False,
            "status": "not_run",
        },
    ),
    (
        "zh-extract",
        "资料：甲项目预算120元，已用45元；乙项目预算90元，已用91元。只返回JSON：甲剩余多少remaining_a；乙是否超预算over_b。",
        {"remaining_a": 75, "over_b": True},
    ),
    (
        "zh-conflict",
        (
            "记录一：订单A数量7。记录二：同一订单A数量9。没有日期或权威标记。返回JSON：conflict为是否冲突，confi"
            "rmed_quantity无从确定则null。"
        ),
        {"conflict": True, "confirmed_quantity": None},
    ),
    (
        "zh-summary",
        "材料：任务甲已完成，任务乙等待外部数据，任务丙未开始。只返回JSON，done数组、blocked数组、pending数组；数组中只写甲乙丙。",
        {"done": ["甲"], "blocked": ["乙"], "pending": ["丙"]},
    ),
]


def parsed(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def safe_expression(node, env):
    # A tiny arithmetic/subscript interpreter, not eval/exec of model-generated code.
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and abs(node.value) < 100000
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -safe_expression(node.operand, env)
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }
    if isinstance(node, ast.BinOp) and type(node.op) in ops:
        return ops[type(node.op)](
            safe_expression(node.left, env), safe_expression(node.right, env)
        )
    if isinstance(node, ast.Subscript):
        return safe_expression(node.value, env)[safe_expression(node.slice, env)]
    raise ValueError("Expression outside allowed repair grammar")


def score(case, text):
    value = parsed(text)
    if case["case"] not in ("code-boundary", "code-ceil-div"):
        return value == case["expected"]
    try:
        if (
            not isinstance(value, dict)
            or set(value) != {"fix"}
            or len(value["fix"]) > 200
        ):
            return False
        tree = ast.parse(value["fix"])
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
            return False
        expr = tree.body[0].value
        tests = (
            [({"xs": xs}, xs[-1]) for xs in ([8], [2, 7], ["x", "y", "z"])]
            if case["case"] == "code-boundary"
            else [
                ({"n": n, "d": d}, (n + d - 1) // d)
                for n, d in ((0, 3), (1, 3), (3, 3), (4, 3), (18, 7), (99, 10))
            ]
        )
        return all(safe_expression(expr, env) == expected for env, expected in tests)
    except (
        ValueError,
        TypeError,
        SyntaxError,
        IndexError,
        ZeroDivisionError,
        KeyError,
    ):
        return False


def image_messages(path, prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(Path(path).read_bytes()).decode()
                    },
                },
            ],
        }
    ]


def tool_case(r, recovery=False):
    name = "tool-error-recovery" if recovery else "tool-dependency"
    prompt = (
        "Use lookup for item alpha, then return only JSON with answer"
        " equal to its value plus 25."
    )
    if recovery:
        prompt += " If lookup fails temporarily, retry once before answering."
    msgs = [{"role": "user", "content": prompt}]
    evidence = []
    calls_needed = 2 if recovery else 1
    for index in range(calls_needed):
        row = r.call(name + "-call-" + str(index), msgs, tools=[pilot.TOOL])
        evidence.append(row)
        msg = row.get("message", {})
        calls = msg.get("tool_calls") or []
        if not (
            row.get("finish_reason") == "tool_calls"
            and len(calls) == 1
            and calls[0].get("id")
            and calls[0].get("function", {}).get("name") == "lookup"
            and parsed(calls[0]["function"].get("arguments")) == {"item": "alpha"}
        ):
            return {
                "case": name,
                "pass": False,
                "status": "invalid_call",
                "evidence": evidence,
            }
        value = (
            '{"error":"temporary_unavailable","retryable":true}'
            if recovery and index == 0
            else "17"
        )
        msgs.extend(
            [msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": value}]
        )
    final = r.call(name + "-answer", msgs, tools=[pilot.TOOL])
    evidence.append(final)
    return {
        "case": name,
        "pass": parsed(final.get("response_text")) == {"answer": 42}
        and final.get("finish_reason") == "stop",
        "status": final.get("status"),
        "evidence": evidence,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--out", required=True)
    p.add_argument("--minutes", type=float, default=20)
    a = p.parse_args()
    r = pilot.Runner(a.base, a.out, a.minutes)
    folder = Path(a.out) / "fixtures"
    images = create(folder)
    specs = [
        {"case": name, "prompt": prompt, "expected": expected}
        for name, prompt, expected in CASES
    ]
    specs.extend(
        [
            {"case": "tool-dependency"},
            {"case": "tool-error-recovery"},
            *images,
            {"case": "long-evidence"},
        ]
    )
    if len(specs) != 12:
        raise ValueError("Exactly twelve independent tasks required")
    (Path(a.out) / "cases.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=2)
    )
    phase_deadline = r.deadline
    results = []
    for case in specs:
        name = case["case"]
        if time.monotonic() >= phase_deadline:
            row = {"case": name, "status": "not-run", "pass": None}
        else:
            r.deadline = min(phase_deadline, time.monotonic() + 180)
            try:
                if name.startswith("tool-"):
                    row = tool_case(r, name == "tool-error-recovery")
                elif name.startswith("vision-"):
                    msgs = image_messages(folder / case["file"], case["prompt"])
                    row = r.call(name, msgs, max_tokens=512)
                    row["fixture_sha256"] = case["sha256"]
                    row["pass"] = parsed(row.get("response_text")) == case["expected"]
                elif name == "long-evidence":
                    filler = r.prefix(
                        16384,
                        "quality-long-context",
                        "Read the facts and answer the final question.",
                    )
                    middle = len(filler) // 2
                    prompt = (
                        "FACT: project maple owner is Lin.\n"
                        + filler[:middle]
                        + "\nFACT: Lin manages team Cedar.\n"
                        + filler[middle:]
                        + (
                            "\nFACT: team Cedar budget is 73.\nReturn only J"
                            "SON with owner "
                            "and budget for project maple."
                        )
                    )
                    row = r.simple(name, prompt, max_tokens=512)
                    row["input"] = prompt
                    row["pass"] = parsed(row.get("response_text")) == {
                        "owner": "Lin",
                        "budget": 73,
                    }
                else:
                    row = r.simple(name, case["prompt"], max_tokens=512)
                    row["pass"] = score(case, row.get("response_text"))
                row["expected"] = case.get("expected")
            except Exception as e:
                row = {
                    "case": name,
                    "status": "error",
                    "error": type(e).__name__,
                    "pass": False,
                }
            finally:
                r.deadline = phase_deadline
        results.append(row)
        r.save(row)
    summary = {
        "total": 12,
        "passed": sum(x["pass"] is True for x in results),
        "failed": sum(x["pass"] is False for x in results),
        "not_run": sum(x["pass"] is None for x in results),
    }
    summary["quality_gate_pass"] = (
        summary["not_run"] == 0
        and summary["passed"] >= 10
        and all(x["pass"] is True for x in results if x["case"].startswith("tool-"))
    )
    summary["scope"] = (
        "synthetic smoke with restricted repair interpreter and propo"
        "sed file edits; no general agent benchmark"
    )
    (Path(a.out) / "quality-summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
