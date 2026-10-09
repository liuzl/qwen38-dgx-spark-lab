#!/usr/bin/env python3
"""Bounded synthetic Strata qualification; no credentials or customer data."""

import argparse
import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from strata_http import Response
from strata_protocols import qualify, usage_ok

MODEL = "flash-next-orca-iq3xxs"
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up the integer value of a named item.",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
        },
    },
}


class Runner:
    def __init__(self, base, output, minutes):
        self.base = base.rstrip("/")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + minutes * 60

    def save(self, row):
        with (self.output / "requests.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in (
                        "case",
                        "status",
                        "pass",
                        "seconds",
                        "input_tokens",
                        "cached_tokens",
                        "output_tokens",
                        "ttft",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def http(self, path, payload=None, timeout=120):
        return Response(self.base + path, payload, self.deadline, timeout)

    def call(self, case, messages, *, stream=False, cancel=False, **options):
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            **options,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        start = time.monotonic()
        text = ""
        usage = {}
        finish = None
        ttft = None
        chunks = 0
        message = {}
        done = False
        tool_parts = {}
        row = {
            "case": case,
            "request_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
        }
        try:
            with self.http("/v1/chat/completions", payload) as response:
                row["headers_seconds"] = time.monotonic() - start
                if stream:
                    for line in response:
                        if not line.startswith(b"data: "):
                            continue
                        if line.strip() == b"data: [DONE]":
                            done = True
                            break
                        event = json.loads(line[6:])
                        choices = event.get("choices") or []
                        if event.get("usage"):
                            usage = event["usage"]
                        if choices:
                            delta_obj = choices[0].get("delta", {})
                            for part in delta_obj.get("tool_calls", []):
                                entry = tool_parts.setdefault(
                                    part["index"],
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if part.get("id"):
                                    entry["id"] += part["id"]
                                for key in ("name", "arguments"):
                                    entry["function"][key] += part.get(
                                        "function", {}
                                    ).get(key, "")
                            delta = delta_obj.get("content") or ""
                            if delta:
                                if ttft is None:
                                    ttft = time.monotonic() - start
                                chunks += 1
                                text += delta
                            finish = choices[0].get("finish_reason") or finish
                        if cancel and chunks >= 4:
                            break
                    row["cancelled_after_chunks"] = chunks if cancel else None
                    row["stream_complete"] = done and finish is not None
                    message = {"role": "assistant", "content": text or None}
                    if tool_parts:
                        message["tool_calls"] = [
                            tool_parts[k] for k in sorted(tool_parts)
                        ]
                    if not cancel and not row["stream_complete"]:
                        raise ValueError("Incomplete SSE")
                else:
                    data = json.load(response)
                    usage = data.get("usage", {})
                    choices = data.get("choices") or []
                    if choices:
                        message = choices[0].get("message", {})
                        text = message.get("content") or ""
                        finish = choices[0].get("finish_reason")
                row.update(
                    status=200,
                    response_text=text,
                    message=message,
                    finish_reason=finish,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    cached_tokens=(usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens"
                    ),
                    ttft=ttft,
                    usage=usage,
                )
        except urllib.error.HTTPError as e:
            row.update(status=e.code, error=e.read(1500).decode(errors="replace"))
        except Exception as e:
            row.update(status="error", error=type(e).__name__)
        row["seconds"] = time.monotonic() - start
        if ttft is not None and row.get("output_tokens") and row["seconds"] > ttft:
            row["client_decode_tps"] = max(0, row["output_tokens"] - 1) / (
                row["seconds"] - ttft
            )
        return row

    def simple(self, case, prompt, **kw):
        return self.call(case, [{"role": "user", "content": prompt}], **kw)

    def tools(self, messages=None, item="alpha", case="tool"):
        msgs = [
            {
                "role": "user",
                "content": (
                    "Use lookup for item alpha. After receiving its value, return"
                    " that integer plus 25, with only the integer."
                ),
            }
        ]
        if messages is not None:
            msgs = messages
        first = self.call(case + "-request", msgs, tools=[TOOL], stream=True)
        self.save(first)
        answer = first.get("message") or {}
        calls = answer.get("tool_calls") or []
        ok = (
            first.get("finish_reason") == "tool_calls"
            and usage_ok(first.get("usage"), "prompt_tokens", "completion_tokens")
            and len(calls) == 1
            and bool(calls[0].get("id"))
            and calls[0].get("function", {}).get("name") == "lookup"
        )
        if ok:
            try:
                ok = json.loads(calls[0]["function"]["arguments"]) == {"item": item}
            except (ValueError, KeyError):
                ok = False
        if not ok:
            self.save(
                {"case": case + "-roundtrip", "status": "invalid_call", "pass": False}
            )
            return
        msgs.extend(
            [answer, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "17"}]
        )
        final = self.call(case + "-roundtrip", msgs, tools=[TOOL], stream=True)
        final["pass"] = (
            final.get("response_text", "").strip() == "42"
            and final.get("finish_reason") == "stop"
            and usage_ok(final.get("usage"), "prompt_tokens", "completion_tokens")
        )
        self.save(final)

    def api(self, image):
        row = self.simple("text", 'Return exactly this JSON: {"ready":true}')
        row["pass"] = row.get(
            "response_text", ""
        ).strip() == '{"ready":true}' and usage_ok(
            row.get("usage"), "prompt_tokens", "completion_tokens"
        )
        self.save(row)
        row = self.simple(
            "stream", "Count from 1 to 20, one number per line.", stream=True
        )
        row["pass"] = (
            row.get("status") == 200
            and usage_ok(row.get("usage"), "prompt_tokens", "completion_tokens")
            and row.get("finish_reason") == "stop"
            and row.get("response_text", "").strip().splitlines()
            == [str(x) for x in range(1, 21)]
        )
        self.save(row)
        self.tools()
        if image:
            data = base64.b64encode(Path(image).read_bytes()).decode()
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Read the image. Reply JSON with code (after C"
                                "ODE:), blue_squ"
                                "ares, red_circles."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + data},
                        },
                    ],
                }
            ]
            row = self.call("vision", msgs)
            text = row.get("response_text", "")
            try:
                observed = json.loads(text[text.index("{") : text.rindex("}") + 1])
            except (ValueError, TypeError):
                observed = None
            row["pass"] = observed == {
                "code": "K7M42",
                "blue_squares": 3,
                "red_circles": 2,
            }
            self.save(row)
            msgs[0]["content"][0]["text"] = (
                "Read CODE from the image. Use lookup with that exact code as"
                " item; after its result return its value plus 25, only the i"
                "nteger."
            )
            self.tools(messages=msgs, item="K7M42", case="vision-tool")
        row = self.simple(
            "cancel",
            "Write the integers 1 through 10000, one per line, do not summarize.",
            stream=True,
            cancel=True,
            max_tokens=2048,
        )
        self.save(row)
        start = time.monotonic()
        cancel_started = (
            row.get("cancelled_after_chunks", 0) >= 4
            and row.get("finish_reason") is None
        )
        old_deadline = self.deadline
        self.deadline = min(self.deadline, start + 5)
        idle = False
        samples = []
        try:
            while time.monotonic() < self.deadline:
                with self.http("/metrics", timeout=2) as response:
                    metrics = response.read().decode()
                values = re.findall(
                    ("^vllm:num_requests_running(?:\\{[^}]*\\})?\\s+([0-9.eE+-]+)"),
                    metrics,
                    re.M,
                )
                running = sum(float(x) for x in values) if values else None
                samples.append(
                    {"seconds": time.monotonic() - start, "running": running}
                )
                if running == 0:
                    idle = True
                    break
                time.sleep(min(0.1, max(0, self.deadline - time.monotonic())))
            row = self.simple("after-cancel", "Reply exactly RECOVERED")
        except Exception as e:
            row = {"case": "after-cancel", "status": "error", "error": type(e).__name__}
        finally:
            self.deadline = old_deadline
        row["recovery_seconds"] = time.monotonic() - start
        row["idle_samples"] = samples
        row["pass"] = (
            cancel_started
            and idle
            and row.get("response_text", "").strip() == "RECOVERED"
            and row["recovery_seconds"] <= 5
        )
        self.save(row)
        for protocol in ("responses", "messages"):
            qualify(self, protocol)

    def prefix(self, target, tag, suffix="Reply with the sum only."):
        # Count via the server's actual tokenizer. Final Chat usage is authoritative.
        unit = "Record alpha has value 17. Record beta has value 25. The sum is 42.\n"
        lo, hi = 1, max(10, target // 4)
        best = unit
        for _ in range(17):
            n = (lo + hi) // 2
            text = tag + "\n" + unit * n + "\n" + suffix
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 64,
                "thinking": {"type": "disabled"},
            }
            with self.http(
                "/v1/messages/count_tokens", payload, timeout=30
            ) as response:
                count = json.load(response)["input_tokens"]
            if count <= target:
                best = text
                lo = n + 1
            else:
                hi = n - 1
            if lo > hi:
                break
        return best

    def cache(self):
        for trial in range(3):
            prompt = self.prefix(16384, "salt-test-" + uuid.uuid4().hex)
            for label, salt in [
                ("A1", "tenant-A"),
                ("A2", "tenant-A"),
                ("B1", "tenant-B"),
                ("B2", "tenant-B"),
            ]:
                row = self.simple(
                    f"cache-{trial}-{label}", prompt, cache_salt=salt, max_tokens=32
                )
                self.save(row)
        for key in ("A", "B", "C"):
            msgs = [
                {
                    "role": "user",
                    "content": self.prefix(
                        32768, "serial-" + key + "-" + uuid.uuid4().hex
                    ),
                }
            ]
            for turn in range(4):
                row = self.call(f"serial-{key}-{turn}", msgs, max_tokens=64)
                self.save(row)
                if row.get("status") != 200:
                    break
                msgs.extend(
                    [
                        {"role": "assistant", "content": row.get("response_text", "")},
                        {
                            "role": "user",
                            "content": (
                                "Confirm the same sum again, with only the integer."
                            ),
                        },
                    ]
                )
        # Single-slot interleaving, separate from simultaneous concurrency measurements.
        sessions = {
            k: [
                {
                    "role": "user",
                    "content": self.prefix(32768, k + "-" + uuid.uuid4().hex),
                }
            ]
            for k in ("A", "B", "C")
        }
        for turn in range(4):
            for key, msgs in sessions.items():
                row = self.call(f"interleave-{key}-{turn}", msgs, max_tokens=64)
                self.save(row)
                if row.get("status") != 200:
                    continue
                msgs.extend(
                    [
                        {"role": "assistant", "content": row.get("response_text", "")},
                        {
                            "role": "user",
                            "content": (
                                "Confirm the same sum again, with only the integer."
                            ),
                        },
                    ]
                )

    def perf(self, concurrency):
        targets = [1024, 16384, 65536] if concurrency == 1 else [16384]
        for target in targets:
            for rep in range(4):
                prompts = [
                    self.prefix(
                        target,
                        uuid.uuid4().hex,
                        (
                            "Write a Python function implementing a stable"
                            " merge sort and"
                            " explain it."
                        ),
                    )
                    for _ in range(concurrency)
                ]
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    rows = list(
                        pool.map(
                            lambda p, target=target, rep=rep: self.simple(
                                f"perf-{target}-c{concurrency}-r{rep}", p, stream=True
                            ),
                            prompts,
                        )
                    )
                elapsed = time.monotonic() - started
                for row in rows:
                    row.update(
                        warmup=rep == 0, concurrency=concurrency, group_seconds=elapsed
                    )
                    self.save(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--out", required=True)
    p.add_argument("--phase", choices=["api", "cache", "perf"], required=True)
    p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--image")
    p.add_argument("--concurrency", type=int, default=1)
    a = p.parse_args()
    run = Runner(a.base, a.out, a.minutes)
    if a.phase == "api":
        run.api(a.image)
    elif a.phase == "cache":
        run.cache()
    else:
        run.perf(a.concurrency)


if __name__ == "__main__":
    main()
