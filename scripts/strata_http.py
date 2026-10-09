"""Deadline-bounded HTTP, including slow headers/bodies and endless SSE.
Each request runs in a killable child; parent owns the absolute wall-clock deadline.
No credentials are passed on argv. Timing includes local worker startup overhead.
"""

import io
import json
import os
import selectors
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class Response:
    def __init__(self, url, payload, deadline, timeout=120):
        self.deadline = min(deadline, time.monotonic() + timeout)
        self.buffer = b""
        self.proc = None
        self.selector = selectors.DefaultSelector()
        try:
            if self.deadline <= time.monotonic():
                raise TimeoutError("phase deadline")
            self.proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self.proc.stdin.write(
                json.dumps(
                    {"url": url, "payload": payload, "timeout": timeout}
                ).encode()
            )
            self.proc.stdin.close()
            self.selector.register(self.proc.stdout, selectors.EVENT_READ)
            first = self.event()
            if first["kind"] == "http_error":
                raise urllib.error.HTTPError(
                    url,
                    first["status"],
                    "HTTP failure",
                    {},
                    io.BytesIO(first["body"].encode()),
                )
            if first["kind"] != "headers":
                raise OSError(first.get("error", "HTTP worker failed"))
        except BaseException:
            self.close()
            raise

    def event(self):
        while True:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("request/phase wall-clock deadline")
            if b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                event = json.loads(line)
                if event["kind"] == "error":
                    raise OSError(event["error"])
                return event
            if not self.selector.select(left):
                raise TimeoutError("request/phase wall-clock deadline")
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise OSError("HTTP worker ended without completion")
            self.buffer += chunk

    def __iter__(self):
        while True:
            event = self.event()
            if event["kind"] == "end":
                return
            if event["kind"] == "line":
                yield event["text"].encode()

    def read(self, size=-1):
        event = self.event()
        if event["kind"] != "body":
            raise OSError("Expected JSON body")
        body = event["text"].encode()
        return body if size < 0 else body[:size]

    def close(self):
        self.selector.close()
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=5)
            self.proc.stdout.close()
            if not self.proc.stdin.closed:
                self.proc.stdin.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def worker():
    def send(kind, **fields):
        print(json.dumps(dict(kind=kind, **fields)), flush=True)

    try:
        cfg = json.load(sys.stdin)
        payload = cfg["payload"]
        req = urllib.request.Request(
            cfg["url"],
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=cfg["timeout"]) as r:
            send("headers", status=r.status)
            if payload and payload.get("stream"):
                for line in r:
                    send("line", text=line.decode())
                send("end")
            else:
                send("body", text=r.read().decode())
    except urllib.error.HTTPError as e:
        send("http_error", status=e.code, body=e.read(1500).decode(errors="replace"))
    except Exception as e:
        send("error", error=type(e).__name__)


if __name__ == "__main__":
    worker()
