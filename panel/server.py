#!/usr/bin/env python3
"""Read-only single-node vLLM dashboard for the DGX Spark."""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
HOST = os.environ.get("PANEL_HOST", "127.0.0.1")
PORT = int(os.environ.get("PANEL_PORT", "18103"))
TARGET = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:18102").rstrip("/")
DB_PATH = Path(os.environ.get("PANEL_DB", str(ROOT / "panel.db")))
BESZEL_URL = os.environ.get("BESZEL_URL", "")
POLL_SECONDS = max(1.0, float(os.environ.get("PANEL_POLL_SECONDS", "2")))
HISTORY_DAYS = max(1, int(os.environ.get("PANEL_HISTORY_DAYS", "14")))

SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def fetch_text(url: str, timeout: float = 3.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "spark-llm-panel/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def fetch_json(url: str, timeout: float = 3.0) -> Any:
    return json.loads(fetch_text(url, timeout))


def parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {
        key: bytes(value, "utf-8").decode("unicode_escape")
        for key, value in LABEL_RE.findall(raw)
    }


def parse_prometheus(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line.strip())
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        samples.append((name, parse_labels(raw_labels), value))
    return samples


def sum_metric(samples: list[tuple[str, dict[str, str], float]], name: str) -> float:
    return sum(value for metric, _, value in samples if metric == name)


def position_metric(
    samples: list[tuple[str, dict[str, str], float]], name: str
) -> dict[int, float]:
    result = defaultdict(float)
    for metric, labels, value in samples:
        if metric == name and labels.get("position", "").isdigit():
            result[int(labels["position"])] += value
    return dict(result)


def histogram_quantile(
    samples: list[tuple[str, dict[str, str], float]], name: str, quantile: float
) -> float | None:
    buckets = defaultdict(float)
    for metric, labels, value in samples:
        if metric != f"{name}_bucket" or "le" not in labels:
            continue
        try:
            upper = float(labels["le"])
        except ValueError:
            continue
        buckets[upper] += value
    if not buckets:
        return None
    ordered = sorted(buckets.items())
    total = ordered[-1][1]
    if total <= 0:
        return None
    rank = total * quantile
    previous_upper = 0.0
    previous_count = 0.0
    for upper, count in ordered:
        if count >= rank:
            if math.isinf(upper):
                return previous_upper
            span = count - previous_count
            if span <= 0:
                return upper
            fraction = (rank - previous_count) / span
            return previous_upper + (upper - previous_upper) * fraction
        previous_upper = upper
        previous_count = count
    return None


def delta(current: float, previous: float | None) -> float:
    if previous is None or current < previous:
        return 0.0
    return current - previous


def clean_number(value: float | None, digits: int = 3) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


@dataclass
class CounterState:
    timestamp: float
    generation: float
    prompt: float
    drafts: float
    draft_tokens: float
    accepted_tokens: float
    positions: dict[int, float]
    prefix_queries: float
    prefix_hits: float


class Monitor:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.snapshot: dict[str, Any] = {
            "online": False,
            "updated_at": int(time.time()),
            "error": "Waiting for first sample",
            "beszel_url": BESZEL_URL,
        }
        self.previous: CounterState | None = None
        self.last_history_minute: int | None = None
        self.last_history_write = 0.0
        self.stop = threading.Event()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    ts INTEGER PRIMARY KEY,
                    generation REAL NOT NULL,
                    prompt REAL NOT NULL,
                    kv REAL NOT NULL,
                    running INTEGER NOT NULL,
                    waiting INTEGER NOT NULL,
                    acceptance_length REAL,
                    acceptance_rate REAL,
                    prefix_hit_rate REAL,
                    ttft_p95_ms REAL,
                    tpot_p95_ms REAL,
                    e2e_p95_ms REAL
                )
                """
            )

    def _save_history(self, snapshot: dict[str, Any]) -> None:
        minute = int(snapshot["updated_at"] // 60 * 60)
        now = time.monotonic()
        if minute == self.last_history_minute and now - self.last_history_write < 10:
            return
        self.last_history_minute = minute
        self.last_history_write = now
        latency = snapshot["latency"]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ts) DO UPDATE SET
                    generation = MAX(samples.generation, excluded.generation),
                    prompt = MAX(samples.prompt, excluded.prompt),
                    kv = MAX(samples.kv, excluded.kv),
                    running = MAX(samples.running, excluded.running),
                    waiting = MAX(samples.waiting, excluded.waiting),
                    acceptance_length = COALESCE(excluded.acceptance_length, samples.acceptance_length),
                    acceptance_rate = COALESCE(excluded.acceptance_rate, samples.acceptance_rate),
                    prefix_hit_rate = COALESCE(excluded.prefix_hit_rate, samples.prefix_hit_rate),
                    ttft_p95_ms = excluded.ttft_p95_ms,
                    tpot_p95_ms = excluded.tpot_p95_ms,
                    e2e_p95_ms = excluded.e2e_p95_ms
                """,
                (
                    minute,
                    snapshot["generation_tok_s"],
                    snapshot["prompt_tok_s"],
                    snapshot["kv_cache_percent"],
                    snapshot["running_requests"],
                    snapshot["waiting_requests"],
                    snapshot["acceptance_length"],
                    snapshot["acceptance_rate"],
                    snapshot["prefix_cache_hit_rate"],
                    latency["ttft_p95_ms"],
                    latency["tpot_p95_ms"],
                    latency["e2e_p95_ms"],
                ),
            )
            cutoff = minute - HISTORY_DAYS * 86400
            connection.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))

    def collect(self) -> dict[str, Any]:
        now = time.time()
        health = fetch_text(f"{TARGET}/health").strip()
        models_response = fetch_json(f"{TARGET}/v1/models")
        samples = parse_prometheus(fetch_text(f"{TARGET}/metrics"))
        models = [item.get("id", "") for item in models_response.get("data", [])]

        current = CounterState(
            timestamp=now,
            generation=sum_metric(samples, "vllm:generation_tokens_total"),
            prompt=sum_metric(samples, "vllm:prompt_tokens_total"),
            drafts=sum_metric(samples, "vllm:spec_decode_num_drafts_total"),
            draft_tokens=sum_metric(samples, "vllm:spec_decode_num_draft_tokens_total"),
            accepted_tokens=sum_metric(samples, "vllm:spec_decode_num_accepted_tokens_total"),
            positions=position_metric(
                samples, "vllm:spec_decode_num_accepted_tokens_per_pos_total"
            ),
            prefix_queries=sum_metric(samples, "vllm:prefix_cache_queries_total"),
            prefix_hits=sum_metric(samples, "vllm:prefix_cache_hits_total"),
        )

        elapsed = now - self.previous.timestamp if self.previous else 0.0
        elapsed = elapsed if elapsed > 0 else 1.0
        generation_delta = delta(
            current.generation, self.previous.generation if self.previous else None
        )
        prompt_delta = delta(current.prompt, self.previous.prompt if self.previous else None)
        drafts_delta = delta(current.drafts, self.previous.drafts if self.previous else None)
        draft_tokens_delta = delta(
            current.draft_tokens, self.previous.draft_tokens if self.previous else None
        )
        accepted_delta = delta(
            current.accepted_tokens,
            self.previous.accepted_tokens if self.previous else None,
        )
        query_delta = delta(
            current.prefix_queries,
            self.previous.prefix_queries if self.previous else None,
        )
        hit_delta = delta(
            current.prefix_hits, self.previous.prefix_hits if self.previous else None
        )

        per_position = []
        for position in sorted(current.positions):
            old = self.previous.positions.get(position) if self.previous else None
            accepted_at_position = delta(current.positions[position], old)
            per_position.append(
                clean_number(accepted_at_position / drafts_delta * 100, 1)
                if drafts_delta > 0
                else None
            )

        acceptance_rate = (
            accepted_delta / draft_tokens_delta * 100 if draft_tokens_delta > 0 else None
        )
        acceptance_length = (
            1 + accepted_delta / drafts_delta if drafts_delta > 0 else None
        )
        prefix_hit_rate = hit_delta / query_delta * 100 if query_delta > 0 else None

        if acceptance_length is None:
            with self.lock:
                acceptance_length = self.snapshot.get("acceptance_length")
                acceptance_rate = self.snapshot.get("acceptance_rate")
                previous_positions = self.snapshot.get("per_position_acceptance")
            if previous_positions:
                per_position = previous_positions

        snapshot = {
            "online": health.lower() in {"", "ok", "healthy"},
            "updated_at": int(now),
            "error": None,
            "target": TARGET,
            "models": models,
            "generation_tok_s": clean_number(generation_delta / elapsed, 2),
            "prompt_tok_s": clean_number(prompt_delta / elapsed, 2),
            "running_requests": int(sum_metric(samples, "vllm:num_requests_running")),
            "waiting_requests": int(sum_metric(samples, "vllm:num_requests_waiting")),
            "kv_cache_percent": clean_number(
                sum_metric(samples, "vllm:kv_cache_usage_perc") * 100, 1
            ),
            "acceptance_rate": clean_number(acceptance_rate, 1),
            "acceptance_length": clean_number(acceptance_length, 2),
            "accepted_tok_s": clean_number(accepted_delta / elapsed, 2),
            "drafted_tok_s": clean_number(draft_tokens_delta / elapsed, 2),
            "per_position_acceptance": per_position,
            "prefix_cache_hit_rate": clean_number(prefix_hit_rate, 1),
            "preemptions_total": int(sum_metric(samples, "vllm:num_preemptions_total")),
            "latency": {
                "ttft_p95_ms": clean_number(
                    (histogram_quantile(samples, "vllm:time_to_first_token_seconds", 0.95) or 0)
                    * 1000,
                    1,
                ),
                "tpot_p95_ms": clean_number(
                    (
                        histogram_quantile(
                            samples, "vllm:request_time_per_output_token_seconds", 0.95
                        )
                        or 0
                    )
                    * 1000,
                    1,
                ),
                "e2e_p95_ms": clean_number(
                    (histogram_quantile(samples, "vllm:e2e_request_latency_seconds", 0.95) or 0)
                    * 1000,
                    1,
                ),
            },
            "beszel_url": BESZEL_URL,
        }
        self.previous = current
        with self.lock:
            self.snapshot = snapshot
        self._save_history(snapshot)
        return snapshot

    def collect_loop(self) -> None:
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                self.collect()
            except Exception as error:  # keep the last good payload visible
                with self.lock:
                    self.snapshot = {
                        **self.snapshot,
                        "online": False,
                        "updated_at": int(time.time()),
                        "error": str(error),
                        "beszel_url": BESZEL_URL,
                    }
            delay = max(0.2, POLL_SECONDS - (time.monotonic() - started))
            self.stop.wait(delay)

    def get_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.snapshot)

    def get_history(self, hours: int) -> list[dict[str, Any]]:
        hours = min(max(hours, 1), HISTORY_DAYS * 24)
        cutoff = int(time.time()) - hours * 3600
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT ts, generation, prompt, kv, running, waiting,
                       acceptance_length, acceptance_rate, prefix_hit_rate,
                       ttft_p95_ms, tpot_p95_ms, e2e_p95_ms
                FROM samples WHERE ts >= ? ORDER BY ts
                """,
                (cutoff,),
            ).fetchall()
        keys = (
            "ts",
            "generation",
            "prompt",
            "kv",
            "running",
            "waiting",
            "acceptance_length",
            "acceptance_rate",
            "prefix_hit_rate",
            "ttft_p95_ms",
            "tpot_p95_ms",
            "e2e_p95_ms",
        )
        return [dict(zip(keys, row)) for row in rows]


MONITOR: Monitor | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "SparkLlmPanel/1"

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self._headers(status, content_type, len(body))
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(
            status,
            "application/json; charset=utf-8",
            json.dumps(payload, separators=(",", ":")).encode(),
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/snapshot":
            assert MONITOR is not None
            self._json(MONITOR.get_snapshot())
            return
        if parsed.path == "/api/history":
            assert MONITOR is not None
            query = urllib.parse.parse_qs(parsed.query)
            try:
                hours = int(query.get("hours", ["24"])[0])
            except ValueError:
                self._json({"error": "hours must be an integer"}, 400)
                return
            self._json({"hours": hours, "samples": MONITOR.get_history(hours)})
            return
        if parsed.path == "/healthz":
            self._json({"ok": True})
            return

        requested = "/index.html" if parsed.path == "/" else parsed.path
        static_files = {
            "/index.html": STATIC / "index.html",
            "/styles.css": STATIC / "styles.css",
            "/app.js": STATIC / "app.js",
        }
        file_path = static_files.get(requested)
        if file_path is None or not file_path.is_file():
            self._json({"error": "not found"}, 404)
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self._send(200, content_type, file_path.read_bytes())

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.path not in {"/api/snapshot", "/healthz"}:
            super().log_message(fmt, *args)


def main() -> None:
    global MONITOR
    MONITOR = Monitor(DB_PATH)
    collector = threading.Thread(target=MONITOR.collect_loop, daemon=True)
    collector.start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"spark-llm-panel listening on http://{HOST}:{PORT} -> {TARGET}", flush=True)
    try:
        server.serve_forever()
    finally:
        MONITOR.stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
