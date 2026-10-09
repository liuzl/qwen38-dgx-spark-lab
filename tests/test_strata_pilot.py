import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from strata_http import Response  # noqa: E402  # noqa: E402
from strata_protocols import qualify  # noqa: E402  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            if self.path == "/headers":
                time.sleep(3)
            self.send_response(200)
            self.end_headers()
            if self.path == "/body":
                for _ in range(50):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.03)
            elif body.get("stream"):
                for _ in range(100):
                    self.wfile.write(
                        b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
                    )
                    self.wfile.flush()
                    time.sleep(0.03)
            else:
                self.wfile.write(b'{"ok":true}')
        except (BrokenPipeError, ConnectionResetError):
            pass


class DeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_slow_headers_body_and_endless_stream(self):
        for path, payload in [
            ("/headers", {}),
            ("/body", {}),
            ("/stream", {"stream": True}),
        ]:
            with self.subTest(path=path):
                start = time.monotonic()
                with (
                    self.assertRaises(TimeoutError),
                    Response(self.url + path, payload, start + 0.3) as response,
                ):
                    if payload.get("stream"):
                        list(response)
                    else:
                        response.read()
                self.assertLess(time.monotonic() - start, 1.2)

    def test_complete_json(self):
        with Response(self.url + "/", {}, time.monotonic() + 2) as response:
            self.assertEqual(json.load(response), {"ok": True})
        self.assertIsNotNone(response.proc.poll())

    def test_stream_never_marked_success_after_deadline(self):
        pilot = load("strata-pilot")
        with tempfile.TemporaryDirectory() as d:
            r = pilot.Runner(self.url, d, 0.005)
            row = r.simple("endless", "x", stream=True)
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["error"], "TimeoutError")


class WatchTests(unittest.TestCase):
    def run_watch(self, mode):
        watch = load("strata-runpod-watch")
        calls = []
        cfg = dict(
            pod_id="test-id",
            pod_name="strata-pilot-test",
            created_epoch=1,
            hourly_rate=2,
            platform_terminate_epoch=10801,
            platform_termination_verified=True,
        )

        def run(args, **kw):
            calls.append(args[2])
            if mode == "timeout":
                raise subprocess.TimeoutExpired(args, 20)
            return subprocess.CompletedProcess(args, 1, "", "")

        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / "state"
            state.write_text(json.dumps(cfg))
            with (
                patch("sys.argv", ["watch", "--state", str(state)]),
                patch.object(watch.time, "time", return_value=20000),
                patch.object(watch.subprocess, "run", side_effect=run),
                patch.object(
                    watch.time, "sleep", side_effect=RuntimeError("end simulation")
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "end simulation"),
            ):
                watch.main()
        return calls

    def test_get_failure_cannot_skip_stop(self):
        self.assertEqual(self.run_watch("failure"), ["stop", "get"])

    def test_command_timeout_cannot_skip_stop(self):
        self.assertEqual(self.run_watch("timeout"), ["stop", "get"])

    def test_platform_guard_required(self):
        w = load("strata-runpod-watch")
        with self.assertRaises(ValueError):
            w.validate(
                dict(
                    pod_id="x",
                    pod_name="strata-pilot-x",
                    created_epoch=1,
                    hourly_rate=1,
                    platform_terminate_epoch=20000,
                    platform_termination_verified=True,
                )
            )


class DownloadTests(unittest.TestCase):
    def test_token_removed_on_network_and_hash_failure(self):
        download = load("strata-download")
        for failure in ("network", "hash"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                folder = Path(d)
                token = folder / "token"
                token.write_text("hf_test_only")
                data = folder / "data"
                data.write_bytes(b"abc")
                manifest = folder / "manifest"
                manifest.write_text(
                    json.dumps(
                        [
                            dict(
                                repo="fake",
                                revision="a" * 40,
                                file="f",
                                bytes=3,
                                lfs={"oid": "0" * 64},
                            )
                        ]
                    )
                )

                def fetch(failure=failure, data=data, token_path=token, **kw):
                    self.assertFalse(token_path.exists())
                    if failure == "network":
                        raise OSError("simulated")
                    return str(data)

                fake = types.SimpleNamespace(hf_hub_download=fetch)
                with (
                    patch.dict(sys.modules, {"huggingface_hub": fake}),
                    patch(
                        "sys.argv",
                        [
                            "download",
                            "--token-file",
                            str(token),
                            "--out",
                            d,
                            "--manifest",
                            str(manifest),
                        ],
                    ),
                    self.assertRaises((OSError, SystemExit)),
                ):
                    download.main()
                self.assertFalse(token.exists())


class ProtocolTests(unittest.TestCase):
    def test_200_without_tool_call_fails(self):
        class Runner:
            def http(self, *args):
                return io.StringIO("{}")

            def save(self, row):
                pass

        for protocol in ("responses", "messages"):
            self.assertFalse(qualify(Runner(), protocol)["pass"])

    def test_quality_incomplete_is_not_pass(self):
        q = load("strata-quality")
        with (
            tempfile.TemporaryDirectory() as d,
            patch("sys.argv", ["quality", "--out", d, "--minutes", "0"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            q.main()
            summary = json.loads((Path(d) / "quality-summary.json").read_text())
            self.assertEqual(summary["not_run"], 12)
            self.assertFalse(summary["quality_gate_pass"])
            self.assertEqual(len(json.loads((Path(d) / "cases.json").read_text())), 12)


class RepairTests(unittest.TestCase):
    def test_hidden_boundary_and_rounding_cases(self):
        q = load("strata-quality")
        self.assertTrue(q.score({"case": "code-boundary"}, '{"fix":"return xs[-1]"}'))
        self.assertFalse(q.score({"case": "code-boundary"}, '{"fix":"return xs[0]"}'))
        self.assertTrue(
            q.score({"case": "code-ceil-div"}, '{"fix":"return (n + d - 1) // d"}')
        )
        self.assertFalse(q.score({"case": "code-ceil-div"}, '{"fix":"return n // d"}'))
        self.assertFalse(
            q.score({"case": "code-ceil-div"}, '{"fix":"return __import__(chr(111))"}')
        )


class ProtocolCompletionTests(unittest.TestCase):
    def test_responses_requires_usage_and_complete_stream(self):
        first = {
            "output": [
                {
                    "type": "function_call",
                    "name": "lookup",
                    "call_id": "call-1",
                    "arguments": '{"item":"alpha"}',
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        final = {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "42"}]}
            ],
            "usage": {"input_tokens": 20, "output_tokens": 2},
        }

        class Runner:
            def save(self, row):
                pass

            def http(self, path, payload):
                if not payload.get("stream"):
                    return io.StringIO(json.dumps(first))
                return io.BytesIO(
                    b"data: "
                    + json.dumps(
                        {"type": "response.completed", "response": final}
                    ).encode()
                    + b"\n\n"
                )

        self.assertTrue(qualify(Runner(), "responses")["pass"])
        final.pop("usage")
        self.assertFalse(qualify(Runner(), "responses")["pass"])

    def test_messages_requires_final_stop(self):
        first = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "name": "lookup",
                    "id": "t1",
                    "input": {"item": "alpha"},
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        events = [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 20, "output_tokens": 0}},
            },
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "42"},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]

        class Runner:
            def save(self, row):
                pass

            def http(self, path, payload):
                if not payload.get("stream"):
                    return io.StringIO(json.dumps(first))
                return io.BytesIO(
                    b"".join(
                        b"data: " + json.dumps(x).encode() + b"\n\n" for x in events
                    )
                )

        self.assertTrue(qualify(Runner(), "messages")["pass"])
        events.pop()
        self.assertFalse(qualify(Runner(), "messages")["pass"])


if __name__ == "__main__":
    unittest.main()
