"""HTTP proxy in front of vLLM for lm-eval runs.

Forwards every request; logs non-200 responses with the request body. Works
around the vLLM 0.28.0 NaN-prompt-logprob band (241-255 prompt tokens on the
A100 profile): when an echo/logprobs request fails with the NaN 400, retry with
leading newline padding until it leaves the band, and log the padded request
separately so the count is reported with the run. Padding is applied
identically to every arm, so relative comparisons are unaffected.
"""

import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP, PORT, LOG = sys.argv[1], int(sys.argv[2]), sys.argv[3]
PAD_LOG = (
    LOG.replace(".non200.jsonl", ".padded.jsonl")
    if LOG.endswith(".non200.jsonl")
    else LOG + ".padded"
)


def call(path, body, method):
    req = urllib.request.Request(
        UP + path,
        data=body,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self._fwd("GET")

    def do_POST(self):
        self._fwd("POST")

    def _fwd(self, method):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n) if n else None
        status, data = call(self.path, body, method)
        if status == 400 and body and b"nan" in data.lower() and b'"echo"' in body:
            try:
                req = json.loads(body)
                prompts = (
                    req["prompt"]
                    if isinstance(req["prompt"], list)
                    else [req["prompt"]]
                )
                for pad in range(1, 17):
                    padded = dict(req)
                    padded["prompt"] = (
                        ["\n" * pad + p for p in prompts]
                        if isinstance(req["prompt"], list)
                        else "\n" * pad + prompts[0]
                    )
                    status, data = call(self.path, json.dumps(padded).encode(), method)
                    if status == 200:
                        with open(PAD_LOG, "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "pad_tokens": pad,
                                        "n_prompts": len(prompts),
                                        "path": self.path,
                                    }
                                )
                                + "\n"
                            )
                        break
            except Exception as exc:  # noqa: BLE001 - report and fall through to logging
                data = json.dumps({"proxy_error": str(exc)}).encode()
        if status != 200:
            with open(LOG, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "status": status,
                            "path": self.path,
                            "response": data.decode(errors="replace")[:600],
                            "request": (body or b"").decode(errors="replace")[:200000],
                        }
                    )
                    + "\n"
                )
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
