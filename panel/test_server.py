from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from panel import server


FIXTURE = """
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="qwen3.8-27b"} 120
vllm:prompt_tokens_total{model_name="qwen3.8-27b"} 80
vllm:num_requests_running{model_name="qwen3.8-27b"} 2
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="qwen3.8-27b"} 2
vllm:time_to_first_token_seconds_bucket{le="0.5",model_name="qwen3.8-27b"} 8
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="qwen3.8-27b"} 10
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 9
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 7
"""


class MetricsTests(unittest.TestCase):
    def test_parser_and_sum(self):
        samples = server.parse_prometheus(FIXTURE)
        self.assertEqual(server.sum_metric(samples, "vllm:generation_tokens_total"), 120)
        self.assertEqual(server.sum_metric(samples, "vllm:num_requests_running"), 2)

    def test_histogram_quantile(self):
        samples = server.parse_prometheus(FIXTURE)
        value = server.histogram_quantile(
            samples, "vllm:time_to_first_token_seconds", 0.5
        )
        self.assertAlmostEqual(value, 0.3)

    def test_position_series(self):
        samples = server.parse_prometheus(FIXTURE)
        self.assertEqual(
            server.position_metric(
                samples, "vllm:spec_decode_num_accepted_tokens_per_pos_total"
            ),
            {0: 9, 1: 7},
        )

    def test_history_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = server.Monitor(Path(directory) / "test.db")
            self.assertEqual(monitor.get_history(24), [])

    def test_history_keeps_minute_peak(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = server.Monitor(Path(directory) / "test.db")
            snapshot = {
                "updated_at": 1_800_000_000,
                "generation_tok_s": 12.0,
                "prompt_tok_s": 20.0,
                "kv_cache_percent": 4.0,
                "running_requests": 1,
                "waiting_requests": 0,
                "acceptance_length": 3.0,
                "acceptance_rate": 40.0,
                "prefix_cache_hit_rate": None,
                "latency": {
                    "ttft_p95_ms": 100.0,
                    "tpot_p95_ms": 20.0,
                    "e2e_p95_ms": 500.0,
                },
            }
            monitor._save_history(snapshot)
            monitor.last_history_write = 0
            monitor._save_history({**snapshot, "generation_tok_s": 5.0})
            rows = monitor.get_history(24 * 365 * 10)
            self.assertEqual(rows[0]["generation"], 12.0)

    def test_service_card_escapes_dynamic_content(self):
        card = server.service_card(
            "Unsafe <name>",
            "Description & detail",
            "TAILNET ONLY",
            "X",
            [("Service", True, "HTTP 200")],
            [("Open", "https://example.com/?a=1&b=2")],
        )
        self.assertIn("Unsafe &lt;name&gt;", card)
        self.assertIn("Description &amp; detail", card)
        self.assertIn("a=1&amp;b=2", card)


if __name__ == "__main__":
    unittest.main()
