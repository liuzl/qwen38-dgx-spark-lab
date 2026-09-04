from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("benchmark-openai.py")
SPEC = importlib.util.spec_from_file_location("benchmark_openai", SCRIPT)
assert SPEC and SPEC.loader
benchmark_openai = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_openai
SPEC.loader.exec_module(benchmark_openai)


class FakeStream:
    def __init__(self, lines: list[bytes]):
        self.lines = iter(lines)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.lines)


class BenchmarkOpenAITest(unittest.TestCase):
    def test_prompt_markers_are_unique(self):
        case = benchmark_openai.DEFAULT_CASES[0]
        first = benchmark_openai.prompt_for(case, 0)
        second = benchmark_openai.prompt_for(case, 1)
        warm = benchmark_openai.prompt_for(case, 0, warmup=True)
        self.assertNotEqual(first[0]["content"], second[0]["content"])
        self.assertNotEqual(first[0]["content"], warm[0]["content"])

    @mock.patch.object(benchmark_openai.urllib_request, "urlopen")
    @mock.patch.object(benchmark_openai.time, "perf_counter")
    def test_done_terminates_stream_and_usage_is_required(self, clock, urlopen):
        chunks = [
            {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]},
            {
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "timings": {
                    "draft_n": 10,
                    "draft_n_accepted": 8,
                },
            },
        ]
        lines = [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks]
        lines.append(b"data: [DONE]\n")
        urlopen.return_value = FakeStream(lines)
        clock.side_effect = [10.0, 10.5, 12.0]

        result = benchmark_openai.stream_request(
            base_url="http://example.invalid",
            api_key=None,
            model="model",
            case=benchmark_openai.DEFAULT_CASES[0],
            request_index=0,
            max_tokens=8,
            timeout=5,
        )

        self.assertEqual(result["prompt_tokens"], 100)
        self.assertEqual(result["completion_tokens"], 8)
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["ttft_s"], 0.5)
        self.assertEqual(result["total_s"], 2.0)
        self.assertEqual(result["itl_ms"], 214.2857)
        self.assertEqual(result["runtime_timings"]["draft_n_accepted"], 8)
        self.assertNotIn("content", result)


if __name__ == "__main__":
    unittest.main()
