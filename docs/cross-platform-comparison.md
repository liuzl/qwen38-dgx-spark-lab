# DGX Spark vs Apple M3 Max

This comparison runs one client-owned prompt corpus against both serving
stacks. It measures the best qualified stack on each machine, not bare silicon:

| Platform | Serving stack |
|---|---|
| DGX Spark GB10, 128 GB | vLLM + mixed NVFP4 target + DFlash2 K7 + FP8 KV |
| Apple M3 Max, 64 GB | oMLX 0.6.3rc3 + oQ4e checkpoint + Lightning MTP |

The servers use different quantization and speculative-decoding algorithms by
design. Results answer "which qualified stack serves this workload faster?",
not "which chip has more theoretical compute?"

## Protocol

[`scripts/benchmark-openai.py`](../scripts/benchmark-openai.py) sends identical
messages through `/v1/chat/completions`:

- deterministic repeated Python corpus with a unique prefix per request;
- thinking off, temperature 0, requested output 256 tokens;
- streaming usage; no response text is retained;
- same-shape warmup before every measured case;
- case order: short C1, short C4, sustained 16K C1;
- decode rate excludes TTFT and the first generated token;
- aggregate output rate includes the whole measured concurrent window.

Actual rendered prompts were 1,080 and 16,345 tokens. All 20 measured requests
completed 256 tokens with `finish_reason=length`; neither result file contains a
request failure or cache hit.

## Results

| Case | Metric | DGX Spark | M3 Max | M3 as % of Spark |
|---|---|---:|---:|---:|
| PP1080/TG256 C1 | decode tok/s | 60.38 | 44.24 | 73.3% |
| PP1080/TG256 C1 | TTFT | 0.55 s | 5.57 s | 10.2x slower |
| PP1080/TG256 C1 | total latency | 4.77 s | 11.34 s | 2.38x slower |
| PP1080/TG256 C4 | aggregate tok/s | 143.70 | 18.19 | 12.7% |
| PP1080/TG256 C4 | mean decode tok/s/request | 47.43 | 7.59 | 16.0% |
| PP1080/TG256 C4 | mean latency | 7.07 s | 53.07 s | 7.51x slower |
| PP16345/TG256 C1 | decode tok/s | 69.83 | 14.65 | 21.0% |
| PP16345/TG256 C1 | TTFT | 1.82 s | 158.75 s | 87.3x slower |
| PP16345/TG256 C1 | total latency | 5.47 s | 176.15 s | 32.2x slower |

The M3 Max comes closest on short single-stream decode. It does not catch the
Spark on sustained long context or concurrency. During the M3 C4 run, oMLX
disabled Lightning MTP for multi-row batches because standard batched decode
was selected as faster; this is a runtime-policy difference that materially
affects the serving result.

## Prefix-cache interpretation

The unique-prefix protocol deliberately prevents cross-request cache reuse.
It should not be used to dismiss the separate Apple cache result: for a real
agent turn with a stable 5.2K prefix, the M3 Max reduced wall time from 25.19 to
6.41 seconds. That is a useful local-agent capability, but it is not a decode
speedup and does not change the no-cache cross-platform ranking.

## Artifacts

- [DGX Spark raw metrics](../benchmarks/results/cross-platform-dgx-spark-2026-08-26.json)
- [M3 Max raw metrics](../benchmarks/results/cross-platform-m3-max-2026-08-26.json)
- [Machine-readable comparison](../benchmarks/results/cross-platform-summary-2026-08-26.json)

## Limits

- One machine of each type and one qualification run per case.
- The long M3 result is intentionally sustained: it follows a same-shape
  warmup and the short C1/C4 cases, so it includes thermal reality rather than
  a cherry-picked cold peak.
- TTFT includes client/TLS/network overhead for Spark and loopback overhead for
  M3. The large gaps dominate that small transport asymmetry, but TTFT is not a
  pure kernel measurement.
- Speculative acceptance is workload-sensitive. The repeated code corpus is
  favorable to Spark DFlash2 and M3 Lightning MTP; prose may rank differently.
- Prefix caching, ANE, and TurboQuant were disabled for the matched M3 run.
- Repeat before making a production or purchasing decision.
