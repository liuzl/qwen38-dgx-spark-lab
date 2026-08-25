# Benchmark methodology

## Environment

- NVIDIA DGX Spark, GB10 `sm_121`, 128 GB unified memory
- Ubuntu 24.04 / DGX OS, CUDA 13.0, driver 580.173.02
- vLLM commit `f94666b60d4c58ec0807d22c837cfae322a1dde9`
- `RadixArk/Qwen3.8-27B-NVFP4` target
- `z-lab/Qwen3.8-27B-DFlash2` drafter
- DFlash2 probabilistic sampling, depth 7
- FP8 E4M3 KV cache, 16 GiB
- thinking disabled, temperature 0

Checkpoint repositories can change in place. Pin revisions and record hashes in
your own run manifest.

## C1

One random request, fixed 512-token requested input and 2,048 generated tokens,
ignore EOS, concurrency 1. Five repetitions established the clean baseline;
the native-LoRA qualification table reports the first matched run after warmup.

## C8

Thirty-two random requests, fixed 1,024-token requested input and 256 generated
tokens, ignore EOS, concurrency 8, seed 101.

The tokenizer rendered 526 actual tokens in the C1 workload and approximately
1,036 per C8 request. Report output throughput, TTFT, request failures, DFlash
acceptance length, and per-position acceptance—not tok/s alone.

## Safety behavior

StrongREJECT Small was run with 60 prompts, 2,048 maximum output tokens,
temperature 0, thinking off, and concurrency 4.

The literal string matcher reported 8/60, but every match came from ordinary
answer text containing broad markers such as `legally`, `illegal`,
`legal ways`, or `harm reduction`. Human inspection treated them as false
positives. The local refusal classifier was the binding result:

```text
strict refusal           0 / 60
disclaimer with answer   8 / 60
normal answer           52 / 60
empty/error              0 / 60
```

Raw responses are intentionally not published. The aggregate result is under
`benchmarks/results/`.

## Limits

- Results are one machine and one pinned software/checkpoint combination.
- No 24-hour soak result is claimed by this repository version.
- DFlash acceptance is workload-sensitive.
- C8 native-LoRA figures are single qualification runs, not confidence
  intervals.
- An adapter that passes refusal testing still requires normal capability,
  agent/tool, and domain-specific evaluation for its intended use.
