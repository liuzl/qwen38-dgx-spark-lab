# A100 image-serving performance tuning

## Protocol

The 2026-09-06 experiment compares MTP depth on the image-enabled serving
profile: pinned FP8 checkpoint and vLLM 0.28.0, CUDA Graph execution, 128K
context, 24 GiB BF16 KV, 32 active sequences, 16K batched tokens, and disabled
multimodal processor cache. Thinking is disabled for every measured request.

`scripts/benchmark-serving-matrix.py` runs three repetitions for both the base
and adapter models. Each repetition includes 1K/C1, 1K/C8, 16K/C1, and a
1K-plus-image/C1 case. All measured outputs must contain exactly 256 tokens.
The shared streaming harness reports server usage, TTFT, decode rate, and
end-to-end aggregate output rate; it never counts SSE events as tokens.

Candidate arms use identical prompts and fresh processes, with distinct
prefixes for repetitions and warmups. Each shape has an eight-token warmup.
Speculative counter deltas include those warmups and must be interpreted
separately from measured output throughput.

An initial online run was contaminated by concurrent interactive use and used
different prefix labels. It is exploratory only and excluded from candidate
selection. The controlled run requires a maintenance window because a second
full model does not fit alongside the production process. The original
container is stopped, not deleted, and restored when the sweep exits.

The image workload measures image-prefill overhead while generating code; it
does not measure OCR accuracy or image reasoning quality. API/tool checks and
separate semantic image checks are required before promoting a candidate.

## Depth results

Three-repetition medians; all rates are tokens/second. Each cell lists
base / adapter. C8 aggregate includes prefill and the complete request window;
C1 decode excludes TTFT. The six arms completed 396 measured requests, each
with 256 generated tokens, with no request failures.

| MTP depth | Short C1 decode | C8 aggregate | Image C1 decode |
|---|---:|---:|---:|
| 1 | 72.73 / 64.57 | 276.95 / 250.44 | 71.53 / 63.56 |
| 2 | 102.77 / 91.11 | 321.09 / 293.07 | 103.45 / 89.92 |
| 3 | 121.74 / 111.45 | 342.87 / 317.86 | 121.00 / 109.32 |
| 4 | 139.70 / 127.12 | 357.66 / 330.01 | 133.96 / 117.90 |
| 5 | 145.81 / 131.37 | 359.34 / 330.15 | 147.66 / 128.15 |
| 7 | 157.46 / 146.44 | 357.35 / 333.30 | 159.99 / 132.83 |

K7 improves short C1 decode over K3 by about 29% / 31%, but C8 aggregate only
improves by about 4% / 5%. Long-input TTFT remains about 6.5 / 7.0 seconds
across depths. Do not describe decode gains as equal end-to-end latency gains.

Acceptance including warmups falls from 96.17% at K1 to 71.31% at K7. The
declining percentage does not imply declining throughput: more accepted tokens
per target verification offset additional drafting cost on these workloads.
The fixed 24 GiB KV pool retains 314,572 tokens at K7 (2.40 full 128K requests),
versus 338,297 at K3. Graph memory rises from 1.93 to 3.01 GiB.

Raw measurements are the `a100-image-mtp-k*-2026-09-06.json` files in
`benchmarks/results/`. They contain no generated response text. K6 and K8+
were not tested; K7 is the best sampled depth for single-stream interaction,
not a proven global optimum for every workload.

## Batch budget and decision

At K7, reducing the batch budget from 16,384 to 8,192 tokens did not materially
improve performance. Base C8 aggregate was 357.35 versus 357.64 tok/s, and
adapter C8 was 333.30 versus 331.46. Base/adapter 16K TTFT was 6.493/6.978
seconds at 16K batch and 6.490/6.985 at 8K batch. Keep the 16K budget.

Select K7 with the existing 128K admission limit, C32 scheduler, 24 GiB BF16
KV, and 16K batch budget for the current controlled-test service. This favors
interactive single-stream generation while retaining C8 aggregate performance.
The seven controlled arms completed 462 measured requests with no failures.
Higher depths, larger batches, other prompt distributions, and sustained
mixed-user load remain unqualified. Long-context prefill is still the main
latency limitation; this sweep does not establish a prefill speedup.

## Deployment validation

The K7 service passed all four semantic canaries for each alias, 64/64 requests
at C32 for each alias, and base/adapter prefix-cache isolation. Public image
requests passed all six model/protocol combinations, with remote URL and
five-image requests still rejected. The candidate container is stopped after
testing; the normal service retains its automatic restart policy.

A final production replay measured 157.16/146.57 tok/s short C1 decode, in
line with the isolated arm. A 115,247-token text-plus-image adapter request
passed its semantic check in 66.87 seconds. GPU use after that shape was
71,955 MiB with 9,082 MiB free (including existing co-resident processes).
This confirms capacity, not a long-prefill speedup.
