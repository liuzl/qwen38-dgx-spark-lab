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

## Batch-size-dependent MTP depth

The 2026-09-07 experiment tested whether speculative depth should fall as the
scheduled batch grows. vLLM 0.28.0 has no `disable_by_batch_size`; its
replacement is `num_speculative_tokens_per_batch_size`, a list of inclusive
`(range_start, range_end, K)` entries keyed by the number of requests scheduled
in a step, with `K=0` allowed. `scripts/sweep-a100-mtp-arms.sh` ran each arm in
a fresh process under the production image profile with 1K prompts at C1, C8,
C16, and C32, three repetitions, 256 output tokens per request. The static
K7 C1 and C8 results reproduce the 2026-09-06 arm within 1%.

| Arm | Short C1 decode | C8 aggregate | C16 aggregate | C32 aggregate |
|---|---:|---:|---:|---:|
| Static K7 | 157.41 / 146.23 | 354.59 / 331.79 | 393.35 / 364.45 | 375.04 / 350.25 |
| Static K3 | 121.72 / 111.30 | 340.72 / 317.07 | 395.68 / 370.15 | 423.25 / 393.62 |
| Static K1 | 74.56 / 62.08 | 277.68 / 250.73 | 354.59 / 325.35 | 401.98 / 373.61 |
| Dynamic 7/4/2 (1-4/5-16/17-32) | 155.09 / 144.32 | 353.34 / 325.48 | 374.16 / 349.88 | 373.77 / 347.87 |
| Dynamic 7/3/0 (1-4/5-16/17-32) | 143.76 / 144.34 | 336.40 / 313.87 | 393.02 / 367.31 | 328.70 / 302.13 |
| Static K7, V2 model runner | 159.58 / 148.09 | 357.01 / 332.31 | 393.56 / 364.14 | 376.32 / 350.91 |

Static results confirm the premise: the best depth falls with batch size. K7
leads at C1 and C8, K3 and K7 tie at C16, and K3 beats K7 by about 13% / 12% at
C32. K1 is also faster than K7 at C32. Disabling speculation entirely at high
batch is the worst option measured: the `K=0` range is 22% below static K3.

The dynamic mechanism does not deliver that gain on this pinned runtime.
Enabling a schedule makes vLLM 0.28.0 override the CUDA Graph mode from
`FULL_AND_PIECEWISE` to `PIECEWISE`; the warning suggests
`VLLM_USE_V2_MODEL_RUNNER=1` to keep full graphs. Under the default runner the
7/4/2 schedule matched static K7 at C1 but reached only 373.77 tok/s at C32,
below static K3, because the piecewise penalty exceeds the drafting savings.
Under the V2 runner the static K7 control reproduced the baseline within 2%,
but the dynamic arm failed at CUDA Graph capture with
`assert 0 < num_reqs <= num_tokens` in `InputBatch.make_dummy`, called from the
V2 speculator capture path. Raw server logs remain on the test node.

Decision: keep static K7 for the current interactive-first service. Dynamic
depth is not usable on vLLM 0.28.0 and should be retested only after a runtime
upgrade. If the workload shifts to sustained C16-plus multi-user load, switch
to static K3 or K4 rather than a schedule; that trade costs about 23% of
single-stream decode for about 13% more C32 aggregate throughput. Raw
measurements are the `a100-mtp-batch-*-2026-09-07.json` files in
`benchmarks/results/`; the aborted V2 static K3 control has no result file.
