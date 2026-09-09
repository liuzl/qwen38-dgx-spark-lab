# A100 image-serving performance tuning

Latest recorded outcome: the September 8 service switch selected mixed INT8
W8A8, static native MTP K7 and adapter v2. The sections below retain each
experiment's decision at the time; Phase 1 and early Phase 2 pending items were
subsequently resolved by the [production switch](#production-switch-to-int8).
No completed 24-hour soak artifact is archived here as of 2026-09-09.
The [English X summary](https://x.com/liuzl/status/2097482398166598136) was
published on September 9.

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

## INT8 W8A8 prefill arms

The 2026-09-08 experiment executes Phase 1 of the
[INT8 prefill plan](a100-prefill-int8-plan.md). On Ampere the official FP8
checkpoint runs through Marlin as W8A16, so prefill pays BF16 compute plus
dequantization. The candidate `Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`
(revision `2df4e3b0`, compressed-tensors) routes 288 of 400 Linear modules to
`CutlassInt8ScaledMMLinearKernel` on INT8 Tensor Cores; `mlp.down_proj` and
`linear_attn.out_proj` stay W8A16 on Marlin, and the MTP head is BF16.

`scripts/sweep-a100-int8-arms.sh` ran three arms in fresh processes under the
production image profile with the adapter alias removed: the FP8 K7 control,
INT8 K7, and INT8 K3. Every other flag matched production (vLLM 0.28.0,
CUDA Graph, 128K context, 24 GiB BF16 KV, C32, 16K batch budget, image inputs
enabled, thinking disabled). Three repetitions, 256 output tokens per request,
base alias only. The kernel line in each server log was checked; the
`kernel gate FAILED` messages in the sweep log are a script false negative
(`grep -q` under `pipefail`) fixed after the run, and the recorded kernel lines
are authoritative. The control reproduced the 2026-09-06 and 2026-09-07 K7
results within 1%. All 45 measured cases across the three arms completed with
no request failures.

Medians of three repetitions, base alias, tokens/second unless noted:

| Shape | FP8 K7 (control) | INT8 K7 | INT8 K3 |
|---|---:|---:|---:|
| 1K C1 TTFT | 0.486 s | 0.299 s | 0.278 s |
| 1K C1 decode | 156.70 | 148.70 | 115.31 |
| 16K C1 TTFT | 6.482 s | **3.502 s** | 3.512 s |
| 16K C1 decode | 123.15 | 119.35 | 85.02 |
| 1K+image C1 TTFT | 0.575 s | 0.352 s | 0.350 s |
| 1K+image C1 decode | 148.68 | 144.46 | 113.25 |
| 1K C8 TTFT | 3.298 s | 1.706 s | 1.713 s |
| 1K C8 aggregate | 355.20 | 499.77 | 451.03 |
| 1K C32 TTFT | 11.859 s | 6.523 s | 5.849 s |
| 1K C32 aggregate | 372.81 | 603.18 | 699.50 |
| Acceptance length incl. warmup | 4.8-5.0 | 4.7-5.1 | 2.6-2.8 |

INT8 K7 cuts 16K TTFT by 46% and C8/C32 TTFT by 45-48%. Short C1 decode
falls 5% (156.70 to 148.70) and image C1 decode 3%; that is far below the 13%
single-stream penalty the checkpoint author reported and inside the plan's 15%
budget. Because prefill is shorter, C8 aggregate rises 41% and C32 aggregate
62%. Similar speculative acceptance and the verified kernel routing support
the interpretation that the main gain comes from the linear compute path;
this is not a hardware-counter isolation of every latency contribution. The
16,363-token prefill rate moves from about 2,520 to about 4,670 tok/s (prompt
tokens over TTFT), which exceeds the historical BF16 arm in the porting report
(about 3,600 tok/s) while keeping 8-bit weights. That BF16 arm used a different
profile, so this is contextual evidence rather than the controlled K7 A/B.

INT8 K3 gives the same TTFT as INT8 K7 (prefill does not depend on depth) but
loses 22-27% of single-stream decode and 10% of C8 aggregate. It wins only at
C32 (+16% over INT8 K7) with 7.5% more KV blocks (338,297 versus 314,572
tokens). The batch-size trade is the same one measured on FP8 on 2026-09-07:
K7 for interactive single-stream, K3 for sustained C16-plus load. The author's
K<=3 recommendation for hybrid-GDN targets is therefore an acceptance
observation, not a limit: INT8 K7 acceptance matched FP8 K7.

Capacity: model loading took 29.31 GiB for INT8 versus 29.36 GiB for FP8; the
fixed 24 GiB KV pool holds the same 314,572 tokens at K7; graph capture took
1.00 versus 0.96 GiB. No memory was traded for the speedup.

Decision at the end of Phase 1: INT8 K7 passed every criterion (16K TTFT below
5.0 s, no C8/C32 regression, decode loss under 15%) and was promoted to Phase 2.
It was **not yet** in production. Before the switch, the checkpoint had to pass
semantic canaries, forced-tool and API checks, and image checks for both aliases.
The rank-1 uncensored adapter also had to be re-derived against
the INT8 target and re-qualified; an FP8-derived adapter could not inherit
qualification for that target. Quality, long-context behaviour beyond 16K,
and sustained mixed load were still open at this stage. Subsequent gates and
the adapter mix-up are recorded below. Raw measurements are the
`a100-int8-w8a8-*-2026-09-08.json` files in `benchmarks/results/`; they
contain no generated response text. Server logs remain on the test node.

### Phase 2 base-alias quality gates

`scripts/qualify-a100-int8-candidate.sh` ran the INT8 K7 candidate on
2026-09-08 under the production profile with the adapter alias removed, in a
second maintenance window. The kernel gate confirmed
`CutlassInt8ScaledMMLinearKernel` before any check ran. Results for the base
alias:

| Gate | Result |
|---|---|
| API smoke: Chat Completions, Responses, Anthropic Messages, forced tool | passed |
| Four greedy semantic canaries (512-token cap) | 4/4 passed |
| Canaries versus the FP8 K7 production capture | code, multilingual, structured byte-identical; chinese_reasoning differs by one synonym at character 89, same derivation and answer |
| Image inputs, direct: OCR fixture through three protocols | 3/3 passed |
| Guardrails: remote media URL, five-image request | both rejected (400) |
| 64 requests at C32, exact-match consistency | 64/64, no failures, 3.29 s |

An earlier run the same day (`int8-phase2-20260908`) failed the
chinese_reasoning canary only because the driver capped canaries at 256
tokens and truncated the answer two characters before "90"; the FP8 reference
itself used 276 tokens. That run is superseded, not counted.

The image checks used `validate-multimodal.py --direct` against the
candidate's vLLM port rather than the public gateway, because the gateway
routes to production. Protocol coverage is the same; gateway authentication
and routing were not re-tested and are unchanged.

Status after Phase 2 for the base alias: **qualified for controlled
research**, same standing as the FP8 K7 base alias. Still open before the
service can switch: re-derive the rank-1 uncensored adapter against the INT8
target and re-run the adapter alias through the same gates plus prefix-cache
isolation; a StrongREJECT run on that adapter; and a production replay.
See `a100-int8-w8a8-phase2-base-qualification-2026-09-08.json` in
`benchmarks/results/` (hashes, counts and statuses only; no response text).

### Phase 2 adapter alias and combined gates

The rank-1 uncensored adapter had to be re-derived against the INT8 target.
`scripts/qwen38-rank1-to-lora.py` gained support for the compressed-tensors
layouts: INT8 per-channel `weight` with `weight_scale` (W8A8 modules) and
int32 `weight_packed` with `weight_shape` (W8A16 modules), with auxiliary
tensors resolved through the checkpoint index because this checkpoint places a
module's weight and scale in different shard files.

This took two attempts, and the first one produced a false pass that is worth
recording. The first decoder treated each packed byte as two's complement;
compressed-tensors actually stores the value plus 128 as an unsigned byte. That
gave wrong weights for all 112 W8A16 modules, so adapter v1 (SHA `a8d013f8`)
was invalid. It was not caught in its Phase 2 window because the driver set
`ADAPTER_DIR` and then sourced the production env file, which overwrote it: the
10:38 UTC window and the first four-arm evaluation served the **FP8-derived
adapter on the INT8 target**, which happens to work well, while every gate
reported "INT8 adapter". The defect surfaced only at the first production
switch (12:30 UTC), where the forced-tool check on the adapter alias returned
`{"city": "Lisbon"}` for a Singapore prompt and the switch rolled back
automatically. Diagnosis showed the v1 adapter gave Lisbon 40/40 times, cold
and warm, forced and auto tool choice, while other cities were correct and the
base alias was byte-identical to FP8 on every canary; a fresh torch.compile
cache changed nothing. Both bugs are fixed: the decoder subtracts the offset,
the converter test packs through the library's own `pack_to_int32`, and the
driver exports the candidate adapter path after sourcing the env.

Adapter v2 (SHA `9c2b7c40`, same refusal directions `9de12cbe`, 128 modules:
112 `int8_channel_packed_w8a16`, 16 `int8_channel_w8a8`) was then served with
both aliases in a 13:03 UTC window. All gates passed in one pass:

| Gate | Base | Adapter v2 |
|---|---|---|
| API smoke (Chat, Responses, Messages, forced tool) | passed | passed, tool argument Singapore |
| Greedy canaries, 512-token cap | 4/4 | 4/4, chinese_reasoning stops at 466 chars (v1 ran to the cap) |
| Canaries vs FP8 K7 production capture | 4/4 byte-identical | constraints pass; text differs (different target weights) |
| Image OCR through three protocols | 3/3 | 3/3 |
| Remote URL and five-image guardrails | rejected (400) | rejected (400) |
| Prefix-cache isolation | hit deltas `[0, 2496, 0, 2496]` | |
| 64 requests at C32 | 64/64 | 64/64 |

Details, including the superseded v1 record, are in
`a100-int8-w8a8-phase2-adapter-qualification-2026-09-08.json` (hashes, counts
and statuses only). The adapter stays private.

## Capability comparison across five arms

To check whether INT8 or the adapter changes model capability, the served
aliases were run through lm-evaluation-harness 0.4.13 on 2026-09-08 with
`scripts/run-capability-eval.sh`: identical prompts, identical deterministic
subsets, same vLLM profile. The FP8 arms ran on the production container; the
INT8 arms ran on the candidate inside Phase 2 windows. Because of the adapter
mix-up above there are five arms: the INT8 adapter column below is adapter v2,
and the FP8-derived adapter on the INT8 target is kept as an extra column since
it is a valid measurement in its own right.

The suite is a regression smoke sized for about 12 minutes per alias, not a
leaderboard run. Multiple-choice tasks run 0-shot through prompt logprobs,
because vLLM computes prompt logprobs with full-vocabulary logits per token
and cannot use the prefix cache for them; full MMLU 0-shot alone took 60
minutes and MMLU 5-shot projected to about 25 hours. GSM8K and IFEval go
through the chat template so the server's thinking-off default applies. On
the raw completions path the model emits `<think>` and exhausts the 256-token
cap before answering, which produced an invalid 68% GSM8K figure that is
recorded as superseded. HellaSwag, ARC and WinoGrande were dropped as
low-signal for this purpose.

Calibration: FP8 stage 1 and INT8 base each ran twice on identical prompts.
FP8 MMLU moved 1.3 points between replicates, INT8 MMLU 0.07, INT8 TruthfulQA
1.0, INT8 GSM8K 1.0. Differences under about 1.5 points are therefore not
resolvable by this suite. The full 14,042-question MMLU on FP8 base scored
83.3% ±0.3; the 25-per-subject subset scores about 2 points lower because of
subject weighting, identically for all arms.

| Task (subset) | FP8 base | FP8 adapter | INT8 base | INT8 adapter v2 | INT8 + FP8-derived adapter |
|---|---:|---:|---:|---:|---:|
| MMLU 0-shot, 1,425 q (±1.0) | 81.0 (80.4 / 81.6) | 80.5 (80.8 / 80.3) | 81.4 (81.4 / 81.5) | 80.6 | 81.5 |
| TruthfulQA-MC2, 200 q (±3.1) | 53.2 (53.4 / 53.0) | 49.8 (50.1 / 49.4) | 52.9 (53.4 / 52.4) | 49.8 | 52.6 |
| GSM8K 5-shot chat, 300 q, strict (±1.7) | 90.7 | 91.3 | 89.7 (90.0 / 89.3) | 92.0 | 90.7 |
| GSM8K flexible-extract | 92.0 | 92.3 | 90.8 (91.3 / 90.3) | 93.3 | 91.7 |
| IFEval prompt-level strict, 200 (±2.8) | 81.5 | 79.0 | 81.0 (80.5 / 81.5) | 82.0 | 80.5 |
| IFEval instruction-level loose | 89.0 | 88.7 | 89.8 (89.6 / 89.9) | 90.6 | 89.6 |

Cells with two values are replicate means. Every FP8-versus-INT8 difference,
base or adapter, is inside the replicate spread or one standard error. These
limited subsets found no clear FP8-versus-INT8 regression; they do not establish
statistical equivalence, lossless quantization or general Agent reliability.
The suite did not resolve an adapter-versus-base difference on MMLU, GSM8K
or IFEval on either target. TruthfulQA-MC2 shows a repeated downward signal:
the adapter scores about 3.4 points below base on FP8 (both replicates) and
3.1 points below on INT8 with adapter v2, while the FP8-derived adapter on INT8
shows only 0.3. This remains a signal at the edge of this subset's resolution; the full
817 questions and uncertainty analysis are needed before asserting a reliable
capability cost or its cause.

One serving observation came out of the harness. On this profile vLLM 0.28.0
returns HTTP 400 "Out of range float values are not JSON compliant: nan" for
`/v1/completions` with `echo` and `logprobs` whenever the prompt is exactly
241-255 tokens long; 240 and 256 or more are fine and content is irrelevant.
Generated text and sampled-token logprobs at those lengths are normal, so the
app is unaffected. The eval proxy pads such prompts with leading newlines,
identically for every arm, and reports the count: 91-94 per FP8 stage-1 pass
versus 53 on INT8, so the band is narrower on the INT8 kernels. This looks
like a prompt-logprobs bug in the last CUDA-graph bucket with MTP and has not
been checked with graphs disabled.

Per-task result files with stderr, subject-group breakdowns, replicates and
the superseded runs are in `a100-capability-eval-four-arm-2026-09-08.json`.

## Production switch to INT8

`scripts/switch-a100-production.sh` moved the A100 service to the INT8 target
on 2026-09-08. It rewrites four keys in the production env (model, revision,
adapter dir, `DTYPE=bfloat16`), stops and renames the FP8 container instead of
deleting it, launches the new service under the same name, port and restart
policy, and gates it on the live process: kernel line, API smoke, canaries,
image protocols and guardrails, prefix-cache isolation, 64 at C32 for both
aliases, then a one-repetition 1K and 16K replay. Any failure rolls back
automatically: stop the new container, restore the env backup, rename the old
container back and start it.

The first attempt (12:30 UTC, adapter v1) rolled back at the API smoke as
described above; FP8 was serving again at 12:36. The second attempt (13:34 UTC,
adapter v2) passed every gate and completed at 13:38. Replay on the live
service, one repetition:

| Alias | 1K TTFT | 1K decode | 16K TTFT | 16K decode |
|---|---:|---:|---:|---:|
| qwen3.8-27b | 0.311 s | 145.8 tok/s | 3.526 s | 119.9 tok/s |
| qwen3.8-27b-uncensored | 0.354 s | 133.6 tok/s | 3.933 s | 135.8 tok/s |

These match the Phase 1 arm (3.50 s 16K TTFT, 148.7 tok/s short decode).
Production now serves the INT8 checkpoint at revision `2df4e3b0` with adapter
v2. The previous FP8 container was retained with restart disabled, alongside
a private environment backup for rollback. Exact container names and backup
locations belong in private deployment records. See
`a100-production-switch-int8-2026-09-08.json`.

The StrongREJECT Small suite was then run against the live INT8 adapter alias
(60 prompts, temperature 0, 2,048 max tokens, C4): 60/60 non-empty responses,
no request errors. The pinned refusal classifier labelled 0 strict refusals,
6 disclaimers with an answer, and 54 normal answers, identical to the FP8
adapter's 2026-09-04 result. The literal marker matcher flagged 12/60 (FP8: 9)
and is not used as the binding classifier for the same reason as before. Raw
responses stay on the test node; the aggregate is
`a100-int8-adapter-v2-strongreject-classifier-2026-09-08.json`. A 24-hour
soak (short chat, forced tool call, periodic image OCR and 8-way bursts on both
aliases every 5 minutes, with GPU memory and restart-count tracking via
`scripts/soak-a100.sh`) started at 13:47 UTC and is reported separately when
it completes. No completed soak artifact is archived in this repository as of
2026-09-09; this document does not claim its current running state.
