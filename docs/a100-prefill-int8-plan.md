# A100 prefill optimization plan: INT8 W8A8

Status: plan, 2026-09-08. No INT8 measurement exists yet. Nothing here is a
result until the controlled arms in this document have run.

## Problem

The current A100 service (official `Qwen/Qwen3.8-27B-FP8`, vLLM 0.28.0,
CUDA Graph, MTP K7, 128K context) is decode-tuned but prefill-limited. The
[2026-09-06 tuning](a100-performance-tuning.md) left 16K-input TTFT at about
6.5 / 7.0 seconds (base / adapter) across every speculative depth, and the
8K versus 16K batch-budget A/B changed nothing. Scheduler parameters are
exhausted; the remaining cost is in the linear kernels.

The startup log names the cause. A100 (`sm_80`) has BF16 and INT8 Tensor
Cores but no FP8 execution, so vLLM loads the FP8 checkpoint through
`MarlinFP8ScaledMMLinearKernel` and warns:

```text
Your GPU does not have native support for FP8 computation but FP8 quantization
is being used. Weight-only FP8 compression will be used leveraging the Marlin
kernel. This may degrade performance for compute-heavy workloads.
```

This is W8A16: weights are dequantized to BF16 and multiplied at BF16 rate,
plus the dequantization cost. Decode is bandwidth-bound and benefits from the
smaller weights; prefill is compute-bound and pays the penalty. The
[porting report](a100-porting-report.md) measured it directly under identical
Graph + MTP K3 settings:

| 16K C1 | FP8 (Marlin W8A16) | BF16 |
|---|---:|---:|
| TTFT | 6.375 s | 4.559 s |
| Approximate prefill rate | ~2,570 tok/s | ~3,600 tok/s |
| Model memory | 28.49 GiB | 51.02 GiB |

BF16 is not an acceptable fix: it removes about 22 GiB of KV capacity and
loses 38-48% of decode. The 128K admission limit and C32 scheduler depend on
that KV pool.

## Lever

INT8 W8A8 is the only path that raises the prefill compute ceiling on
Ampere. INT8 Tensor Cores run at twice the BF16 dense rate, and vLLM 0.28.0
routes compressed-tensors W8A8 modules to `CutlassInt8ScaledMM` on `sm_80`.
The production container already exposes `_custom_ops.cutlass_scaled_mm`.
Weights stay 8-bit, so KV capacity is unchanged relative to FP8.

The trade is known in advance: Marlin is the better small-batch kernel, so
single-stream decode is expected to fall by roughly 10-15%. This plan accepts
that trade only if TTFT improves materially; the decision rule is below.

## Candidate checkpoint

`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`, revision
`2df4e3b00d4b865d59a7de0dc286fb18fd455a1e` (v3, published 2026-09-03,
29.1 GiB, compressed-tensors, `Qwen3_5ForConditionalGeneration`).

Published properties, to be verified locally rather than trusted:

- Mixed precision: 288 of 400 Linear modules are W8A8 (INT8 per-channel
  weights, INT8 per-token dynamic activations: `q/k/v/o_proj`,
  `gate/up_proj`, `linear_attn.in_proj_qkv/in_proj_z`). 112 modules stay
  W8A16 on Marlin (`mlp.down_proj`, `linear_attn.out_proj`). `embed_tokens`,
  `lm_head`, `mtp.*`, vision tower and GDN gates are BF16.
- Author's teacher-forced full-vocab KLD versus BF16 is 0.00556, lower than
  the official FP8 checkpoint's 0.00584 on the same harness.
- Author's prefill throughput is 1.42x a W8A16 checkpoint of the same model
  on 2x RTX 3090 TP2; single-stream decode is about 13% slower.
- Ships a BF16 MTP head, so the current `{"method":"mtp"}` path applies.
- The author cites vllm#37035 and recommends MTP K<=3 on hybrid-GDN targets.
  The production FP8 service runs K7 correctly, so this is treated as an
  acceptance-rate hypothesis to measure, not a correctness limit.

Rejected alternative: `lued/Qwen3.8-27B-INT8-W8A16-MTP` is INT8 weight-only.
It still runs through Marlin and cannot change prefill compute.

If 1.42x holds, 16K TTFT falls from about 6.4 s to about 4.5 s, matching
BF16 while keeping FP8-class memory. That is the expectation, not a claim.

## Constraints carried from the current service

- Everything else stays pinned: vLLM 0.28.0 image, CUDA Graph
  `FULL_AND_PIECEWISE`, 128K max model len, 24 GiB BF16 KV, C32, 16K batch
  budget, thinking disabled, same image limits and media-domain block.
- A second 27B model does not fit beside production. Every INT8 arm needs a
  maintenance window; the production container is stopped, not deleted, and
  restored on exit, exactly as `scripts/sweep-a100-mtp-arms.sh` does.
- The rank-1 uncensored adapter was derived against the FP8 revision
  `017b9c7a` and is not transferable. The INT8 target needs its own
  derivation by the method in the porting report before the adapter alias can
  be measured or promoted. Phase 1 therefore measures the base alias only.
- Keep `--dtype bfloat16`; the author reports fp16 breaks speculative
  acceptance on this hybrid target.
- Kernel selection is a gate. The startup log must show `CutlassInt8ScaledMM`
  for the W8A8 groups and Marlin only for the W8A16 groups. Any other routing
  is a different arm and must not inherit the INT8 label.

## Phase 0: preparation (no maintenance window)

1. Download the pinned revision into the shared HF cache
   (`/hf-cache/hub`) via a throwaway CPU-only container; the production
   container is intentionally offline. Started 2026-09-08.
2. Verify the snapshot: 49 files, `config.json` `quantization_config`
   matches the description above, `model-mtp.safetensors` present.
3. Add an INT8 arm definition to `scripts/sweep-a100-mtp-arms.sh` that
   swaps only the model path and revision, drops `--lora-modules`, and keeps
   every other production flag.

## Phase 1: base-alias performance arms (maintenance window)

Protocol identical to the 2026-09-06 and 2026-09-07 sweeps: fresh process per
arm, eight-token warmup per shape, distinct prefixes per repetition, exactly
256 output tokens, three repetitions, streaming usage only.

| Arm | Purpose |
|---|---|
| FP8 static K7 (production, control) | Reproduce 2026-09-06 within 2% before trusting the session |
| INT8 static K7 | Direct comparison; isolates the kernel change |
| INT8 static K3 | Tests the author's K<=3 acceptance advice |

Shapes: 1K/C1, 16K/C1, 1K/C8, 1K+image/C1, plus 1K/C32 to check that the
INT8 batched path does not regress the C32 aggregate. Record 16K TTFT,
decode rate, aggregate rate, acceptance including warmups, graph memory and
available KV blocks per arm.

Estimated window: about 1.5 hours including model load and graph capture for
three arms. Stop early per the decision rule.

## Decision rule

- Promote to Phase 2 only if INT8 16K C1 TTFT is below 5.0 s (at least a
  22% improvement) and the C8/C32 aggregate is not worse than FP8 K7.
- Accept up to 15% lower 1K C1 decode as the price of that TTFT. A larger
  decode loss needs an explicit product decision, not a default.
- If TTFT does not clear 5.0 s, stop. Do not derive an adapter, and record
  the negative result in `a100-performance-tuning.md`.

## Phase 2: quality and adapter (only after Phase 1 passes)

1. Four semantic canaries per alias, forced-tool checks, the three API
   surfaces, and the six model/protocol image combinations, exactly as in
   the K7 deployment validation.
2. Re-derive the rank-1 uncensored LoRA against the INT8 target, then repeat
   the adapter workloads and the base/adapter prefix-cache isolation test.
3. 64/64 requests at C32 per alias, then a production replay.
4. Only then swap the systemd/container definition and write the result into
   `a100-performance-tuning.md` and the README measured-result table.

## Out of scope

- FP8 KV cache: FlashAttention 2 on Ampere cannot read quantized KV, so it
  forces a FlashInfer backend switch and is a separate arm.
- DFlash2 drafting on A100, a vLLM upgrade, or dynamic MTP depth
  (blocked on vLLM 0.28.0 per the 2026-09-07 result).
- Self-quantizing with llm-compressor. Only justified if the community
  checkpoint fails the Phase 1 kernel gate or Phase 2 quality gates.
