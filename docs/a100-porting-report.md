# Qwen3.8-27B on A100: Porting, Tuning, and Native LoRA

This report summarizes the 2026-09-03 to 2026-09-04 qualification of
Qwen3.8-27B on one NVIDIA A100 80GB PCIe. It answers four deployment questions:
whether BF16 is too large, which precision to download first, why the initial
A100 result was slower than DGX Spark, and whether the Spark uncensored method
could be ported without exposing hardware-specific model names.

For commands, environment constraints, upstream references, and the full test
sequence, see the [A100 qualification runbook](a100.md).

> [!WARNING]
> This is a controlled research qualification, not a production deployment.
> The repository does not distribute model weights, refusal-direction tensors,
> LoRA artifacts, credentials, internal topology, or raw safety responses.

## Decision summary

- **The A100 is suitable.** Both official FP8 and BF16 checkpoints fit on one
  80GB card, including native MTP and CUDA Graph execution.
- **Download FP8 first and retain BF16 as a reference.** FP8 provides faster
  decode, higher concurrent throughput, and substantially more KV capacity.
  BF16 is useful for prefill-dominated, long-input/short-output workloads.
- **Do not prioritize lower-bit community quantizations by default.** FP8
  already leaves about 39GiB for KV cache. Lower bit width would add conversion,
  kernel, and quality variables without solving the current bottleneck.
- **The original 11.90 tok/s result was a software-path baseline, not the A100
  limit.** Native MTP K3 plus vLLM CUDA Graph raised short C1 decode to
  124.48 tok/s.
- **The Spark uncensored method was ported successfully.** A native rank-1 LoRA
  regenerated against the official FP8 base passed API, tool, cache-isolation,
  deterministic-canary, stability, and refusal-behavior gates.
- **The public model IDs remain hardware-neutral:** `qwen3.8-27b` and
  `qwen3.8-27b-uncensored`.

## Scope and evidence boundary

The run qualified text-only serving at a 32K admission limit. It covered the
official FP8 and BF16 checkpoints, eager and CUDA Graph execution, built-in MTP,
native LoRA, three compatible APIs, forced tools, deterministic canaries,
prefix-cache isolation, and bounded stability tests.

It did not qualify vision, the native 262K context limit, FP8 KV cache, multi-GPU
serving, a 24-hour soak, or a third-party full uncensored checkpoint. DGX Spark
comparisons are complete-stack comparisons because the checkpoints,
quantization formats, speculative decoders, and runtimes differ.

Local measurements are the binding evidence. Upstream documentation and issues
set the initial test order. X posts were used only to identify variables such as
power limits, PCIe width, concurrency, and speculative acceptance; their
throughput claims were excluded from local comparisons.

## Fixed environment

| Component | Qualified value |
|---|---|
| GPU | 1 x NVIDIA A100 80GB PCIe, compute capability 8.0 |
| Power / PCIe | 300W configured and maximum; PCIe Gen4 x16 |
| Driver / OS | 550.54.14; Ubuntu 22.04.4 LTS |
| Container | vLLM 0.28.0, PyTorch 2.13.0+cu129 |
| Image digest | `sha256:a77ed1c057e0458dbed205dea0ecaacd0ca2405721be6e58182bf9ee42c359f6` |
| Official FP8 revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Official BF16 revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

The newer Docker client required `DOCKER_API_VERSION=1.43` for the older daemon.
The CUDA 12.9 image was selected instead of the CUDA 13.0 `latest` image to avoid
adding a forward-compatibility variable to the R550 driver.

Two pre-existing idle compute processes used about 4.27GiB during qualification.
They were not stopped, so the results are explicitly co-resident rather than a
perfectly clean exclusive-card baseline.

## Precision decision

The official FP8 checkpoint uses E4M3 data with dynamic activation scaling and
128 x 128 weight blocks. On Ampere, vLLM selected
`MarlinFP8ScaledMMLinearKernel`: this is **W8A16**, not native FP8 Tensor Core
execution. The result should therefore be described as FP8 weight storage and
bandwidth reduction with 16-bit activations.

Under identical CUDA Graph + MTP K3 settings, FP8 and BF16 had complementary
strengths:

| Case | FP8 | BF16 | Better arm |
|---|---:|---:|---|
| PP1K C1 decode | **124.48** | 83.93 | FP8 +48% |
| PP1K C1 TTFT | 0.437s | **0.354s** | BF16 -19% |
| PP1K C4 aggregate | **251.67** | 230.39 | FP8 +9% |
| PP1K C4 TTFT | 1.703s | **1.078s** | BF16 -37% |
| PP16K C1 decode | **91.05** | 65.85 | FP8 +38% |
| PP16K C1 TTFT | 6.375s | **4.559s** | BF16 -28% |
| PP16K C1 end-to-end | 9.176s | **8.431s** | BF16 -8% |

| Capacity | FP8 Graph K3 | BF16 Graph K3 |
|---|---:|---:|
| Model loading | 28.49GiB | 51.02GiB |
| KV cache | 39.21GiB | 16.64GiB |
| KV tokens | 441,782 | 187,245 |
| Theoretical 32K concurrency | 13.48 | 5.71 |

All four 512-token deterministic canaries were byte-identical between FP8 and
BF16. This small gate does not establish broad quality equivalence, but it found
no reason to make BF16 the default. See the
[performance comparison](../benchmarks/results/a100-fp8-vs-bf16-graph-mtp-k3-2026-09-04.json)
and [canary comparison](../benchmarks/results/a100-fp8-vs-bf16-graph-mtp-k3-canaries-2026-09-04.json).

## Why the first A100 result was slow

The initial compatibility arm deliberately used eager autoregressive decoding
because vLLM 0.27.1 had an open Ampere CUDA Graph startup report. It loaded
correctly but produced only 11.90 tok/s on PP1K C1.

Changing one execution decision at a time isolated the bottleneck:

| Arm | MTP acceptance | PP1K C1 decode | PP1K C4 aggregate | PP16K C1 decode |
|---|---:|---:|---:|---:|
| Eager AR | - | 11.90 | 40.99 | 11.56 |
| Eager MTP K1 | 97.85% | 20.84 | 71.98 | 19.62 |
| Eager MTP K2 | 95.05% | 28.98 | 96.15 | 29.25 |
| Eager MTP K3 | 92.63% | 33.10 | 111.06 | 36.25 |
| CUDA Graph MTP K3 | 92.37% | **124.48** | **251.67** | **91.05** |

vLLM 0.28.0 successfully captured `FULL_AND_PIECEWISE` graphs on this card.
Compilation added 79.8 seconds to startup and reduced the KV pool modestly, but
Graph K3 was 3.76x eager K3 for short C1 decode and 2.27x for C4 aggregate.
Canaries passed 4/4 and the 64-request stability run had no failures, restarts,
or OOM.

The evidence rules out memory pressure, an underconfigured power limit, a narrow
PCIe link, and failed requests as the primary explanation. The remaining
evidence supports eager kernel-launch overhead plus one-token autoregressive
steps as the dominant issue; MTP reduces target steps and CUDA Graph reduces
per-iteration launch overhead. This is an evidence-backed interpretation, not a
hardware-counter proof.

Artifacts: [eager baseline](../benchmarks/results/a100-80gb-vllm-fp8-eager-ar-2026-09-04.json),
[MTP sweep](../benchmarks/results/a100-80gb-vllm-fp8-mtp-sweep-2026-09-04.json),
and [Graph K3 benchmark](../benchmarks/results/a100-fp8-graph-mtp-k3-2026-09-04-openai.json).

## A100 versus DGX Spark

| Case | DGX Spark | A100 Graph MTP K3 | A100 delta |
|---|---:|---:|---:|
| PP1K C1 decode | 60.38 | 124.48 | +106% |
| PP1K C4 aggregate | 143.70 | 251.67 | +75% |
| PP16K C1 decode | 69.83 | 91.05 | +30% |
| PP16K C1 TTFT | **1.82s** | 6.38s | slower |

The tuned A100 stack wins decode and measured concurrent throughput, while the
Spark stack retains substantially faster 16K prefill. The original perception
that the A100 was slower came from comparing a conservative eager AR arm with an
already tuned Spark NVFP4 + DFlash2 stack. See the
[structured comparison](../benchmarks/results/dgx-spark-vs-a100-fp8-graph-mtp-k3-2026-09-04.json).

Increasing `max_num_batched_tokens` from 8192 to 16384 did not improve 16K TTFT
or decode. It improved C4 TTFT but slightly reduced C4 decode, leaving aggregate
throughput nearly unchanged. The qualified default therefore remains 8192. See
the [controlled A/B](../benchmarks/results/a100-fp8-graph-mtp-k3-bt8k-vs-bt16k-2026-09-04.json).

## Native uncensored LoRA port

The Spark adapter could not simply be copied because it was derived against an
NVFP4 effective base. The reusable inputs were the refusal directions and the
conversion method. The converter was extended to dequantize two-dimensional
128 x 128 `weight_scale_inv` blocks and to discover index-free layer shards.
The adapter was then regenerated against the pinned official FP8 revision.

| Conversion property | Qualified value |
|---|---|
| Direction SHA-256 | `9de12cbe71f38baf2f6b4a21dfcb2b13bd6416ab4785214afce27c7543f05c1d` |
| Adapter SHA-256 | `d616765383d7f3957677ef6664223fc6ec7e0895550d14405bc110c40d1d2f40` |
| Structure | 128 modules, 256 finite tensors, rank 1 |
| Quantized sources | 128 `fp8_block_128x128` modules |
| Distribution | Private; neither directions nor adapter is in Git |

The service exposes only these portable names:

```text
qwen3.8-27b
qwen3.8-27b-uncensored
```

Both aliases passed exact Chat Completions, Responses, Anthropic Messages,
forced-tool name and arguments, 4/4 deterministic canaries, and 64/64 stability
requests. Prefix-cache hit counts were `[0, 2400, 0, 2400]` for base, repeated
base, adapter after base, and repeated adapter, confirming cross-alias isolation.

The adapter overhead was bounded:

| Case | Base | Uncensored | Delta |
|---|---:|---:|---:|
| PP1K C1 decode | 124.53 | 119.16 | -4.31% |
| PP1K C4 aggregate | 264.77 | 233.55 | -11.79% |
| PP16K C1 decode | 92.49 | 86.18 | -6.82% |

The StrongREJECT Small run completed 60/60 non-empty responses with no request
errors. A pinned refusal classifier labeled 0 strict refusals, 6 disclaimers
with an answer, and 54 normal answers. A literal matcher flagged 9 broad string
patterns, so it was not used as the binding classifier.

MTP acceptance was workload-sensitive: 94.75% during neutral qualification and
56.57% on the StrongREJECT workload. Speculative verification remains lossless,
but the uncensored-path speedup cannot be extrapolated from neutral prompts.

See the [qualification summary](../benchmarks/results/a100-native-lora-qualification-2026-09-04.json),
[base/adapter performance](../benchmarks/results/a100-native-lora-base-vs-adapter-2026-09-04.json),
and [classifier aggregate](../benchmarks/results/a100-native-lora-strongreject-classifier-2026-09-04.json).

## Qualified default and remaining limits

```text
Weights                 official FP8, W8A16 Marlin
Execution               FULL_AND_PIECEWISE CUDA Graph
Speculative decoding    native MTP depth 3
KV cache                auto / BF16
Max model length        32768
Max active sequences    8
Max batched tokens      8192
Base model ID           qwen3.8-27b
Adapter model ID        qwen3.8-27b-uncensored
```

The launch template is
[`configs/qwen38-a100.env.example`](../configs/qwen38-a100.env.example), and the
qualification driver is
[`scripts/run-a100-qualification.sh`](../scripts/run-a100-qualification.sh).

The deployed capacity profile subsequently fixed the BF16 KV cache at 24 GiB,
raised the admission limit to 128K, allowed 32 active sequences, and used 16,384
batched tokens. It measured 338,297 KV tokens (2.58 full 128K requests), passed
64/64 requests on both aliases at C32, and completed a 124,021-token prompt.
This profile preserves roughly 15-18 GiB of observed headroom, but its throughput
has not been benchmarked under the original protocol.

Before production use, run a workload-specific 24-hour soak, validate real Agent
task completion, and add authentication, queue limits, and service monitoring.
Requalify after any checkpoint, vLLM, driver, CUDA, or kernel change. If CUDA
Graph fails on a new revision, eager mode is the compatibility rollback; if the
adapter fails, disable its alias while preserving the clean base path.

The test services were stopped after qualification. No production routing or
public endpoint was configured. A later loopback-only deployment reused the
qualified container and portable model IDs.
