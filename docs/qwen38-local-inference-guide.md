# Choosing a Qwen3.8-27B Local Inference Stack

> Data cutoff: 2026-08-27. Runtimes, model conversions, and kernels are changing
> quickly. Exact numbers illustrate methods and decision boundaries rather than
> a permanent ranking.

The easiest local-inference question is “how many tokens per second?” The more
important question is “which complete stack works best for the real workload?”
Quantization, runtime, speculative decoding, context length, concurrency, and
tool correctness all affect that answer.

This guide uses controlled Qwen3.8-27B experiments to present a reusable
qualification method. The method matters more than the hardware-specific
headline numbers.

## Key conclusions

1. **Lower-bit quantization primarily solves capacity, not speed.** A smaller
   model file does not guarantee an efficient kernel on CUDA, Metal, or another
   target backend.
2. **Treat the runtime, model artifact, and acceleration method as one stack.**
   Comparing different converted artifacts is not a pure runtime comparison.
3. **Speculative decoding is workload-dependent.** Its useful depth changes
   with prompts, context, concurrency, quantization, and hardware.
4. **Single-stream, concurrent, and long-context workloads are separate
   problems.** A winner in one shape may lose badly in another.
5. **Agent qualification starts with task success, not tok/s.** Faster output is
   irrelevant if tool names, arguments, or multi-turn completion are wrong.

## Define the optimization target

| Target | Primary metrics | Common mistake |
|---|---|---|
| Interactive use | TTFT, single-stream decode, end-to-end time | Reporting only peak decode |
| Multi-request service | Aggregate throughput, p95, queue length | Extrapolating from C1 |
| Long documents/codebases | Prefill, long-context decode, memory headroom | Configuring a large window without testing it |
| Agents and tools | Task success, tool choice, arguments, final result | Treating parseable JSON as success |

These targets require separate tests. A single “overall speed” number hides the
actual bottleneck.

## Compare complete stacks with the same client protocol

The same client-owned prompts, temperature, output length, concurrency, and
unique request prefixes were sent to qualified 4-bit stacks on DGX Spark and an
M3 Max:

| Case | Metric | DGX Spark vLLM + DFlash2 | M3 Max native MLX |
|---|---|---:|---:|
| PP1080/TG256 C1 | decode tok/s | 60.38 | 44.24 |
| PP1080/TG256 C4 | aggregate tok/s | 143.70 | 18.19 |
| PP16345/TG256 C1 | decode tok/s | 69.83 | 14.65 |

Apple Silicon reached a substantial fraction of Spark performance for short
single-stream generation, but the gap widened under concurrency and sustained
long context. Memory bandwidth, batching, KV management, and speculative
decoding all contribute. Cross-hardware comparisons should therefore include
C1, a realistic concurrency level, and a target long-context case. See the
[cross-platform protocol](cross-platform-comparison.md).

## Lower bit width does not imply higher speed

A separate experiment ran a roughly 10.6GB community IQ2_M GGUF through
llama.cpp Metal on the M3 Max. Because this checkpoint differs from the 4-bit
mainline, it tests whether the compact GGUF route is useful rather than serving
as a pure quantization-quality A/B.

| Mode | Decode tok/s | Versus AR |
|---|---:|---:|
| AR | 11.71 | - |
| MTP depth 1 | 16.23 | +38.5% |
| MTP depth 2 | 18.28 | +56.1% |
| MTP depth 3 | 19.90 | +69.9% |

Built-in MTP improved the low-bit GGUF substantially, but the result remained
slower than the native 4-bit MLX stack on the same machine. The 2-bit route is
valuable for model size, GGUF portability, and constrained systems; it should
not become the default solely because its bit width is lower. See the
[Apple Silicon experiment](apple-silicon.md#experimental-2-bit-gguf-track) and
[structured summary](../benchmarks/results/apple-m3-max-llamacpp-iq2-summary-2026-08-26.json).

Four temperature-zero code, arithmetic, multilingual, and structured-output
canaries were identical between AR and MTP depths 1 through 3. This confirms
runtime consistency for those cases, not broad quality equivalence.

## Measure speculative decoding on the target workload

Speculative decoding drafts tokens and asks the target model to verify them.
High acceptance can produce large gains; poor acceptance adds draft and
verification work with little benefit. The best depth may change by task,
context, concurrency, quantization, and hardware.

A useful report includes:

- the autoregressive baseline and accelerated decode rate;
- drafted and accepted token counts;
- prompt type, input length, and output length;
- concurrency and per-request latency;
- temperature, seed, and sampling method;
- output consistency or task correctness before and after acceleration.

“MTP doubled performance” is not portable evidence without these details.

## Agent correctness is a binding gate

One public DGX Spark comparison reported that SGLang led vLLM in coding
throughput, while vLLM completed 18 of 18 Agent Tool Calling cases and SGLang
completed 16. The stacks used different NVFP4 artifacts, and 18 cases are not
enough for a universal ranking. The useful lesson is the decision order:

> Pass the Agent correctness gate first, then compare performance among the
> passing candidates.

A tool benchmark should verify the selected tool, schema-valid arguments,
correct values, continuation after receiving the tool result, multi-call order,
and the final task outcome. A forced-tool smoke test proves API connectivity,
not production Agent reliability.

## Five invalid benchmark patterns

### Context does not fit each serving slot

Concurrent slots may divide available context. An HTTP 200 response can still
stop before the requested output length. Always record actual prompt and
completion token counts plus the finish reason.

### Prefix cache contaminates TTFT

Repeated prompts can turn later requests into cache hits. Cache reuse is a valid
product feature, but it is not cold-prefill performance. Use unique benchmark
prefixes and report cache experiments separately.

### One model name hides different artifacts

NVFP4, GGUF, MLX, and community conversions may differ in tensor types,
calibration, or model content. When files differ, describe the result as a
complete-stack comparison rather than a pure runtime A/B.

### `model loaded` is treated as qualification

Successful loading does not prove that real requests, long context, chat
templates, tools, or concurrency work. Qualification begins with target-shaped
API requests.

### Speed is reported without validity

Record finish reason, failures, actual token counts, TTFT, end-to-end latency,
decode, aggregate throughput, and speculative acceptance. Do not promote a
speed result whose correctness or request-validity gate failed.

## Reusable qualification sequence

1. **Fix the artifact and environment.** Pin model revision/hash, runtime
   commit, driver, dependencies, quantization, KV type, context, batch settings,
   and concurrency. Isolate experiments from production services.
2. **Establish correctness.** Run deterministic text and structured canaries,
   compare autoregressive and speculative outputs, test APIs and tools, and
   check completion counts and finish reasons.
3. **Benchmark distinct shapes.** Measure short C1, target concurrency, target
   long context, cold prefill, and cache hits separately. Sweep only parameters
   that are plausibly decision-changing.
4. **Use real tasks as the final gate.** Agent and domain workloads need known
   outcomes, enough samples to reveal low-frequency errors, categorized
   failures, and performance results conditioned on stable task success.

## Deployment choices

- For personal single-stream use, start with a platform-native, quality-qualified
  4-bit or FP8 route and adjust precision only when capacity requires it.
- For constrained or cross-platform systems, test 2-bit or 3-bit GGUF while
  independently validating quality and long tasks.
- For concurrent serving, prioritize continuous batching, KV management, queue
  behavior, and p95 rather than peak C1 speed.
- For Agents, treat correctness as a hard gate and throughput as a ranking
  metric among passing candidates.
- For speculative decoding, select depth on real prompts and concurrency rather
  than copying another machine's optimum.

The correct choice is not the smallest model file or most popular runtime. It is
the complete deployment stack that reliably completes the target work on the
target hardware and concurrency profile.

The repository's [platform-neutral benchmark](../scripts/benchmark-openai.py),
[Apple Silicon track](apple-silicon.md), [A100 track](a100.md), and
[Spark benchmark methodology](benchmarks.md) provide reproducible starting
points.
