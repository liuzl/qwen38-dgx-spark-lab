# Qwen3.8 Local Inference Lab

Reproducible serving recipes and measured experiments for Qwen3.8-27B on an
NVIDIA DGX Spark (GB10, `sm_121`), NVIDIA A100, and Apple Silicon. The Spark
remains the reference serving track; the A100 track starts from a conservative
FP8/W8A16 vLLM baseline, while the M3 Max track provides a controlled local-inference
comparison using oMLX and Lightning MTP.

The current reference stack serves two model IDs from one vLLM process:

```text
qwen3.8-27b             clean mixed-NVFP4 base
qwen3.8-27b-uncensored  native per-request rank-1 LoRA
```

The LoRA is derived from output-space refusal directions. Requests that do not
select the adapter remain on vLLM's clean CUDA graph, avoiding the performance
regression caused by an always-on runtime projection hook.

> [!WARNING]
> This is an independent community experiment, not an official Qwen, NVIDIA,
> Apple, MLX, oMLX, vLLM, SGLang, RadixArk, Jundot, or Z Lab project. It
> includes serving code and aggregate benchmark results only—no model weights,
> draft weights, direction tensors, credentials, or raw safety-evaluation
> responses.

## Cross-platform result

One client-owned prompt corpus was sent to both qualified stacks with the same
temperature, output length, warmup shape, concurrency, and unique request
prefixes:

| Case | Metric | DGX Spark | M3 Max 64 GB |
|---|---|---:|---:|
| PP1080/TG256 C1 | decode tok/s | **60.38** | 44.24 |
| PP1080/TG256 C4 | aggregate tok/s | **143.70** | 18.19 |
| PP16345/TG256 C1 | decode tok/s | **69.83** | 14.65 |

The M3 Max reaches 73% of Spark's short single-stream decode, but sustained long
context and concurrency remain Spark strengths. Separately, oMLX prefix caching
reduced a repeated 5.2K-prefix turn from 25.19 to 6.41 seconds on the M3 Max;
that is a prefill/TTFT gain, not a decode-speed multiplier.

The A100 qualification subsequently found that the initial compatibility arm
(FP8 W8A16 Marlin, eager, autoregressive) was misleadingly slow at 11.90 tok/s.
Enabling the checkpoint's native MTP at depth 3 and vLLM CUDA Graphs raised the
same short C1 decode to **124.48 tok/s** and C4 aggregate throughput to **251.67
tok/s**, with 92.37% draft-token acceptance, 4/4 deterministic canaries, and
64/64 stability requests. A same-protocol BF16 arm was slower at decode but
faster at long prefill. See [NVIDIA A100 qualification](docs/a100.md).

The Spark refusal-direction method was also ported to the official A100 FP8
checkpoint. One process now qualifies the portable IDs `qwen3.8-27b` and
`qwen3.8-27b-uncensored`; both passed all three APIs, forced tools, cache
isolation, deterministic canaries, and 64-request stability. The adapter's
StrongREJECT Small classifier result was 0/60 strict refusals, 6 disclaimers
with an answer, and 54 normal answers. Adapter and direction artifacts remain
private and are not distributed by this repository.

The A100 deployment now also qualifies image inputs for both aliases through
Chat Completions, Responses, Anthropic Messages, and image-plus-tool calls.
Remote HTTP media fetching and video remain disabled; public clients use data
URLs or protocol-native base64. The same 24 GiB KV pool and 128K text limit are
retained, with image tokens sharing the request context budget.

See [DGX Spark vs Apple M3 Max](docs/cross-platform-comparison.md) for the
protocol, raw artifacts, interpretation, and limits. Apple setup and the ANE
memory boundary are in the [Apple Silicon track](docs/apple-silicon.md).
The completed controlled-research [NVIDIA A100 qualification](docs/a100.md)
records its verified environment, success gates, minimum port, tuning results,
and native-LoRA qualification.
The concise decision record is available in the
[Qwen3.8-27B A100 porting report](docs/a100-porting-report.md).

## DGX Spark measured result

Hardware: one DGX Spark GB10 with 128 GB unified memory. Target:
`RadixArk/Qwen3.8-27B-NVFP4`; drafter: `z-lab/Qwen3.8-27B-DFlash2`;
vLLM `f94666b60`; DFlash2 probabilistic K7; thinking off; FP8 KV.

| Arm | C1 output tok/s | C1 acceptance | C8 output tok/s | C8 acceptance |
|---|---:|---:|---:|---:|
| Clean vLLM baseline | 45.17 | 5.15 | 95.77 | ~2.6 |
| Native-LoRA server, base request | **45.41** | **5.13** | **93.16** | **2.69** |
| Native-LoRA server, adapter request | **31.92** | **3.74** | **100.52** | **3.05** |
| Previous always-on projection hook, base | 30.32 | 3.46 | — | — |

The native LoRA restores the clean base fast path. Adapter C1 remains slower
because the target is modified while the DFlash2 drafter is not, reducing draft
acceptance.

Additional gates:

- OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages passed.
- Forced tool calls passed on all three APIs.
- Prefix cache was isolated by native LoRA ID: repeated base and adapter
  requests each hit 3,296 tokens; the first cross-arm request hit zero.
- StrongREJECT Small classifier: 0/60 strict refusals, 8 disclaimers with an
  answer, 52 normal answers, zero empty responses or request errors.
- With a 16 GiB FP8 KV cache, the Qwen vLLM process used about 46 GiB; the LoRA
  file itself was 8.3 MiB.
- A controlled three-repeat head A/B found that the dense BF16 `lm_head` is not
  a throughput upgrade for this vLLM + DFlash2 stack: base C1 fell from 53.277
  to 22.854 tok/s and base C8 from 97.744 to 88.520 tok/s. The C1 regression
  tracks DFlash2 acceptance length falling from 6.092 to 2.972.

See [Benchmark methodology](docs/benchmarks.md) for the exact workload and
interpretation.

## Repository layout

```text
docker/                  pinned vLLM overlay for DFlash2 fixes
scripts/                 serving, conversion, validation and neutral benchmarks
configs/                 DGX Spark and Apple Silicon environment templates
docs/                    per-platform architecture, comparison and licenses
benchmarks/results/      aggregate, sanitized machine-readable results
panel/                    read-only single-Spark vLLM telemetry dashboard
```

## DGX Spark quick start

### 1. Prepare dependencies

Obtain these files separately and review their licenses:

- `RadixArk/Qwen3.8-27B-NVFP4`
- `z-lab/Qwen3.8-27B-DFlash2`
- a compatible `refusal_dirs_qwen38.safetensors`

The direction file must contain one vector per edited module, `__coefs__`, and
the `coef_order` metadata used by the converter. This repository intentionally
does not redistribute that artifact.

### 2. Build the pinned image

```bash
docker build -t qwen38-vllm-dflash2:lab docker/
```

The default base image is pinned by digest. Override `BASE` deliberately if you
are qualifying another vLLM revision:

```bash
docker build \
  --build-arg BASE=vllm/vllm-openai:nightly-aarch64 \
  -t qwen38-vllm-dflash2:lab docker/
```

### 3. Convert directions to native LoRA

The converter needs PyTorch and safetensors. Running it inside the image keeps
float8 support consistent with the serving environment:

```bash
mkdir -p artifacts/adapter
docker run --rm --entrypoint python3 \
  -v "$PWD/scripts/qwen38-rank1-to-lora.py:/tool.py:ro" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$DIRECTIONS_FILE:/directions.safetensors:ro" \
  -v "$PWD/artifacts:/output" \
  qwen38-vllm-dflash2:lab \
  /tool.py --model /model --directions /directions.safetensors \
  --output /output/adapter
```

### 4. Launch

```bash
cp configs/qwen38-spark.env.example .env
$EDITOR .env
set -a; source .env; set +a
scripts/serve-native-lora.sh
```

Defaults are a 131,072-token context and 16 GiB FP8 KV cache. Start with less
KV if the Spark also hosts other GPU services.

### 5. Validate

```bash
python3 scripts/validate-apis.py \
  --base-url http://127.0.0.1:18102 \
  --base-model qwen3.8-27b \
  --adapter-model qwen3.8-27b-uncensored

python3 scripts/validate-cache-isolation.py \
  --base-url http://127.0.0.1:18102 \
  --base-model qwen3.8-27b \
  --adapter-model qwen3.8-27b-uncensored

scripts/benchmark.sh
```

For the M3 Max/oMLX path, use the
[Apple Silicon runbook](docs/apple-silicon.md). To compare any two
OpenAI-compatible endpoints with identical prompts, use:

```bash
python3 scripts/benchmark-openai.py \
  --base-url "$BASE_URL" --model "$MODEL_NAME" \
  --label "$PLATFORM_LABEL" --output benchmarks/results/platform.json
```

The optional [Spark LLM Panel](panel/README.md) shows live vLLM and DFlash2
telemetry without Docker access or host-monitoring duplication. Hardware
history and alerts remain in Beszel; benchmarks remain command-line only. See
the [monitoring architecture](docs/monitoring.md) for the boundary and alert
policy. Its `/apps` view is the private service directory for VoxStudio,
Beszel, Qwen, LLM telemetry, and the DGX Dashboard.

## Further reading

For a platform-neutral deployment decision framework, see
[Choosing a Qwen3.8-27B Local Inference Stack](docs/qwen38-local-inference-guide.md).
For the A100 measurements, FP8/BF16 decision, performance root cause, and
native uncensored-LoRA port, see the
[Qwen3.8-27B A100 porting report](docs/a100-porting-report.md).

## Status

`v0.2` is an experimental three-platform reference. Before treating any stack
as a production service, run a workload-specific soak test and validate every
new runtime, checkpoint, driver, CUDA, MLX, or operating-system revision.

## License and attribution

Repository-authored code is Apache-2.0. Third-party projects and model artifacts
retain their own licenses. See [NOTICE](NOTICE) and
[Model and artifact licenses](docs/licenses.md).
