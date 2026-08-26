# Apple Silicon track

This track qualifies Qwen3.8-27B on an Apple M3 Max with 64 GB unified memory.
It complements the DGX Spark reference stack; it does not reuse CUDA images,
NVFP4 kernels, FP8 KV cache, or vLLM patches.

## Qualified candidate

The measured candidate used:

- oMLX `0.6.3rc3`, MLX `0.32.0`, mlx-lm `0.31.3`;
- `Jundot/Qwen3.8-27B-oQ4e-mtp`;
- Lightning MTP enabled;
- TurboQuant KV and Qwen ANE prefill disabled;
- oMLX prefix cache enabled only for the dedicated cache experiment.

The checkpoint and runtime are obtained separately. Pin their revisions and
verify hashes before comparing results because model repositories and release
assets can change independently of this repository.

## Isolated launch

Create a dedicated oMLX environment and copy
[`configs/qwen38-apple-silicon.env.example`](../configs/qwen38-apple-silicon.env.example)
to a local, untracked file. A representative launch is:

```bash
omlx serve \
  --model-dir "$MODEL_DIR" \
  --base-path "$BASE_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-concurrent-requests 4 \
  --memory-guard safe \
  --no-cache \
  --api-key "$API_KEY"
```

For agent sessions with a stable system prefix, replace `--no-cache` with a
bounded local cache:

```bash
omlx serve \
  --model-dir "$MODEL_DIR" \
  --base-path "$BASE_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-concurrent-requests 1 \
  --memory-guard safe \
  --paged-ssd-cache-dir "$BASE_PATH/cache" \
  --paged-ssd-cache-max-size "$OMLX_SSD_CACHE_MAX_SIZE" \
  --hot-cache-max-size "$OMLX_HOT_CACHE_MAX_SIZE" \
  --api-key "$API_KEY"
```

Set per-model acceleration through oMLX and verify the resolved load log, not
the settings file alone:

```text
mtp_enabled=true
turboquant_kv_enabled=false
qwen35_ane_prefill_enabled=false
```

## Same-version qualification

A same-model throughput run compared oMLX 0.6.2 with 0.6.3rc3 using its local
`code_python` benchmark, TG256, Lightning MTP, and no cache/ANE/TurboQuant:

| Context | Metric | 0.6.2 | 0.6.3rc3 | Delta |
|---:|---|---:|---:|---:|
| 1K | decode tok/s | 37.6 | 42.9 | +14.1% |
| 4K | decode tok/s | 28.5 | 29.6 | +3.9% |
| 16K | decode tok/s | 31.0 | 28.6 | -7.7% |
| 16K | prefill tok/s | 174.5 | 183.4 | +5.1% |
| 16K | end-to-end seconds | 102.156 | 98.323 | -3.8% |

Both 16K runs started at thermal level 0 and reached 2. This is a matched
sustained comparison, not a cold-machine peak.

## Prefix cache result

With a 10 GB SSD cache and 4 GB hot cache, oMLX selected a 4,096-token block
for this hybrid model. Two requests shared the same code system prompt and
changed only the final short instruction:

| Request | Prompt tokens | Cached tokens | Wall time |
|---|---:|---:|---:|
| cold | 5,226 | 0 | 25.191 s |
| next turn | 5,227 | 4,096 | 6.408 s |

The next turn was 3.93x faster end to end. It was not sub-second because the
remaining 1,131 tokens still required prefill. Cache claims must report both
the cached and uncached token counts.

## Why ANE stays off on 64 GB

The tested ANE configuration compiled 64 MLP and 48 GDN procedures. It raised
the measured process peak from about 21.6 GB to 43.2 GB for a 4K request while
improving prefill by only 3.2%. At 16K it triggered adaptive memory throttling,
released 22.69 GB of ANE banks, and fell back toward GPU execution:

| Context | Metric | GPU only | ANE | Delta |
|---:|---|---:|---:|---:|
| 4K | prefill tok/s | 209.6 | 216.3 | +3.2% |
| 4K | process peak | 21.63 GB | 43.20 GB | +21.57 GB |
| 16K | prefill tok/s | 183.4 | 123.9 | -32.4% |
| 16K | end-to-end seconds | 98.323 | 143.244 | +45.7% |

Positive ANE results from 128 GB Macs must not be extrapolated to this 64 GB
configuration.

## Reproduce the platform-neutral run

The same client corpus can be sent to any OpenAI-compatible endpoint:

```bash
OPENAI_API_KEY="$API_KEY" python3 scripts/benchmark-openai.py \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL_NAME" \
  --label apple-m3-max-64gb-omlx-mtp \
  --output benchmarks/results/my-m3-max.json
```

The harness stores token counts, TTFT, latency, finish reason, and throughput;
it does not store response text, the endpoint URL, or credentials.

## Operational boundary

- Keep experimental serving loopback-only unless a separate access review is
  completed.
- Isolate runtime state and restore any service stopped to free unified memory.
- Check for generated helper symlinks after an oMLX experiment.
- A release candidate is not a production recommendation; repeat the small
  qualification when oMLX 0.6.3 final ships.
- Run agent/tool correctness and cache-reuse canaries before adding a permanent
  selector.

See [Cross-platform comparison](cross-platform-comparison.md) for the matched
DGX Spark result.
