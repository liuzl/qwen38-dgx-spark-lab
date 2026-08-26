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

## Experimental 2-bit GGUF track

An additional experiment qualified the fused-MTP
`JonathanColetti/Qwen3.8-27B-Uncensored-IQ2_M.gguf` with a pinned llama.cpp
Metal build. The 10,624,771,968-byte file was pinned to repository revision
`b7ff25715ee2ae49c9ff32159bc73de864648aef` and verified with SHA-256
`28e0f88eea09438220a086c2a1e5180ad83764c748856a28fd63ce1c0fbef187`.
llama.cpp was pinned to `fc35562ba46fbbf8e30cac85edbb39642c37d248`.

The controlled single-stream sweep used Metal, Flash Attention, q8_0 KV,
18,432 tokens per slot, temperature zero, and the same PP1K/TG256 client
corpus:

| Mode | MTP depth | Decode tok/s | TTFT | Change vs AR |
|---|---:|---:|---:|---:|
| llama.cpp AR | - | 11.71 | 8.18 s | - |
| llama.cpp MTP | 1 | 16.23 | 7.73 s | +38.5% |
| llama.cpp MTP | 2 | 18.28 | 6.63 s | +56.1% |
| llama.cpp MTP | 3 | 19.90 | 6.61 s | +69.9% |
| oMLX 4-bit reference | runtime-selected | 44.24 | 5.57 s | +277.7% vs AR |

All four benign greedy canaries (code, Chinese arithmetic, multilingual, and
structured output) matched the AR text exactly at MTP depths 1, 2, and 3.
This is a narrow correctness gate, not a general quality evaluation.

Depth 3 did not scale as strongly outside short single-stream decode:

| Case | Metric | llama.cpp AR | llama.cpp MTP depth 3 | oMLX 4-bit |
|---|---|---:|---:|---:|
| PP1K/TG256 C4 | aggregate tok/s | 9.13 | 10.01 | 18.19 |
| PP16K/TG256 C1 | decode tok/s | invalid | 6.65 | 14.65 |
| PP16K/TG256 C1 | MTP acceptance | - | 173 / 243 (71.2%) | not comparable |

The AR 16K row is invalid because the first server allocated only 16,384
tokens per slot: a 16,345-token prompt left room for 39 output tokens, not the
required 256. It must not be used as a long-context AR comparison. A later
depth-3 full run also reused the preceding C1 sweep's identical prompt cache;
its C1 TTFT and end-to-end rate are retained in the raw result but excluded
from the primary single-stream table. The independent depth-3 sweep supplies
the reported 19.90 tok/s result.

Depths 4 through 8 were stopped once the decision boundary was clear. Depth 3
remained below half of the qualified oMLX 4-bit decode rate, so a deeper sweep
could tune the experimental llama.cpp route but could not make it the daily
runtime recommendation. IQ2_M remains useful when compact storage, GGUF
portability, or a llama.cpp fallback matters; it is not the preferred quality
or performance configuration on this 64 GB M3 Max.

Ollama 0.33.0 was tested only through the local import stage. Its isolated
content store reached 20 GB while copying and parsing the 10.6 GB GGUF, which
reduced free disk from 39 GiB to 19 GiB. The import and benchmark were stopped,
the isolated store was removed, and free space returned to 39 GiB. This is a
peak import-space observation, not a steady-state model-size measurement.
Revisit Ollama after freeing at least 25 GiB of additional headroom or after a
verified zero-copy import path is available. Prefer qualifying Ollama's native
MLX model path over duplicating this GGUF when the experiment is resumed.

### External cross-check

An X search on 2026-08-26 found directional agreement, but no public result
with this exact IQ2_M file, M3 Max 64 GB, llama.cpp revision, and client corpus:

- a controlled dynamic 6-bit Apple Silicon comparison found that the best MTP
  depth changed by machine: depth 1 on M4 Max and depth 3 on M5 Max, with deeper
  settings losing performance. This supports treating depth as a measured
  runtime parameter rather than a universal constant ([post and chart](https://x.com/StefanoGogioso/status/2092557016308850840));
- a 24 GB M4 mini report measured a 2-bit Qwen3.8-27B at 7.4 tok/s with 64K
  context and described it as possible but painful
  ([post](https://x.com/WescheNex1q/status/2091293504089588181));
- a community deployment article recommended Q4 as the balance point and
  reported lower completion on long agent tasks for 2-bit. Much of its Mac
  table is explicitly estimated or attributed to external reports, so it is
  supporting context rather than controlled evidence
  ([article](https://x.com/servasyy_ai/status/2091416214283379123));
- an M4 Pro 64 GB user reported 17-26 tok/s for an Ollama Qwen3.8 4-bit MLX
  model and 28.41 tok/s for a direct MLX+DFlash path. The protocols differ, but
  the result supports testing native MLX and imported GGUF as separate runtime
  tracks ([post](https://x.com/aaronedell/status/2092035159667442137)).

These observations were not mixed into the tables above. The local controlled
run remains the source of record for this hardware and checkpoint.

The consolidated evidence and exclusions are recorded in
[`apple-m3-max-llamacpp-iq2-summary-2026-08-26.json`](../benchmarks/results/apple-m3-max-llamacpp-iq2-summary-2026-08-26.json).
