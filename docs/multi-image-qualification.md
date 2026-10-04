# Bounded multi-image serving

The `configs/media16.env.example` profile sets an explicit count of 16 images,
no video, 1,048,576 processor pixels and no processor cache. It is intended for
image-capable Qwen3.8 services behind a gateway that separately bounds request
bytes, decoded image bytes and source dimensions. It is not a hard pixel quota
for untrusted clients connecting directly to vLLM: gateways must reject
`mm_processor_kwargs` and `media_io_kwargs` overrides.

`serve-a100.sh` and `serve-native-lora.sh` consume these environment variables.
The latter keeps its previous behavior when the variables are absent. Do not
change weights, quantization, MTP depth or KV allocation while measuring the
media change. Keep multimodal startup profiling enabled.

## Fixtures and protocol qualification

Use Pillow to generate synthetic 1024x768 browser screenshots; no real user data
is needed. The 16 distinct pages show CODE 7400 through CODE 7415. The optional
`--font` argument pins a TrueType font; record which font was used with results.

```sh
python3 scripts/generate-browser-fixtures.py /tmp/browser-fixtures
python3 scripts/qualify-multi-image.py \
  --base-url http://127.0.0.1:18102 --fixtures /tmp/browser-fixtures \
  --label qwen38-nvfp4-v029-mtp-k7-media16 \
  --mode protocols --output /tmp/media-protocols.json
```

This checks 6/8/16 images on both aliases through Chat, Responses and Messages,
including two historical tool-result batches in Messages. The answer must
identify the code in the last image, rather than merely return HTTP 200.
Responses image blocks explicitly include `detail: auto`; reasoning is disabled
in that protocol for this OCR workload. The 17-image boundary must return 400 in
all three protocols. Results contain status, usage, timing and pass/fail, never
image payloads or generated text.

## Capacity qualification

Drain the target node first. `--mode capacity --repetitions 2` uses 16 images,
C1/C4/C8 image requests with short text and C1/C4 with longer background text
(1,500 repetitions of a fixed sentence; actual prompt tokens are in usage).
Each image batch has one concurrent small text request, so total concurrency is
image concurrency plus one. Both aliases are tested, with text-only baselines.
Each client/repetition gets a fresh cache salt and then an exact replay. Inspect
reported cached tokens to confirm cold/warm behavior instead of assuming a hit.
For this profile the multimodal processor cache remains off in both phases.

Run resource sampling during the tests and report GPU/system memory, KV use,
preemptions, failures, TTFT and effect on the concurrent text request. Keep the
runtime/quantization/context/concurrency/cache/repetition metadata alongside the
results. Do not compare timings from different protocols or quantizations as if
they shared one benchmark protocol.

## Rolling switch

- `switch-a100-production.sh` accepts `NEW_LIMIT_MM_PER_PROMPT` and restores the
  old environment and container on gate failure. Its existing image validator
  checks the new count plus one.
- For Spark, export the media profile and `MAX_IMAGES=16` when invoking
  `switch-spark-production.sh`; retain the old container and profile for rollback.
- `validate-multimodal.py --max-images 16` retains the existing single-image OCR
  and remote-URL checks and tests rejection at 17 images.
- Keep the gateway at its old count until all routing candidates qualify; then
  deploy and smoke the gateway. Roll the gateway back before reducing an engine
  limit. Host-specific launch paths and observed results belong in oplog.
