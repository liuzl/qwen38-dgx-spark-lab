# bj-123 Strata V100 pilot

Isolated experiment, not a tokenflow serving node. The 2026-10-09 SM80/89 CUDA 13
image is not compatible with V100; this build uses the already-installed CUDA 12.3
and GCC 11 without changing drivers or system packages.

- Strata v0.1.44.1: `296b59d2f73c8f04a805df29c2fa97655adace78`.
- llama.cpp: `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S,
  revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
- First shard: 54,817,524,224 bytes,
  SHA256 `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`.
- Second shard: 28,800,138,432 bytes,
  SHA256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.
- Remote root: `/home/zliu/experiments/strata-v100-20261014`.
- Runtime scope: GPUs 1,2 only, loopback port 18144, one slot, 32K context,
  INT8 KV. GPU 0 is occupied by another user's application.
- Protected baseline: yutu `/health` on port 2025, active and database OK.

Copy `build.sh` to the experiment root. In `source/`, create a private venv with
upstream requirements, unpack the pinned llama.cpp at `third_party/llama.cpp`,
then run `bash ../build.sh`. Build is CPU-only, four compiler jobs, SM70, portable
CPU kernels. Check build exit status, dependencies, cubin architecture and SHA256.

Do not run or pack partial model files. Download with the above hashes and verify
both exact sizes and hashes before using them. An allocated file size is not
proof of download completion. Preserve aria2 state for resuming failed downloads.

Initial decision contract: first establish correct JSON/text and two-turn tool
responses with the stock model. Use the existing scripts/strata-pilot.py HTTP
helper and strata-quality.py task scoring. Vision is out of scope for this first
text-only build and cannot be counted as passed. After a healthy baseline,
compare the bundled control vector off/on on identical benign tasks. Record model,
engine, actual input/output tokens, elapsed time, TTFT, GPU memory and failures.
Do not infer safety properties or general model quality from model self-reports.

Stop on hash mismatch, GPU 0 use, repeated OOM, protected-service degradation,
or cancellation that does not release the engine. Preserve evidence and stop only
the experiment's processes. No production route, tokenflow registration, driver
upgrade, paid resource creation or public endpoint is part of this pilot.

## Download lifecycle

Before running `bash download.sh EXPERIMENT_ROOT`, copy `download.aria2` to
`EXPERIMENT_ROOT/evidence/strata-download.aria2` and `model-manifest.json` to
`EXPERIMENT_ROOT/evidence/strata-iq3s-manifest.json` (create `evidence/` first).
The external user service supplies the resource limits and deadline below;
invoking the script alone does not impose a 72-hour deadline.

The user confirmed that bj-series internet bandwidth is physically constrained,
and requested **no download speed limit**. `download.sh` has no bandwidth cap.
The user unit `strata-v100-download-limited-20261014` retains its historical name
but its restarted aria2 command is uncapped. It has a 72-hour runtime deadline,
2 GiB process memory limit, two concurrent files, and eight connections per file.
It only downloads and verifies; it never starts a GPU process automatically.
Completion requires `evidence/model-verified.json` and `download.status=verified`.
The unit is transient (not a boot-enabled service); if interrupted, restart the
same script against the preserved aria2 files. Do not delete the partial weights.

Once `build.sh` finishes, `verify-kernel.sh` builds and runs the upstream synthetic
attention parity check on GPU 1. A passing kernel test is not model qualification.

## Verified 2026-10-14

Native build succeeded: x86_64 ELF, 62 sm_70 cubins. An initial CMake search mixed
CUDA 12.3 compilation with old system CUDA 11 libraries; build.sh now pins the
three dynamic CUDA library paths and rejects missing/old dependencies. Final
cudart/cuBLAS/cuBLASLt all resolve to `/usr/local/cuda-12.3/lib64/*.so.12`.
Engine SHA256: `d9b014906bb600e7cba268f9009d2c9b01e4dab39a7c7cfee62549a751be1194`.

The final GPU 1 attention parity run passed all four applicable cases (INT8 at
32K/1500/2100 context, FP16 at 32K), with zero failures; two Q4_0 cases were skipped
because their tensor-core path requires SM80+. This is synthetic kernel validation,
not an end-to-end model benchmark. Evidence: remote `evidence/build-cuda12-fixed.log`
and `evidence/kernel-parity-cuda12.log`. GPU memory was released and yutu health
remained OK. Weights are still downloading; no model inference or ESP A/B result.
