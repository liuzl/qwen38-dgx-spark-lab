# Strata RunPod pilot build and qualification

Experimental only. No Pod is created by these scripts. Model success, GPU performance,
platform cleanup and token revocation still require the bounded live experiment.

## Build

Use an x86 Linux CPU builder. No GPU is passed to Docker. Limit the builder to two
compiler jobs, 2 CPU cores and 12 GiB RAM on shared hosts. Build context is a clean
Strata checkout at `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, **not this lab repository**.

1. Copy this directory's `build-engine.py` into that checkout as
   `strata-pilot-build-engine.py`.
2. Build using this directory's absolute Dockerfile path:

   ```sh
   docker build --cpu-quota 200000 --memory 12g \
     -f /absolute/path/to/lab/docker/strata-runpod/Dockerfile \
     -t strata-pilot:fb58e0d-sm80-sm89-portable /absolute/path/to/Strata
   ```

3. Inspect both `/opt/strata/engine/strata` and `strata-vision` with
   `cuobjdump --list-elf`: each must contain **sm_80 and sm_89**. Inspect BUILD.json,
   ELF architecture, dynamic library dependencies and SHA256. A Docker exit code alone
   is not sufficient. Do not run inference on an occupied production GPU.
4. From the lab checkout, build the final image containing the tested client scripts:

   ```sh
   docker build -f docker/strata-runpod/Dockerfile.runtime \
     -t strata-pilot:fb58e0d-sm80-sm89-qualified .
   ```

The engine image uses portable AVX2 CPU instructions and both A100/Ada CUDA targets.
Other SM targets, ARM CPUs and non-CUDA backends are outside this image's contract.
CUDA 13 runtime qualification requires compatible host drivers; this pilot selects
R580 or newer. The builder's older driver is not used for CPU-only compilation.
The final image requires an explicit Orca config; it bypasses automatic installer
model selection and recompilation. Publish/export and verify the exact image digest
before renting a GPU. No registry delivery has been established by this build alone.

## Local fault tests

```sh
uv run --no-project --with pillow python -m unittest discover -s tests -p test_strata_pilot.py -v
uvx ruff@0.16.6 check scripts/strata*.py docker/strata-runpod/build-engine.py tests/test_strata_pilot.py
```

The HTTP helper uses a child process per request; a parent deadline terminates slow
headers, slow JSON bodies and endless SSE. TTFT includes worker startup and network
latency. It is client-observed latency, not pure engine prefill time. All streams
require an explicit end event/finish reason; partial output cannot pass a gate.

`strata-quality.py --out PRIVATE_RUN_DIR --minutes 20` fixes twelve tasks:
2 repairs with restricted arithmetic/subscript interpretation against held-out cases,
1 JSON/schema task, 1 constrained proposed file edit with truthful not-run status,
3 Chinese tasks, 2 tool tasks, 2 actual PNG inputs, 1 linked long-context task.
It never executes model-generated programs. The file-edit case validates a proposed
edit, not an autonomous filesystem agent. Missing cases are `not-run`, and both tool
cases must pass in addition to the 10/12 threshold. Images, prompts, expected answers,
request hashes and results are retained in the private output directory.

`strata-pilot.py --phase api --image PRIVATE_RUN_DIR/fixtures/shapes.png --out ...`
tests streaming Chat tool use, image-dependent tool arguments, Responses/Messages
roundtrips with streamed answers/usage, and actual running-request metrics after
cancellation. The five-second recovery gate includes idle observation and the canary.
API/metrics absence is failure or unknown, never inferred success. Run on an isolated
single-slot engine for cancellation. Cache/performance phases require 128K context;
record actual context/slots and restart canaries before each configuration change.

## Budget and credential lifecycle

Before creating a Pod, arrange and verify platform-side termination within three
hours of billing start, including storage deletion semantics. If the selected API
cannot support and confirm that guard, do not create a Pod. The external watchdog
is an additional guard, not a substitute for platform termination.

Private watchdog state must contain:

```json
{
  "pod_id": "EXACT_NEW_POD_ID",
  "pod_name": "strata-pilot-UNIQUE_RUN_ID",
  "created_epoch": 0,
  "hourly_rate": 1.0,
  "stop_seconds": 10500,
  "platform_terminate_epoch": 10800,
  "platform_termination_verified": true
}
```

Replace all illustrative timestamps/IDs with creation-response facts. Set the verified
flag only after read-back or documented platform confirmation. Start the watchdog
immediately after creation; no workload starts before it is running. Identity and
limits are frozen at launch. It attempts stop before GET once the deadline is reached,
retries timeouts/errors, and fails closed after three failed queries. A confirmed stop
still requires evidence export and scoped Pod/volume termination; stopped disk fees
continue. Never treat a state flag as proof of a successful platform mutation.

The download script accepts only commit-pinned manifests and exact sizes/SHA256.
The temporary HF token transport file is deleted immediately after being read;
`finally` also removes it on import/read/validation failure. The token exists only in
process memory during downloading. Final experiment cleanup must still revoke the HF
token and remove its Keychain entry, as authorized; process cleanup is not revocation.

## Verified build (2026-10-09)

On the CPU-only izia100 builder, the final image ID is
`sha256:f75c663ec54f6012b8dae90195b41eb9e50bb51d9fcb7a59bf28b65412f88cae`
(amd64, 8,907,279,926 bytes). Both engine binaries contain sm_80 and sm_89, and
runtime dependency inspection passed. All nine Python source files in the image
match the reviewed source hashes. All 12 regression tests pass both on yutu and
inside the final image without GPU access or external networking.

The `qualified` tag means build/fault-test qualification only. No real GPU inference,
model-quality score, provider-side termination test or registry publication has been
completed. Those remain explicit paid-experiment admission gates.
