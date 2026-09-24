# Spark LLM Panel

A small read-only dashboard for the single vLLM service on the DGX Spark. It
fills the application-observability gap left by Beszel without duplicating
host monitoring.

## Boundaries

By default, the panel reads three local endpoints:

- `/health`
- `/v1/models`
- `/metrics`

It has no mutation endpoints, Docker socket, SSH support, power controls, or
benchmark trigger. Optional request observability adds read-only access to a
dedicated audit database; it does not grant general host filesystem access. Hardware history and alerts stay
in Beszel. Load-generating benchmarks stay in `scripts/benchmark.sh` and
`scripts/benchmark-head-ab.sh`.

`/apps` is an optional server-rendered directory for operator-configured web
services. Protect it with private access controls. It probes only a fixed
server-side allowlist and never accepts a browser-supplied target.

Configure public and private links separately. Actual hostnames, ports and
route mappings are intentionally omitted from this public documentation.

## Metrics

- live prompt and generation token rates;
- running and waiting requests;
- KV-cache utilization, prefix-cache hit rate, and preemptions;
- DFlash2 acceptance rate, acceptance length, accepted/drafted rates, and
  per-position acceptance;
- lifetime p95 TTFT, TPOT, and end-to-end latency from vLLM histograms;
- minute-level local history retained for 14 days by default.

## Local run

The server uses only the Python standard library:

```bash
PANEL_HOST=127.0.0.1 \
PANEL_PORT=18103 \
VLLM_BASE_URL=http://127.0.0.1:18102 \
PANEL_DB=/tmp/spark-llm-panel.db \
python3 panel/server.py
```

Then open `http://127.0.0.1:18103`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PANEL_HOST` | `127.0.0.1` | Listen address |
| `PANEL_PORT` | `18103` | Listen port |
| `VLLM_BASE_URL` | `http://127.0.0.1:18102` | Read-only vLLM target; the service-directory Qwen probe uses this URL |
| `PANEL_DB` | `panel/panel.db` | SQLite history path |
| `PANEL_POLL_SECONDS` | `2` | Live sampling interval |
| `PANEL_HISTORY_DAYS` | `14` | Minute-history retention |
| `PANEL_PAGE_TITLE` | `Spark LLM Panel` | Browser title |
| `PANEL_MARK` | `S` | One- or two-character visual mark |
| `PANEL_EYEBROW` | `DGX SPARK · GB10` | Small heading above the panel name |
| `PANEL_HEADING` | `Spark LLM` | Visible panel name |
| `PANEL_DESCRIPTION` | Spark-specific description | HTML metadata description |
| `PANEL_KV_CACHE_CODE` | `FP8` | KV cache badge text |
| `PANEL_KV_CACHE_NOTE` | Spark 16 GiB cache note | KV cache explanatory text |
| `PANEL_SPECULATIVE_LABEL` | `DFLASH2` | Speculative decoding label |
| `PANEL_FOOTER_NOTE` | Spark benchmark note | Right-side footer text |
| `PANEL_ENABLE_SERVICE_DIRECTORY` | `true` | Enable the Spark-specific `/apps` route |
| `BESZEL_URL` | unset | Hardware-dashboard link |
| `BESZEL_PROBE_URL` | unset | Server-side Beszel health target |
| `VOX_PUBLIC_URL` | unset | Public VoxStudio link and health target |
| `VOX_TAILNET_URL` | unset | Private VoxStudio link |
| `LLM_PANEL_URL` | unset | Private telemetry-panel link |
| `QWEN_API_URL` | unset | Private OpenAI-compatible base URL |
| `DGX_DASHBOARD_URL` | unset | Private NVIDIA dashboard link |

The provided systemd unit runs as a dynamic unprivileged user with a read-only
system view and a single writable state directory.

For a neutral deployment, install `llm-panel.service` and copy
`llm-panel.env.example` to `/etc/default/llm-panel`. This profile removes
machine/vendor branding and disables the Spark-specific `/apps` directory.

## Tests

```bash
python3 -m unittest panel.test_server
```

## Optional request observability

`request_audit.RequestAuditMiddleware` is a pure ASGI middleware loaded by vLLM.
It captures POST chat/completions, completions and Responses calls without
buffering or modifying the wire response. SQLite writes run on a bounded
background queue. It stores JSON/SSE bodies, status, elapsed time, upstream
usage and timing metrics. Request authorization/cookie headers are never stored.
The request body itself may contain private content and must be treated accordingly.

Enable via the A100 launcher with `ENABLE_REQUEST_AUDIT=1`, `AUDIT_DIR` pointing
to a private host directory, and optional `AUDIT_RETENTION_DAYS=7`. This adds
`--enable-force-include-usage`, `--enable-per-request-metrics` and the middleware;
keep `--enable-prompt-tokens-details` enabled. No image build is needed. These
flags require a container replacement / model reload. Streaming clients will
receive an additional final usage chunk if they did not already request one.

Set `PANEL_AUDIT_DB` to the read-only database path and `PANEL_AUDIT_TOKEN` to a
random operator secret via a private environment file. Grant the panel process
read access through a dedicated supplementary group. Do not grant write access
to the audit directory. Keep the existing private access controls on the panel.

- `/api/audit/summary?hours=1`: aggregate counts, tokens, exact nearest-rank
  P50/P95 latency from retained completed records (including failed requests).
- `/api/audit/requests`: paginated metadata; operator Bearer key required.
  Supports `hours` (1–168), exact `task` / `model` filters and `before` cursor.
- `/api/audit/requests/<id>`: bodies, usage, upstream ID and timing metrics;
  operator key required. Browser keeps the key in memory only; locking clears it.
- vLLM `/audit-health`: internal capture queue size, written/dropped records and
  last persistence error. Never add this endpoint to the public API allowlist.

The gateway can forward a caller-supplied `X-Task-ID`, or its own chat ID, and
set `X-Audit-Caller` from the verified user ID **only for the local inference
upstream**. These are attribution hints, not authorization. Direct local clients
can set headers themselves. Missing attribution remains unknown. Task progress,
artifacts and human questions require a separate task-runner integration; they
cannot be inferred from inference traffic.

Each input and output capture is capped at 2 MiB (`AUDIT_BODY_LIMIT`), with an
explicit truncation flag. Usage parsing continues after this cap. Missing usage
is NULL, not zero. SSE content arrival supplies fallback TTFT; server timing
metrics take precedence and also cover non-streaming responses when available.
The seven-day cleanup runs on writes, at most hourly. SQLite reuses freed space;
its file does not automatically shrink. Disk usage should be monitored.
A full 32-record queue or a write failure drops telemetry, not inference;
`/audit-health` reports these drops. Existing historical requests cannot be recovered.

Validation: `python3 -m unittest panel.test_server panel.test_audit`.
