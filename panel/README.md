# Spark LLM Panel

A small read-only dashboard for the single vLLM service on the DGX Spark. It
fills the application-observability gap left by Beszel without duplicating
host monitoring.

## Boundaries

The panel reads only three local endpoints:

- `/health`
- `/v1/models`
- `/metrics`

It has no mutation endpoints, Docker socket, SSH support, host filesystem
mounts, power controls, or benchmark trigger. Hardware history and alerts stay
in Beszel. Load-generating benchmarks stay in `scripts/benchmark.sh` and
`scripts/benchmark-head-ab.sh`.

`/apps` is a server-rendered Tailnet-only directory for the Spark's public and
private web services. It probes only a fixed server-side allowlist and never
accepts a browser-supplied target.

The deployed directory keeps VoxStudio's public Cloudflare hostname distinct
from every Tailnet-only management and inference endpoint.

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
| `VLLM_BASE_URL` | `http://127.0.0.1:18102` | Read-only vLLM target |
| `PANEL_DB` | `panel/panel.db` | SQLite history path |
| `PANEL_POLL_SECONDS` | `2` | Live sampling interval |
| `PANEL_HISTORY_DAYS` | `14` | Minute-history retention |
| `BESZEL_URL` | unset | Hardware-dashboard link |
| `BESZEL_PROBE_URL` | unset | Server-side Beszel health target |
| `VOX_PUBLIC_URL` | unset | Public VoxStudio link and health target |
| `VOX_TAILNET_URL` | unset | Private VoxStudio link |
| `LLM_PANEL_URL` | unset | Private telemetry-panel link |
| `QWEN_API_URL` | unset | Private OpenAI-compatible base URL |
| `DGX_DASHBOARD_URL` | unset | Private NVIDIA dashboard link |

The provided systemd unit runs as a dynamic unprivileged user with a read-only
system view and a single writable state directory.

## Tests

```bash
python3 -m unittest panel.test_server
```
