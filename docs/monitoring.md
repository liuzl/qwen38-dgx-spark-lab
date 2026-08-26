# Monitoring

The Spark uses two deliberately separate monitoring surfaces:

- **Beszel** owns host, GPU, Docker, history, and alerts across the tailnet.
- **Spark LLM Panel** owns read-only vLLM and DFlash2 telemetry for this one
  inference endpoint.

This avoids deploying sparkDash's second host-monitoring stack, privileged
container, SSH registry, and power controls for a single-Spark installation.

## Beszel NVIDIA agent

Beszel 0.18.8 has an ARM64 NVIDIA agent image. The Spark agent service uses:

```yaml
image: henrygd/beszel-agent-nvidia:0.18.8
environment:
  BESZEL_AGENT_GPU_COLLECTOR: nvidia-smi
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [utility]
```

On GB10, `nvidia-smi` reports utilization, temperature, and power. Dedicated
VRAM fields are unavailable because DGX Spark uses unified memory; Beszel's
host-memory chart remains the memory source of truth.

Recommended Spark alert policy:

| Alert | Threshold | Duration |
|---|---:|---:|
| Status down | — | 2 minutes |
| Memory | 90% | 5 minutes |
| Disk | 85% | 5 minutes |
| Highest temperature sensor | 80°C | 5 minutes |

Do not alert on high GPU utilization: sustained load is expected during local
inference and benchmarks.

## Spark LLM Panel

See [`panel/README.md`](../panel/README.md) for metrics, configuration, and the
systemd unit. Production deployment should bind only to the Spark's Tailscale
address or a loopback address behind Tailscale Serve.

The service requires no elevated permissions. Keep these invariants when
updating it:

- no Docker socket;
- no SSH credentials;
- no arbitrary proxy target supplied by the browser;
- no POST/mutation routes;
- no benchmark trigger;
- a single writable SQLite state directory.

## Benchmarks are operational tools

The dashboard is passive. Run load-generating measurements explicitly from the
repository:

```bash
scripts/benchmark.sh
scripts/benchmark-head-ab.sh
```

This separation prevents a dashboard click from saturating the only inference
machine or contaminating production telemetry.
