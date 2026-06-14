# vllmtop

**`nvtop` for vLLM** — a zero-infrastructure interactive terminal dashboard for vLLM serving performance.

![vllmtop](docs/screenshot.svg)

---

## Why vllmtop?

The standard observability stack for vLLM is Prometheus + Grafana: powerful, but heavyweight. You need a running Prometheus instance, a Grafana server, a dashboard JSON import, and a browser tab — all just to see whether your inference server is busy.

`vllmtop` replaces that for day-to-day monitoring. One command, no infrastructure. It scrapes the vLLM server's built-in `/metrics` endpoint directly and renders everything in your terminal, refreshing every second.

There is one other terminal tool (`vllm-top` on PyPI), but it is a basic `watch`-style metrics printer: no interactivity, no GPU panel, no latency percentiles, no speculative-decoding acceptance, no KV-compression ratio. `vllmtop` fills that gap — it is closer to `nvtop` than to `watch`.

---

## Install

```bash
pip install vllmtop
```

Or with pipx (isolated install, globally available):

```bash
pipx install vllmtop
```

Or run it ephemerally without installing:

```bash
uvx vllmtop
```

---

## Usage

Point it at your vLLM server and it starts immediately:

```bash
vllmtop
```

```bash
# Different host / port
vllmtop --url http://my-gpu-host:8000
```

```bash
# Try the dashboard without a real server (uses synthetic data)
vllmtop --mock
```

```bash
# Print a single snapshot as JSON and exit — useful for scripting / alerting
vllmtop --once --json
```

### Key bindings

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / resume polling |
| `g` | Toggle GPU panel on/off |
| `+` / `=` | Halve the refresh interval (faster) |
| `-` | Double the refresh interval (slower) |

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-u` / `--url` | `http://localhost:8000` | vLLM server base URL |
| `--metrics-path` | `/metrics` | Prometheus metrics path |
| `-i` / `--interval` | `1.0` | Refresh interval in seconds |
| `--api-key` | — | Bearer token (`VLLM_API_KEY` env var also accepted) |
| `--no-gpu` | — | Disable the GPU panel entirely |
| `--mock` | — | Use synthetic data — no server required |
| `--once --json` | — | Print one snapshot as JSON and exit |
| `--version` | — | Print version and exit |

---

## What it shows

- **Concurrency** — running requests, waiting queue depth, preemption rate, with mini sparklines.
- **Throughput** — generation tok/s, prompt tok/s, tokens per iteration, requests per second.
- **Cache & KV memory** — prefix-cache hit rate (windowed and lifetime), token-source breakdown (compute vs. cache-hit vs. external KV transfer), KV-cache utilisation percentage, KV-cache capacity in tokens, and — when a quantised KV dtype is detected — the dtype (`fp8_e4m3`, `turboquant_k3v4_nc`, …), effective compression ratio vs. fp16, and how much fp16 memory the model's full context would require. For example, a `turboquant k3v4` cache shows ~4.6× compression and a note that the full context would need 25.8 GB in fp16.
- **Latency percentiles** — TTFT, TPOT, end-to-end, and queue-wait time, each at p50 / p90 / p99, computed over a rolling window so recent spikes are visible immediately.
- **Speculative decoding** — acceptance rate, accepted tokens per draft, per-position acceptance (when the server reports it). The panel is hidden when spec-decode is not active.
- **Per-GPU stats** — utilisation %, VRAM used / total, temperature, power draw vs. limit, SM clock, memory clock. Multi-GPU servers show each GPU in a column.

---

## Remote and containerised setups

`vllmtop` does not need to run on the GPU machine. If NVML and `nvidia-smi` are not reachable from the machine you run it on — for example, when monitoring a remote server or when vLLM is isolated in its own GPU container — the GPU panel shows "unavailable" and all the vLLM telemetry panels (concurrency, throughput, cache, latency, spec-decode) continue to work normally. Pass `--no-gpu` to suppress the panel entirely.

---

## Requirements

- Python ≥ 3.10
- A running vLLM server that exposes its Prometheus `/metrics` endpoint (all vLLM ≥ 0.4 deployments do this by default)
- NVML / `nvidia-smi` — **optional**, only needed for the GPU panel

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Apache-2.0. See [LICENSE](LICENSE).
