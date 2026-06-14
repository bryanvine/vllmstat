# vllmstat

**`nvtop` for vLLM** — a zero-infrastructure interactive terminal dashboard for vLLM serving performance.

![vllmstat](https://raw.githubusercontent.com/bryanvine/vllmstat/main/docs/screenshot.png)

---

## Why vllmstat?

The standard observability stack for vLLM is Prometheus + Grafana: powerful, but heavyweight. You need a running Prometheus instance, a Grafana server, a dashboard JSON import, and a browser tab — all just to see whether your inference server is busy.

`vllmstat` replaces that for day-to-day monitoring. One command, no infrastructure. It scrapes the vLLM server's built-in `/metrics` endpoint directly and renders everything in your terminal, refreshing every second.

There is one other terminal tool (`vllm-top` on PyPI), but it is a basic `watch`-style metrics printer: no interactivity, no GPU panel, no latency percentiles, no speculative-decoding acceptance, no KV-compression ratio. `vllmstat` fills that gap — it is closer to `nvtop` than to `watch`.

---

## Install

```bash
pip install vllmstat
```

Or with pipx (isolated install, globally available):

```bash
pipx install vllmstat
```

Or run it ephemerally without installing:

```bash
uvx vllmstat
```

---

## Usage

Point it at your vLLM server and it starts immediately:

```bash
vllmstat
```

```bash
# Different host / port
vllmstat --url http://my-gpu-host:8000
```

```bash
# Try the dashboard without a real server (uses synthetic data)
vllmstat --mock
```

```bash
# Print a single snapshot as JSON and exit — useful for scripting / alerting
vllmstat --once --json
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
- **Per-GPU stats** — utilisation %, VRAM used / total, temperature, power draw vs. limit, clocks, fan. Works on NVIDIA, AMD, and Intel GPUs (see [GPU support](#gpu-support) for what each vendor reports). Multi-GPU and mixed-vendor hosts show every GPU.

---

## GPU support

`vllmstat` detects each GPU's vendor from its DRM device and reads stats from the best source available. Every field degrades to `—` when its source is unavailable, and a missing driver, tool, or sysfs file never crashes the dashboard — it just shows less.

| Vendor | What works | Prerequisite |
|--------|-----------|--------------|
| **NVIDIA** | Full: util %, VRAM used/total, temperature, power draw/limit, SM & memory clocks, fan %. | NVIDIA driver. The bundled `nvidia-ml-py` uses NVML; `nvidia-smi` on `PATH` is used as a fallback. |
| **AMD** | Full: util %, VRAM used/total, temperature, power draw/limit, fan RPM, clock — via the `amdgpu` kernel driver's sysfs. | `amdgpu` kernel driver (in-tree on modern Linux). Install ROCm's `amd-smi` (or `rocm-smi`) for richer data; it's used automatically when on `PATH`. |
| **Intel** | Temperature, power draw/limit, clock, and fan RPM out of the box via the `xe`/`i915` sysfs. **util % and VRAM are best-effort and usually unavailable** on the `xe` driver. | `xe` or `i915` kernel driver. No extra tools needed for temp/power/clock/fan. |

**Intel limitation (known):** the `xe` driver exposes no `gpu_busy_percent` and no `mem_info_vram_*`, so utilisation and VRAM cannot be read from sysfs. `vllmstat` makes a best-effort attempt to derive util % from per-process `drm-cycles` in `/proc/*/fdinfo` (which typically requires root and is unsupported on `xe`), and otherwise shows `—` with a `(util/VRAM: see prereqs)` hint. `intel_gpu_top` only supports `i915`, so it is not used for `xe`. Full Intel util/VRAM via Level-Zero / `xpu-smi` is planned for a future release. Intel power is derived from the `energy1_input` counter, so it appears one refresh after the panel opens.

---

## Remote and containerised setups

`vllmstat` does not need to run on the GPU machine. If no GPU is reachable from the machine you run it on — no NVML/`nvidia-smi`, no `amdgpu`/`xe` sysfs — for example when monitoring a remote server or when vLLM is isolated in its own GPU container, the GPU panel shows "unavailable" and all the vLLM telemetry panels (concurrency, throughput, cache, latency, spec-decode) continue to work normally. Pass `--no-gpu` to suppress the panel entirely.

---

## Requirements

- Python ≥ 3.10
- A running vLLM server that exposes its Prometheus `/metrics` endpoint (all vLLM ≥ 0.4 deployments do this by default)
- A GPU driver — **optional**, only needed for the GPU panel. NVIDIA (NVML/`nvidia-smi`), AMD (`amdgpu`), or Intel (`xe`/`i915`); see [GPU support](#gpu-support).

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Apache-2.0. See [LICENSE](LICENSE).
