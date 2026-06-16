# vllmstat

**`nvtop` for vLLM** — a zero-infrastructure interactive terminal dashboard for vLLM serving performance.

![vllmstat](https://raw.githubusercontent.com/bryanvine/vllmstat/main/docs/demo.gif)

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
| `g` | Toggle GPU panel / column on/off |
| `r` | Reset the SESSION averages (of the selected instance) |
| `t` | Toggle the TEE request-feed panel |
| `↑` / `↓` (or `k` / `j`) | Fleet overview: move the selection |
| `Enter` | Fleet overview: open the selected instance's dashboard |
| `Esc` | Drill-in: return to the fleet overview |
| `+` / `=` | Halve the refresh interval (faster) |
| `-` | Double the refresh interval (slower) |

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-u` / `--url` | `http://localhost:8000` | vLLM server base URL. **Repeatable** — pass it more than once for a fleet. |
| `--config` | — | Path to a TOML config file defining instances (see [Fleet monitoring](#fleet--multi-instance-monitoring)) |
| `--discover-docker` | — | Auto-discover local vLLM Docker containers and add them to the fleet (also wires each one's log tee) |
| `--logs` | — | Tail a log source into the TEE request-feed panel: `docker:NAME` or a file path |
| `--proxy` | — | Run a reverse proxy on `[HOST:]PORT` that tees full prompts/responses (needs `vllmstat[proxy]`) |
| `--metrics-path` | `/metrics` | Prometheus metrics path |
| `-i` / `--interval` | `1.0` | Refresh interval in seconds |
| `--api-key` | — | Bearer token (`VLLM_API_KEY` env var also accepted) |
| `--no-gpu` | — | Disable the GPU panel entirely |
| `--mock` | — | Use synthetic data — no server required |
| `--once --json` | — | Print one snapshot as JSON and exit (a JSON array in fleet mode) |
| `--version` | — | Print version and exit |

---

## What it shows

- **Concurrency** — running requests and waiting queue depth (each with its session-**peak** high-water mark, so you can size `--max-num-seqs` and decide whether you're batch-bound or queue-bound), preemption rate, with mini sparklines.
- **Throughput** — generation tok/s, prompt tok/s, tokens per iteration, requests per second.
- **Session (while serving)** — running averages accumulated only while the server is actively serving (i.e. requests in flight, so idle gaps don't dilute the numbers): average decode and prefill/pp tok/s, the busy/idle split with the fraction of time spent serving, total requests completed, average generated tokens per request, and cumulative generated/prompt token totals. Press `r` to reset these counters at any time.
- **Cache & KV memory** — prefix-cache hit rate (windowed and lifetime), token-source breakdown (compute vs. cache-hit vs. external KV transfer), KV-cache utilisation percentage, KV-cache capacity in tokens, and — when a quantised KV dtype is detected — the dtype (`fp8_e4m3`, `turboquant_k3v4_nc`, …), effective compression ratio vs. fp16, and how much fp16 memory the model's full context would require. For example, a `turboquant k3v4` cache shows ~4.6× compression and a note that the full context would need 25.8 GB in fp16.
- **Latency percentiles** — TTFT, TPOT, end-to-end, and queue-wait time, each at p50 / p90 / p99, computed over a rolling window so recent spikes are visible immediately — plus the **per-phase breakdown** (queue → prefill → decode) so you can see *where* end-to-end latency is spent.
- **Request shape** — prompt-length and generation-length distributions (avg / p50 / p90 token counts), so you can see the workload at a glance: long-context vs. long-generation.
- **Max context (lifetime)** — the largest single request the server has seen, as bucketed prompt-token and output-token maxima. Their sum is the smallest `--max-model-len` that would still fit every request observed, shown as a percentage of your configured context window — so a low percentage means headroom to shrink the context and reclaim KV cache. (Bucketed upper bounds; hidden until the first request completes.)
- **Outcomes & goodput** — finish-reason breakdown (stop vs. length-truncated vs. abort/error) and the fraction of requests meeting an SLO (defaults: TTFT < 1 s, TPOT < 50 ms).
- **Efficiency** — model FLOP/s, memory bandwidth, MFU, and **energy efficiency** — tokens per watt and joules per token as **session averages** (accumulated only while actively generating, so they stay on screen but stop changing once the server goes idle, rather than decaying or dividing by zero) — plus the **average idle power** the GPU draws while not serving.
- **Speculative decoding** — acceptance rate, accepted tokens per draft, per-position acceptance (when the server reports it). The panel is hidden when spec-decode is not active.
- **Config advisor** — a panel that surfaces *only* when something looks wrong or sub-optimal, each line flagged `⚠` (sub-optimal) or `✖` (error / silent loss): KV-cache exhaustion (preemptions), output truncation at the length cap, request errors, an over-provisioned context window (using the max-context metric above), under-committed GPU memory, a queue backlog, ineffective speculative decoding, disabled prefix caching, and GPUs near their thermal/power limit. Each issue comes with the knob to turn. Stays hidden when the server is healthy.
- **Per-GPU stats** — utilisation %, VRAM used / total, temperature, power draw vs. limit, clocks, fan. Works on NVIDIA, AMD, and Intel GPUs (see [GPU support](#gpu-support) for what each vendor reports). Multi-GPU and mixed-vendor hosts show every GPU.
- **Fleet / multi-instance** — monitor many vLLM servers at once (local Docker containers and/or remote hosts) from one nvtop-style overview, and drill into any instance's full dashboard. See [Fleet monitoring](#fleet--multi-instance-monitoring).
- **Tee** — a live, toggleable panel of traffic: a request feed tailed from the server logs, or (in proxy mode) the full prompts and streamed completions. See [Tee](#tee--request-feed--content-tee).

---

## Fleet / multi-instance monitoring

Point one `vllmstat` at **many** vLLM servers at once — several local Docker containers each pinned to different GPUs, remote servers across your network, or both. You get an nvtop-style **overview** with one line per instance; press <kbd>Enter</kbd> to **drill into** any instance's full dashboard and <kbd>Esc</kbd> to come back.

![vllmstat fleet overview](https://raw.githubusercontent.com/bryanvine/vllmstat/main/docs/fleet.png)

A single `--url` (or no arguments at all) keeps the classic single-instance dashboard unchanged — fleet mode activates only when more than one instance is resolved.

### Three ways to define a fleet

They all merge together, de-duplicated by URL:

**1. Repeatable `--url`** — ad-hoc, no config:

```bash
vllmstat --url http://localhost:8000 --url http://gpu-box-2:8000
```

**2. A config file** — first found of `--config PATH`, `$VLLMSTAT_CONFIG`, `./vllmstat.toml`, or `~/.config/vllmstat/config.toml`:

```toml
# optional global defaults (an explicit CLI flag still overrides these)
interval = 1.0
gpu = true

[[instance]]
name = "qwen3-30b"
url  = "http://localhost:8000"
gpus = [0]                         # local → show GPU 0's hardware stats

[[instance]]
name = "llama-70b"
url  = "http://localhost:8001"
gpus = [1]

[[instance]]
name    = "remote-a100"
url     = "http://gpu-box-2:8000"  # remote → serving metrics only
api_key = "sk-..."
```

**3. Docker auto-discovery** — scan the local Docker daemon for vLLM containers and add them automatically, including each one's published port and `--gpus` / `NVIDIA_VISIBLE_DEVICES` pinning:

```bash
vllmstat --discover-docker
```

It looks for containers whose image or command mentions `vllm`. If Docker isn't installed or reachable, discovery is silently skipped — it never crashes the dashboard.

**Zero-config auto-discovery:** if you just run `vllmstat` with no arguments and nothing is listening on the default `localhost:8000`, it automatically falls back to Docker discovery and monitors whatever vLLM containers it finds (so a containerised vLLM on some other port just works). Pass any explicit `--url`/`--config`/`--discover-docker` and this fallback is skipped.

### Local vs. remote

Each instance is classified **local** or **remote** automatically from its hostname (override with `local = true` / `local = false` in the config). Local instances are mapped to the GPUs listed in `gpus = [...]` (or found by Docker discovery) and show those GPUs' hardware stats — utilisation, VRAM, temperature, power — sliced from the host. Remote instances show serving metrics only: reading another machine's GPU hardware over HTTP isn't possible, since vLLM's `/metrics` endpoint doesn't expose it.

### Scripting a fleet

`--once --json` emits a single object for one instance, or a JSON **array** (one element per instance, tagged with `name` / `url` / `locality`) for a fleet:

```bash
vllmstat --once --json --url http://localhost:8000 --url http://localhost:8001
```

---

## Tee — request feed & content tee

A **TEE** panel under the dashboard shows traffic to your vLLM server, from either of two sources. Press **`t`** to toggle it.

### Request feed (from logs — zero setup)

Tail the server's logs for a live feed of incoming requests:

![vllmstat tee request feed](https://raw.githubusercontent.com/bryanvine/vllmstat/main/docs/tee.png)

```bash
vllmstat --logs docker:vllm-xpu        # tail a Docker container's logs
vllmstat --logs /var/log/vllm.log      # …or a log file
```

You can set it per-instance in the config (`logs = "docker:NAME"`), and `--discover-docker` wires it up automatically for every vLLM container it finds. It shows method, path, status, and client per request (`4xx`/`5xx` flagged); health-check / metrics noise (`/health`, `/metrics`, `/v1/models`) is filtered. It does **not** show prompt/response *text* — modern vLLM (the V1 engine) doesn't log content, only access lines. For that, use proxy mode ↓.

### Content tee (proxy — full prompts & responses)

Run vllmstat as a small reverse proxy in front of vLLM and point your client at it; it forwards every request (streaming included, byte-for-byte) and tees the **actual prompts and completions**:

![vllmstat content tee](https://raw.githubusercontent.com/bryanvine/vllmstat/main/docs/proxy.png)

```bash
pip install 'vllmstat[proxy]'                        # adds aiohttp
vllmstat --proxy 9000 --url http://localhost:8000    # clients now call :9000
```

Point your app (or e.g. open-webui) at `http://<host>:9000`. Streaming responses are relayed to the client unchanged while the completion is accumulated live in the panel. The proxy targets a single upstream. Captured prompts/responses render only in your local terminal — nothing is stored or sent anywhere — but treat the panel as sensitive if your prompts are.

---

## GPU support

`vllmstat` detects each GPU's vendor from its DRM device and reads stats from the best source available. Every field degrades to `—` when its source is unavailable, and a missing driver, tool, or sysfs file never crashes the dashboard — it just shows less.

| Vendor | What works | Prerequisite |
|--------|-----------|--------------|
| **NVIDIA** | Full: util %, VRAM used/total, temperature, power draw/limit, SM & memory clocks, fan %. | NVIDIA driver. The bundled `nvidia-ml-py` uses NVML; `nvidia-smi` on `PATH` is used as a fallback. |
| **AMD** | Full: util %, VRAM used/total, temperature, power draw/limit, fan RPM, clock — via the `amdgpu` kernel driver's sysfs. | `amdgpu` kernel driver (in-tree on modern Linux). Install ROCm's `amd-smi` (or `rocm-smi`) for richer data; it's used automatically when on `PATH`. |
| **Intel** | Utilisation %, temperature, power draw/limit, GPU clock, fan RPM, and **total VRAM** out of the box via the `xe`/`i915` sysfs — **no root**. **VRAM used** via DRM `fdinfo` — see the note below for the root requirement. The `xe` driver exposes no memory clock, so the clock shows just the GPU clock (`clk 2800 MHz`, no `/mem`). | `xe` or `i915` kernel driver. No extra tools needed; util/temp/power/clock/fan/total-VRAM work as a normal user. **Root** (or matching UID) is only needed for VRAM *used*. |

**Intel utilisation (no root):** the `xe` driver exposes no `gpu_busy_percent`, but it does expose a world-readable, cumulative GT-idle counter at `…/device/tile*/gt*/gtidle/idle_residency_ms`. `vllmstat` reads it each refresh and derives util % as `100 × (1 − Δidle_ms / Δwall_ms)`, taking the busiest GT (a card can have a render/compute `gt0` and a media `gt1`). No root, no extra tools. Utilisation needs two refreshes to produce its first delta; Intel power is derived from the `energy1_input` counter, so it likewise appears one refresh after the panel opens.

**Intel VRAM (DRM `fdinfo`, root-gated):** the `xe` driver exposes no `mem_info_vram_*` in sysfs, so `vllmstat` reads VRAM *used* the way `nvtop` does — by summing each GPU client's `drm-resident-vram0` from `/proc/<pid>/fdinfo/<fd>`. Reading another process's `fdinfo` requires a matching UID or root, so VRAM *used* appears only when `vllmstat` can read the vLLM worker processes (see **Getting GPU stats** below). Without that access used-VRAM shows `—` with a `(VRAM needs root)` hint. **Total** VRAM, however, comes from the GPU's largest prefetchable PCI BAR (`…/device/resource`) — world-readable, no root — so the memory percentage and `used/total` render as soon as used-VRAM is available.

---

## Getting GPU stats

The GPU panel works with no configuration on all three vendors — but each vendor sources its data differently, and one case (Intel VRAM) can need elevated permissions. Here's how to get the full set.

### NVIDIA

Install the NVIDIA driver. Utilisation, VRAM used/total, temperature, power draw/limit, and SM/memory clocks all come from NVML via the bundled `nvidia-ml-py`; if NVML isn't importable, `vllmstat` falls back to `nvidia-smi` on your `PATH`. No root required.

### AMD

The in-tree `amdgpu` kernel driver (present on modern Linux) exposes utilisation, VRAM used/total, temperature, power, and fan via sysfs out of the box — no root, no extra tools. For richer data, install ROCm's `amd-smi` (or the older `rocm-smi`); `vllmstat` uses whichever is on your `PATH` automatically.

### Intel (Arc / `xe` or `i915`)

**Utilisation, temperature, power, clocks, and fan work out of the box, no root** — they come from world-readable sysfs (utilisation from the GT idle-residency counter; see [GPU support](#gpu-support) above for details).

**VRAM** is the one exception. It's read per-process from DRM `fdinfo`, so it only appears when `vllmstat` can read the GPU process. If your vLLM runs as **root** (e.g. inside Docker) while you run `vllmstat` as a normal user, VRAM shows `—` with a `(VRAM needs root)` hint. To get VRAM, either:

- **Run `vllmstat` as the same user as vLLM** (simplest if you launched vLLM yourself), or
- **Run `vllmstat` as root** to match a root-owned vLLM:

  ```bash
  sudo $(which vllmstat)
  # for a pipx install:
  sudo ~/.local/bin/vllmstat
  ```

> Note: `kernel.yama.ptrace_scope` does **not** help here. Reading another user's `fdinfo` is blocked by a cross-UID `ptrace_may_access` check that requires a matching UID or root — relaxing `ptrace_scope` does not change it.

### Keeping vllmstat current

```bash
pipx upgrade vllmstat
```

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
