# Multi-vendor GPU support (v0.2.0) — Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD, checkbox steps.

**Goal:** Make vllmstat's GPU panel work for NVIDIA, AMD, and Intel GPUs (not just NVIDIA), with per-field graceful degradation and documented per-vendor prerequisites.

**Architecture:** Detect each DRM card's vendor, then read stats from the best available source per vendor, filling a `GpuSample` per device. Every field is optional and renders `—` when a source can't provide it (the render layer is already None-safe). No new hard Python deps — read sysfs directly and shell out to vendor CLIs (`amd-smi`/`rocm-smi`, `nvidia-smi`) only when present; NVIDIA also uses the bundled `nvidia-ml-py`.

**Grounded in real hardware:** the dev box has an **Intel Arc Battlemage G31** (`8086:e223`, `xe` driver). Captured live: `tile0/gt0/freq0/cur_freq=2800`, `max_freq=2800`; hwmon `name=xe` with `temp2_input`(pkg)=57000 mC, `temp3_input`(vram)=52000 mC, `energy1_input`=84023575462341 µJ (label `card`), `power1_cap`=275000000 µW, `fan1_input`=1060. No `gpu_busy_percent`, no `mem_info_vram_*` on `xe`. `intel_gpu_top` supports only `i915`, so it is NOT used for `xe`.

---

## Vendor → source matrix (what each backend reads)

| Field | NVIDIA (NVML) | AMD (amdgpu sysfs) | Intel (xe/i915 sysfs) |
|---|---|---|---|
| name | `nvmlDeviceGetName` | `/sys/.../device` PCI id → name map | PCI id → name map (`8086:e223`→"Intel Arc B-series") |
| util % | `GetUtilizationRates.gpu` | `gpu_busy_percent` | best-effort fdinfo `drm-cycles` agg; else `None` |
| mem used/total | `GetMemoryInfo` | `mem_info_vram_used`/`_total` | `None` on xe (document xpu-smi/L0 as future) |
| temp °C | `GetTemperature` | hwmon `temp1_input` | hwmon `temp2_input` (pkg) |
| power W | `GetPowerUsage`/1000 | hwmon `power1_average`/1e6 | hwmon `energy1_input` Δ/Δt /1e6 |
| power limit W | `GetEnforcedPowerLimit`/1000 | hwmon `power1_cap`/1e6 | hwmon `power1_cap`/1e6 |
| fan % or RPM | `GetFanSpeed` (%) | hwmon `fan1_input` (RPM) | hwmon `fan1_input` (RPM) |
| clock MHz (sm/mem) | `GetClockInfo` | hwmon `freq1_input`/`freq2_input` or `pp_dpm_sclk` | `tile0/gt0/freq0/cur_freq` |

AMD richer source (optional): `amd-smi metric --json` / `rocm-smi --showuse --showmemuse --showtemp --showpower --json` when the CLI is on PATH.

---

## Task 1: `GpuSample.vendor` + fan unit note

**Files:** Modify `src/vllmstat/core/state.py`; Test `tests/test_state.py`.

- [ ] Add `vendor: str = ""` and `fan_rpm: int | None = None` to `GpuSample` (keep `fan_pct` for NVIDIA %). Add a test asserting defaults. Run `pytest tests/test_state.py -q`. Commit `feat(gpu): add vendor and fan_rpm fields to GpuSample`.

## Task 2: vendor detection + sysfs helpers (`providers/gpu_sysfs.py`)

**Files:** Create `src/vllmstat/providers/gpu_sysfs.py`; Test `tests/test_gpu_sysfs.py`.

Pure helpers (all take a card directory path so tests use `tmp_path`):
- `detect_cards(drm_root="/sys/class/drm") -> list[Card]` where `Card(index:int, path:str, vendor:str)` and vendor ∈ {"nvidia","amd","intel","other"} from `device/vendor` (`0x10de`/`0x1002`/`0x8086`).
- `read_text(path) -> str|None` (None on missing/err); `read_int(path)`.
- `pci_name(card_path) -> str` via `device/device`+`device/vendor` → a small built-in id→name map with sensible fallback `f"{vendor} GPU {id}"`.

- [ ] TDD with `tmp_path` fake DRM trees (write `device/vendor` files etc.). Tests: detect 1 intel card (vendor file `0x8086`), mixed vendors, missing files → None. Commit `feat(gpu): DRM vendor detection + sysfs helpers`.

## Task 3: AMD backend (`providers/gpu_amd.py`)

**Files:** Create `src/vllmstat/providers/gpu_amd.py`; Test `tests/test_gpu_amd.py`.

- `read_amd_sysfs(card_path) -> GpuSample`: util `gpu_busy_percent`; mem `mem_info_vram_used/total` (bytes); hwmon (`device/hwmon/hwmon*/`) `temp1_input`/1000, `power1_average`/1e6, `power1_cap`/1e6, `fan1_input`, `freq1_input`. vendor="amd".
- `parse_amd_smi_json(text) -> list[GpuSample]`: parse `amd-smi metric --json` (list of GPUs with `usage.gfx_activity`, `mem_usage.used_vram`/`total_vram`, `temperature.edge`, `power.socket_power`, etc.). Be defensive about key names (support both amd-smi and rocm-smi shapes; missing→None).

- [ ] TDD: build a fake amdgpu sysfs tree in `tmp_path` (write `gpu_busy_percent`=`"42"`, `mem_info_vram_used/total`, a `hwmon/hwmon0/` with temp/power/fan), assert the sample. Add a captured `amd-smi --json` sample string and assert the parser. Commit `feat(gpu): AMD backend (amdgpu sysfs + amd-smi/rocm-smi)`.

## Task 4: Intel backend (`providers/gpu_intel.py`)

**Files:** Create `src/vllmstat/providers/gpu_intel.py`; Test `tests/test_gpu_intel.py`.

- `read_intel_sysfs(card_path, prev_energy: tuple[int,float]|None, now: float) -> tuple[GpuSample, tuple[int,float]|None]`:
  - freq: `device/tile0/gt0/freq0/cur_freq` (MHz) → `clock_sm_mhz`.
  - temp: hwmon (`name`=`xe` or `i915`) `temp2_input`/1000 (pkg) → `temp_c` (fallback `temp1_input`).
  - power: from `energy1_input` (µJ): if `prev_energy=(e_prev,t_prev)`, `power_w = (e-e_prev)/1e6 / (now-t_prev)`; return new `(e,now)` to carry forward. First call → power None.
  - power limit: `power1_cap`/1e6.
  - fan: `fan1_input` → `fan_rpm`.
  - util/mem: `None` (xe). vendor="intel".
- `intel_util_via_fdinfo(card_minor=128) -> float|None`: BEST-EFFORT — scan `/proc/*/fdinfo/*`, sum `drm-cycles-*` and `drm-total-cycles-*` deltas for the matching DRM client; return util% or None if unreadable/unsupported. Wrap all OS errors → None (root often required; must never raise).

- [ ] TDD: fake Intel sysfs tree in `tmp_path` (tile0/gt0/freq0/cur_freq, hwmon with name=`xe`, temp2_input, energy1_input, power1_cap, fan1_input). Assert temp/freq/fan/limit; assert power None on first call then a positive watt value on a second call with a higher energy + later `now`. Commit `feat(gpu): Intel xe/i915 sysfs backend (temp/power/clock/fan; best-effort util)`.

## Task 5: wire backends into `GpuProvider` (`providers/gpu.py`)

**Files:** Modify `src/vllmstat/providers/gpu.py`; Test `tests/test_provider_gpu.py`.

- Keep existing NVML path (`read_nvml`) and `parse_nvidia_smi_csv`.
- New `GpuProvider.sample()` logic: if `not enabled` → unavailable. Else `cards = detect_cards()`. If any NVIDIA card and NVML import works → read all NVIDIA via NVML (as today). For AMD cards → `amd-smi`/`rocm-smi` if on PATH else `read_amd_sysfs`. For Intel cards → `read_intel_sysfs` (carry per-card prev-energy in `self._intel_energy: dict[int, tuple[int,float]]`) + best-effort `intel_util_via_fdinfo`. Aggregate all samples into one `GpuSnapshot(available=bool(gpus), source="multi", gpus=[...])`. If detection finds nothing and NVML fails → `available=False` with a helpful error.
- Set `GpuSnapshot.source` to the actual source(s) used.

- [ ] Keep all existing gpu tests green; add a test that, given a `tmp_path` DRM root with one Intel card and a stubbed `now`, `GpuProvider` (with detection pointed at the tmp root via an injected `drm_root`/clock seam) returns an Intel sample with temp/freq. Commit `feat(gpu): vendor-dispatched multi-GPU sampling`.

## Task 6: render the vendor + RPM fan; show a per-field hint

**Files:** Modify `src/vllmstat/render.py`; Test `tests/test_render.py`.

- In `gpu()`: show vendor + name; util `—` when None; mem `—/—` when None; fan as `RPM` when `fan_rpm` set, `%` when `fan_pct` set; when util AND mem are both None (typical Intel xe), append a short hint like `(util/VRAM: see prereqs)`.
- [ ] Add a render test for an Intel-style sample (util/mem None, temp/freq/power set) asserting it shows the name, temp, and the hint, and does not raise. Commit `feat(ui): render GPU vendor, RPM fan, and missing-field hint`.

## Task 7: README prerequisites + version bump

**Files:** Modify `README.md`, `pyproject.toml` (`version = "0.2.0"`), `src/vllmstat/__init__.py` (`__version__ = "0.2.0"`).

- [ ] Add a **GPU support** section to the README: a table of vendor → what works → prereq:
  - NVIDIA: full (util/VRAM/temp/power/clocks). Prereq: NVIDIA driver (the bundled `nvidia-ml-py` uses NVML; `nvidia-smi` is the fallback).
  - AMD: full via `amdgpu` sysfs (util/VRAM/temp/power/fan/clocks). Prereq: `amdgpu` kernel driver; install ROCm `amd-smi`/`rocm-smi` for richer data.
  - Intel: temp/power/clocks/fan via `xe`/`i915` sysfs out of the box; util% and VRAM are best-effort on the `xe` driver (need per-process `drm-cycles` fdinfo, often root) — documented as a known limitation; `intel_gpu_top` only supports `i915`.
- [ ] Bump version to 0.2.0 in both files. Run full `ruff check . && ruff format --check . && pyright && pytest -q`. Commit `docs: GPU prerequisites; chore: bump to 0.2.0`.

## Out of scope (later)
Intel util%/VRAM via Level-Zero/xpu-smi; AMD via `amdsmi` python lib; per-instance GPU mapping (that's the v0.3 fleet feature).
