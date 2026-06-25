from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable

from vllmstat.core.state import GpuSample, GpuSnapshot
from vllmstat.providers.gpu_amd import parse_amd_smi_json, read_amd_sysfs
from vllmstat.providers.gpu_fdinfo import read_fdinfo
from vllmstat.providers.gpu_intel import (
    pdev_for_card,
    read_gtidle_util,
    read_intel_sysfs,
    read_pci_vram_total,
)
from vllmstat.providers.gpu_sysfs import Card, detect_cards

_SMI_QUERY = (
    "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
    "power.draw,power.limit,clocks.sm,clocks.mem,fan.speed"
)


def _f(x: str) -> float | None:
    x = x.strip()
    if not x or x.upper() in ("N/A", "[N/A]"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def read_nvml(nvml: object) -> GpuSnapshot:
    nvml.nvmlInit()  # type: ignore[attr-defined]
    try:
        gpus: list[GpuSample] = []
        for i in range(nvml.nvmlDeviceGetCount()):  # type: ignore[attr-defined]
            h = nvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore[attr-defined]
            util = nvml.nvmlDeviceGetUtilizationRates(h)  # type: ignore[attr-defined]
            mem = nvml.nvmlDeviceGetMemoryInfo(h)  # type: ignore[attr-defined]
            name = nvml.nvmlDeviceGetName(h)  # type: ignore[attr-defined]
            if isinstance(name, bytes):
                name = name.decode()

            def _try(fn, *a):  # type: ignore[no-untyped-def]
                try:
                    return fn(*a)
                except Exception:  # noqa: BLE001 - optional metric not supported
                    return None

            power = _try(nvml.nvmlDeviceGetPowerUsage, h)  # type: ignore[attr-defined]
            limit = _try(nvml.nvmlDeviceGetEnforcedPowerLimit, h)  # type: ignore[attr-defined]
            gpus.append(
                GpuSample(
                    index=i,
                    name=name,
                    util_gpu=float(util.gpu),
                    mem_used=int(mem.used),
                    mem_total=int(mem.total),
                    temp_c=_try(nvml.nvmlDeviceGetTemperature, h, nvml.NVML_TEMPERATURE_GPU),  # type: ignore[attr-defined]
                    power_w=(power / 1000.0) if power is not None else None,
                    power_limit_w=(limit / 1000.0) if limit is not None else None,
                    fan_pct=_try(nvml.nvmlDeviceGetFanSpeed, h),  # type: ignore[attr-defined]
                    clock_sm_mhz=_try(nvml.nvmlDeviceGetClockInfo, h, nvml.NVML_CLOCK_SM),  # type: ignore[attr-defined]
                    clock_mem_mhz=_try(nvml.nvmlDeviceGetClockInfo, h, nvml.NVML_CLOCK_MEM),  # type: ignore[attr-defined]
                )
            )
        return GpuSnapshot(available=True, source="nvml", gpus=gpus)
    finally:
        try:
            nvml.nvmlShutdown()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def parse_nvidia_smi_csv(text: str) -> list[GpuSample]:
    gpus: list[GpuSample] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue
        mu, mt = _f(parts[3]), _f(parts[4])
        clk_sm, clk_mem = _f(parts[8]), _f(parts[9])
        gpus.append(
            GpuSample(
                index=int(_f(parts[0]) or 0),
                name=parts[1],
                util_gpu=_f(parts[2]),
                mem_used=int(mu * 1024 * 1024) if mu is not None else None,
                mem_total=int(mt * 1024 * 1024) if mt is not None else None,
                temp_c=_f(parts[5]),
                power_w=_f(parts[6]),
                power_limit_w=_f(parts[7]),
                clock_sm_mhz=int(clk_sm) if clk_sm is not None else None,
                clock_mem_mhz=int(clk_mem) if clk_mem is not None else None,
                fan_pct=_f(parts[10]),
            )
        )
    return gpus


def _run_cli(cmd: list[str], timeout: float = 3.0) -> str | None:
    """Run a CLI, returning stdout or ``None`` on any failure (never raises)."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True
        ).stdout
    except Exception:  # noqa: BLE001 - tool missing/failed; degrade gracefully
        return None


class GpuProvider:
    def __init__(
        self,
        *,
        enabled: bool = True,
        drm_root: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        proc_root: str = "/proc",
        pdev_resolver: Callable[[str], str | None] = pdev_for_card,
    ) -> None:
        self.enabled = enabled
        self._drm_root = drm_root
        self._clock = clock
        self._proc_root = proc_root
        self._pdev_resolver = pdev_resolver
        self._mode: str | None = None
        self._nvml: object | None = None
        # Per-card Intel energy carry: card index -> (energy_uj, time).
        self._intel_energy: dict[int, tuple[int, float]] = {}
        # Per-card Intel fdinfo cycle carry for util deltas: card index -> eng->cycles.
        self._intel_fdinfo_busy: dict[int, dict[str, int]] = {}
        self._intel_fdinfo_total: dict[int, dict[str, int]] = {}
        # Per-card Intel gtidle carry for non-root util: idle-residency ms + time.
        self._intel_idle: dict[int, dict[str, int]] = {}
        self._intel_idle_t: dict[int, float] = {}

    # -- vendor backends -------------------------------------------------

    def _read_nvidia(self) -> tuple[list[GpuSample], str] | None:
        """Read all NVIDIA GPUs via NVML, falling back to nvidia-smi."""
        if self._mode in (None, "nvml"):
            try:
                if self._nvml is None:
                    import pynvml  # nvidia-ml-py exposes the `pynvml` module

                    self._nvml = pynvml
                assert self._nvml is not None
                snap = read_nvml(self._nvml)
                self._mode = "nvml"
                for g in snap.gpus:
                    g.vendor = "nvidia"
                return snap.gpus, "nvml"
            except Exception:  # noqa: BLE001 - fall back to nvidia-smi
                self._nvml = None
        smi = shutil.which("nvidia-smi")
        if smi:
            out = _run_cli([smi, f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"])
            if out is not None:
                self._mode = "nvidia-smi"
                gpus = parse_nvidia_smi_csv(out)
                for g in gpus:
                    g.vendor = "nvidia"
                return gpus, "nvidia-smi"
        return None

    def _read_amd(self, cards: list[Card]) -> tuple[list[GpuSample], str]:
        for tool in ("amd-smi", "rocm-smi"):
            exe = shutil.which(tool)
            if not exe:
                continue
            args = (
                [exe, "metric", "--json"]
                if tool == "amd-smi"
                else [exe, "--showuse", "--showmemuse", "--showtemp", "--showpower", "--json"]
            )
            out = _run_cli(args)
            if out is not None:
                gpus = parse_amd_smi_json(out)
                if gpus:
                    return gpus, tool
        # sysfs fallback, one sample per detected AMD card
        gpus = []
        for c in cards:
            g = read_amd_sysfs(c.path)
            g.index = c.index
            gpus.append(g)
        return gpus, "amdgpu-sysfs"

    def _read_intel(self, cards: list[Card]) -> tuple[list[GpuSample], str]:
        now = self._clock()
        gpus: list[GpuSample] = []
        for c in cards:
            prev = self._intel_energy.get(c.index)
            g, new_energy = read_intel_sysfs(c.path, prev, now)
            g.index = c.index
            if new_energy is not None:
                self._intel_energy[c.index] = new_energy

            # Total VRAM capacity from the largest prefetchable PCI BAR — the xe
            # driver exposes no VRAM-size sysfs node, but the whole of VRAM is
            # mapped through a big prefetchable BAR (world-readable, no root).
            g.mem_total = read_pci_vram_total(c.path)

            # Utilisation from the GT idle-residency counter — world-readable as
            # non-root, so this is the primary util source on xe/i915.
            gt_util, new_idle, idle_t = read_gtidle_util(
                c.path,
                prev_idle=self._intel_idle.get(c.index),
                now=now,
                prev_now=self._intel_idle_t.get(c.index),
            )
            self._intel_idle[c.index] = new_idle
            self._intel_idle_t[c.index] = idle_t
            g.util_gpu = gt_util

            # VRAM (and a util fallback) via DRM fdinfo. VRAM is root-gated when
            # the GPU process runs as a different user; util here is only a
            # fallback for when gtidle is unreadable.
            pdev = self._pdev_resolver(c.path)
            if pdev:
                stats, busy, total = read_fdinfo(
                    pdev,
                    proc_root=self._proc_root,
                    prev_busy=self._intel_fdinfo_busy.get(c.index),
                    prev_total=self._intel_fdinfo_total.get(c.index),
                    now=now,
                )
                if g.util_gpu is None:
                    g.util_gpu = stats.util_pct
                g.mem_used = stats.vram_used_bytes
                # mem_total comes from the PCI BAR above (fdinfo has no total).
                self._intel_fdinfo_busy[c.index] = busy
                self._intel_fdinfo_total[c.index] = total
            gpus.append(g)
        return gpus, "intel-sysfs"

    # -- public API ------------------------------------------------------

    def sample(self) -> GpuSnapshot:
        """Public entry. Degrades to an unavailable snapshot if a backend read
        raises unexpectedly, so one flaky sysfs/driver read can't crash the app.
        """
        if not self.enabled:
            return GpuSnapshot(available=False, source="none", error="disabled")
        try:
            return self._sample()
        except Exception as e:  # noqa: BLE001 - never let a backend read crash the tick
            return GpuSnapshot(
                available=False, source="error", error=f"gpu sample failed: {e}"
            )

    def _sample(self) -> GpuSnapshot:
        root = self._drm_root if self._drm_root is not None else "/sys/class/drm"
        cards = detect_cards(root)
        gpus: list[GpuSample] = []
        sources: list[str] = []

        nvidia_cards = [c for c in cards if c.vendor == "nvidia"]
        amd_cards = [c for c in cards if c.vendor == "amd"]
        intel_cards = [c for c in cards if c.vendor == "intel"]

        # NVIDIA: read via NVML/nvidia-smi when an NVIDIA card is present, or
        # when detection found nothing at all (NVML works without DRM sysfs).
        if nvidia_cards or not cards:
            nv = self._read_nvidia()
            if nv is not None:
                gpus.extend(nv[0])
                sources.append(nv[1])

        if amd_cards:
            amd_gpus, amd_src = self._read_amd(amd_cards)
            gpus.extend(amd_gpus)
            sources.append(amd_src)

        if intel_cards:
            intel_gpus, intel_src = self._read_intel(intel_cards)
            gpus.extend(intel_gpus)
            sources.append(intel_src)

        if not gpus:
            return GpuSnapshot(
                available=False,
                source="none",
                error="no GPUs detected (no NVML/nvidia-smi, no amdgpu/xe sysfs)",
            )

        source = sources[0] if len(sources) == 1 else "+".join(sources)
        return GpuSnapshot(available=True, source=source, gpus=gpus)
