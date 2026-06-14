"""AMD GPU backend: amdgpu sysfs reader + amd-smi/rocm-smi JSON parser.

Both entry points never raise: a missing sysfs file or an unexpected JSON shape
degrades to ``None`` fields rather than an exception, so the app falls back to
``—`` instead of crashing.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any

from vllmstat.core.state import GpuSample
from vllmstat.providers.gpu_sysfs import pci_name, read_int, read_text

_SCLK_ACTIVE_RE = re.compile(r"^\s*\d+:\s*([\d.]+)\s*MHz\s*\*", re.IGNORECASE | re.MULTILINE)


def _hwmon_dir(card_path: str, want: str = "amdgpu") -> str | None:
    """Return the hwmon dir whose ``name`` matches ``want`` (else the first one)."""
    base = os.path.join(card_path, "device", "hwmon")
    try:
        candidates = sorted(glob.glob(os.path.join(base, "hwmon*")))
    except OSError:
        return None
    if not candidates:
        return None
    for hw in candidates:
        if read_text(os.path.join(hw, "name")) == want:
            return hw
    return candidates[0]


def _div(path: str, denom: float) -> float | None:
    val = read_int(path)
    return (val / denom) if val is not None else None


def _sclk_from_pp_dpm(card_path: str) -> int | None:
    text = read_text(os.path.join(card_path, "device", "pp_dpm_sclk"))
    if not text:
        return None
    m = _SCLK_ACTIVE_RE.search(text)
    if not m:
        return None
    try:
        return int(float(m.group(1)))
    except ValueError:
        return None


def read_amd_sysfs(card_path: str) -> GpuSample:
    """Build a GpuSample from amdgpu sysfs under ``card_path`` (never raises)."""
    dev = os.path.join(card_path, "device")
    util = read_int(os.path.join(dev, "gpu_busy_percent"))
    mem_used = read_int(os.path.join(dev, "mem_info_vram_used"))
    mem_total = read_int(os.path.join(dev, "mem_info_vram_total"))

    temp_c = power_w = power_limit_w = None
    fan_rpm = None
    clock_sm = None
    hw = _hwmon_dir(card_path)
    if hw is not None:
        temp_c = _div(os.path.join(hw, "temp1_input"), 1000.0)
        power_w = _div(os.path.join(hw, "power1_average"), 1e6)
        power_limit_w = _div(os.path.join(hw, "power1_cap"), 1e6)
        fan_rpm = read_int(os.path.join(hw, "fan1_input"))
        freq1 = read_int(os.path.join(hw, "freq1_input"))
        if freq1 is not None:
            clock_sm = int(freq1 / 1e6)  # Hz -> MHz
    if clock_sm is None:
        clock_sm = _sclk_from_pp_dpm(card_path)

    return GpuSample(
        index=0,
        name=pci_name(card_path),
        vendor="amd",
        util_gpu=float(util) if util is not None else None,
        mem_used=mem_used,
        mem_total=mem_total,
        temp_c=temp_c,
        power_w=power_w,
        power_limit_w=power_limit_w,
        fan_rpm=fan_rpm,
        clock_sm_mhz=clock_sm,
    )


def _num(node: Any) -> float | None:
    """Extract a number from a plain scalar, numeric string, or {value:...} dict."""
    if isinstance(node, bool):  # bool is an int subclass; ignore it
        return None
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, str):
        try:
            return float(node.strip())
        except ValueError:
            return None
    if isinstance(node, dict):
        return _num(node.get("value"))
    return None


def _first(d: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        if k in d:
            n = _num(d[k])
            if n is not None:
                return n
    return None


def _find_key(d: dict[str, Any], *needles: str) -> float | None:
    """Case-insensitive substring key lookup (for rocm-smi's verbose labels)."""
    for key, value in d.items():
        kl = key.lower()
        if all(n in kl for n in needles):
            n = _num(value)
            if n is not None:
                return n
    return None


def _sample_from_amd_smi(index: int, gpu: dict[str, Any]) -> GpuSample:
    usage = gpu.get("usage", {}) if isinstance(gpu.get("usage"), dict) else {}
    mem = gpu.get("mem_usage", {}) if isinstance(gpu.get("mem_usage"), dict) else {}
    temp = gpu.get("temperature", {}) if isinstance(gpu.get("temperature"), dict) else {}
    power = gpu.get("power", {}) if isinstance(gpu.get("power"), dict) else {}

    util = _first(usage, "gfx_activity", "gfx", "gpu_activity")
    used_mb = _first(mem, "used_vram", "used_memory")
    total_mb = _first(mem, "total_vram", "total_memory")
    temp_c = _first(temp, "edge", "hotspot", "junction")
    power_w = _first(power, "socket_power", "average_socket_power", "current_socket_power")
    power_cap = _first(power, "power_cap", "cap")

    return GpuSample(
        index=index,
        name="AMD GPU",
        vendor="amd",
        util_gpu=util,
        mem_used=int(used_mb * 1024 * 1024) if used_mb is not None else None,
        mem_total=int(total_mb * 1024 * 1024) if total_mb is not None else None,
        temp_c=temp_c,
        power_w=power_w,
        power_limit_w=power_cap,
    )


def _sample_from_rocm_smi(index: int, gpu: dict[str, Any]) -> GpuSample:
    util = _find_key(gpu, "gpu use")
    total_b = _find_key(gpu, "vram", "total", "memory")
    used_b = _find_key(gpu, "vram", "used")
    temp_c = _find_key(gpu, "temperature", "edge")
    power_w = _find_key(gpu, "power")
    return GpuSample(
        index=index,
        name="AMD GPU",
        vendor="amd",
        util_gpu=util,
        mem_used=int(used_b) if used_b is not None else None,
        mem_total=int(total_b) if total_b is not None else None,
        temp_c=temp_c,
        power_w=power_w,
    )


def parse_amd_smi_json(text: str) -> list[GpuSample]:
    """Parse ``amd-smi metric --json`` or ``rocm-smi --json`` output.

    Supports the amd-smi list-of-GPU-objects shape (with ``{value, unit}``
    nesting) and the rocm-smi ``{"card0": {...}}`` flat-label shape. Returns an
    empty list on any parse error or unexpected shape.
    """
    if not text or not text.strip():
        return []
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []

    samples: list[GpuSample] = []
    if isinstance(data, list):
        # amd-smi: list of GPU objects.
        for i, gpu in enumerate(data):
            if not isinstance(gpu, dict):
                continue
            idx = gpu.get("gpu")
            samples.append(_sample_from_amd_smi(int(idx) if isinstance(idx, int) else i, gpu))
    elif isinstance(data, dict):
        # rocm-smi: {"card0": {...}, "card1": {...}}.
        for key, gpu in data.items():
            if not isinstance(gpu, dict):
                continue
            m = re.search(r"(\d+)", key)
            samples.append(_sample_from_rocm_smi(int(m.group(1)) if m else 0, gpu))
    return samples
