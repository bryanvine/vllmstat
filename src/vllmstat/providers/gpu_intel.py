"""Intel GPU backend for the ``xe`` (and ``i915``) drivers via sysfs.

The ``xe`` driver exposes no ``gpu_busy_percent`` and no ``mem_info_vram_*`` in
sysfs. Utilisation, however, *can* be derived without root from the per-GT
idle-residency counter (``read_gtidle_util``); only VRAM still requires the DRM
``fdinfo`` aggregator (see ``gpu_fdinfo``), which is root-gated when the GPU
process runs as a different user. What this module reads out of the box, all
world-readable as non-root: GPU clock, package temperature, fan RPM, power cap,
a power figure derived from the ``energy1_input`` counter delta, and GPU
utilisation from ``gtidle/idle_residency_ms``. It also resolves a card's PCI
address (``pdev``) so the fdinfo aggregator can be pointed at the right device.

Every read catches OS errors and degrades to ``None``; nothing here ever raises.
"""

from __future__ import annotations

import glob
import os

from vllmstat.core.state import GpuSample
from vllmstat.providers.gpu_sysfs import pci_name, read_int, read_text

# Energy-delta carry: (energy_microjoules, monotonic_seconds).
EnergyState = tuple[int, float]


def _hwmon_dir(card_path: str) -> str | None:
    """Return the xe/i915 hwmon dir (else the first hwmon under the card)."""
    base = os.path.join(card_path, "device", "hwmon")
    try:
        candidates = sorted(glob.glob(os.path.join(base, "hwmon*")))
    except OSError:
        return None
    if not candidates:
        return None
    for hw in candidates:
        if read_text(os.path.join(hw, "name")) in ("xe", "i915"):
            return hw
    return candidates[0]


def _div(path: str, denom: float) -> float | None:
    val = read_int(path)
    return (val / denom) if val is not None else None


def read_intel_sysfs(
    card_path: str,
    prev_energy: EnergyState | None,
    now: float,
) -> tuple[GpuSample, EnergyState | None]:
    """Build a GpuSample from Intel xe/i915 sysfs.

    ``prev_energy`` is the ``(energy_uj, time)`` from the previous sample (or
    ``None`` on the first call). Power is computed as
    ``(e - e_prev) / 1e6 / (now - t_prev)`` and ``None`` when there is no prior
    sample or no time elapsed. Returns ``(sample, new_energy_state)`` where the
    caller carries the state forward per card.
    """
    dev = os.path.join(card_path, "device")

    # GPU clock (MHz): tile0/gt0/freq0/cur_freq.
    clock_sm = read_int(os.path.join(dev, "tile0", "gt0", "freq0", "cur_freq"))

    temp_c = power_limit_w = None
    fan_rpm = None
    new_energy: EnergyState | None = None
    power_w: float | None = None

    hw = _hwmon_dir(card_path)
    if hw is not None:
        # Package temp (temp2_input); fall back to temp1_input.
        temp_c = _div(os.path.join(hw, "temp2_input"), 1000.0)
        if temp_c is None:
            temp_c = _div(os.path.join(hw, "temp1_input"), 1000.0)
        power_limit_w = _div(os.path.join(hw, "power1_cap"), 1e6)
        fan_rpm = read_int(os.path.join(hw, "fan1_input"))

        energy = read_int(os.path.join(hw, "energy1_input"))
        if energy is not None:
            new_energy = (energy, now)
            if prev_energy is not None:
                e_prev, t_prev = prev_energy
                dt = now - t_prev
                if dt > 0:
                    power_w = (energy - e_prev) / 1e6 / dt

    return (
        GpuSample(
            index=0,
            name=pci_name(card_path),
            vendor="intel",
            util_gpu=None,  # not available on xe
            mem_used=None,  # not available on xe
            mem_total=None,
            temp_c=temp_c,
            power_w=power_w,
            power_limit_w=power_limit_w,
            fan_rpm=fan_rpm,
            clock_sm_mhz=clock_sm,
        ),
        new_energy,
    )


def read_gtidle_util(
    card_path: str,
    prev_idle: dict[str, int] | None,
    now: float,
    prev_now: float | None,
) -> tuple[float | None, dict[str, int], float]:
    """Compute GPU utilisation from the per-GT idle-residency counter (no root).

    The ``xe`` driver exposes a cumulative GT-idle counter in milliseconds at
    ``<card>/device/tile*/gt*/gtidle/idle_residency_ms``, world-readable as
    non-root. Over a wall-clock window, the busy fraction of a GT is
    ``1 - Δidle_ms / Δwall_ms``; util% is ``100 * busy`` clamped to ``[0, 100]``.
    A card may expose several GTs (e.g. a render/compute ``gt0`` and a media
    ``gt1``); we take the **busiest** GT (max util) as the GPU utilisation.

    ``prev_idle`` maps each idle-counter path to its previous reading (or
    ``None`` on the first call) and ``prev_now`` is the matching timestamp.
    Returns ``(util_or_none, new_idle, now)``: ``util`` is ``None`` when there is
    no previous sample, no GT counter is readable, or the wall delta is not
    positive. The caller carries ``new_idle`` and ``now`` forward as the next
    call's ``prev_idle``/``prev_now``. Never raises.
    """
    pattern = os.path.join(card_path, "device", "tile*", "gt*", "gtidle", "idle_residency_ms")
    try:
        paths = sorted(glob.glob(pattern))
    except OSError:
        paths = []

    new_idle: dict[str, int] = {}
    best: float | None = None
    have_prev = prev_idle is not None and prev_now is not None
    dwall_ms = (now - prev_now) * 1000.0 if prev_now is not None else 0.0

    for path in paths:
        idle = read_int(path)
        if idle is None:
            continue
        new_idle[path] = idle
        if not have_prev or dwall_ms <= 0:
            continue
        assert prev_idle is not None  # narrowed by have_prev
        prev = prev_idle.get(path)
        if prev is None:
            continue
        didle = idle - prev
        util_gt = 100.0 * (1.0 - didle / dwall_ms)
        util_gt = max(0.0, min(100.0, util_gt))
        if best is None or util_gt > best:
            best = util_gt

    return best, new_idle, now


def pdev_for_card(card_path: str) -> str | None:
    """Return the PCI address (``pdev``) backing ``card_path``, e.g.
    ``0000:06:00.0``.

    ``<card_path>/device`` is a symlink into the PCI tree; its realpath
    basename is the PCI bus address that ``fdinfo`` reports as ``drm-pdev``.
    Returns ``None`` when the link can't be resolved. Never raises.
    """
    try:
        target = os.path.realpath(os.path.join(card_path, "device"))
    except OSError:
        return None
    name = os.path.basename(target)
    return name or None
