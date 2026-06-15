"""DRM vendor detection and small, never-raising sysfs read helpers.

These are pure functions that take an explicit path so tests can point them at
a fake ``/sys`` tree built under ``tmp_path``. Every reader catches OS errors and
returns ``None``/empty so a missing file or permission error degrades to ``—``
rather than crashing the app.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

DEFAULT_DRM_ROOT = "/sys/class/drm"

# PCI vendor ids → short vendor key used throughout the GPU backends.
_VENDOR_IDS = {
    0x10DE: "nvidia",
    0x1002: "amd",
    0x8086: "intel",
}

# Best-effort PCI device id → human name map. The fallback covers everything
# not listed here, so this only needs entries we want to show nicely.
_DEVICE_NAMES = {
    (0x8086, 0xE223): "Intel Arc B-series (Battlemage)",
    (0x8086, 0xE20B): "Intel Arc B580 (Battlemage)",
    (0x8086, 0xE20C): "Intel Arc B570 (Battlemage)",
    (0x8086, 0x56A0): "Intel Arc A770 (Alchemist)",
    (0x8086, 0x56A1): "Intel Arc A750 (Alchemist)",
}

_CARD_RE = re.compile(r"^card(\d+)$")


@dataclass(frozen=True)
class Card:
    index: int
    path: str
    vendor: str  # "nvidia" | "amd" | "intel" | "other"


def read_text(path: str) -> str | None:
    """Return the stripped contents of ``path`` or ``None`` on any error."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def read_int(path: str) -> int | None:
    """Return ``path`` parsed as an int (decimal or ``0x`` hex) or ``None``."""
    raw = read_text(path)
    if raw is None or not raw:
        return None
    try:
        # int(x, 0) honours a 0x/0o/0b prefix; default to base 10 otherwise.
        return int(raw, 0) if raw.lower().startswith(("0x", "0o", "0b")) else int(raw)
    except ValueError:
        return None


def hwmon_dir(card_path: str, names: tuple[str, ...] = ()) -> str | None:
    """Return the hwmon dir under ``card_path`` whose ``name`` matches ``names``.

    When ``names`` is empty or no name matches, returns the first hwmon dir.
    Never raises.
    """
    base = os.path.join(card_path, "device", "hwmon")
    try:
        candidates = sorted(glob.glob(os.path.join(base, "hwmon*")))
    except OSError:
        return None
    if not candidates:
        return None
    if names:
        for hw in candidates:
            n = read_text(os.path.join(hw, "name"))
            if n in names:
                return hw
    return candidates[0]


def scale_int(path: str, denom: float) -> float | None:
    """Return ``read_int(path) / denom`` or ``None`` when unreadable."""
    val = read_int(path)
    return (val / denom) if val is not None else None


def _vendor_key(vendor_id: int | None) -> str:
    if vendor_id is None:
        return "other"
    return _VENDOR_IDS.get(vendor_id, "other")


def detect_cards(drm_root: str = DEFAULT_DRM_ROOT) -> list[Card]:
    """Find real DRM cards under ``drm_root``.

    Only ``cardN`` directories that expose ``device/vendor`` are returned;
    connector nodes (``card0-DP-1``) and render nodes (``renderD128``) are
    skipped. Results are sorted by card index. Never raises.
    """
    try:
        entries = os.listdir(drm_root)
    except OSError:
        return []
    cards: list[Card] = []
    for entry in entries:
        m = _CARD_RE.match(entry)
        if not m:
            continue
        card_path = os.path.join(drm_root, entry)
        vendor_id = read_int(os.path.join(card_path, "device", "vendor"))
        if vendor_id is None:
            continue
        cards.append(Card(index=int(m.group(1)), path=card_path, vendor=_vendor_key(vendor_id)))
    cards.sort(key=lambda c: c.index)
    return cards


def pci_name(card_path: str) -> str:
    """Return a human-readable name for the GPU at ``card_path``.

    Uses the ``device/vendor`` + ``device/device`` PCI ids against a small
    built-in map, falling back to ``"<vendor> GPU <id>"``. Never raises.
    """
    vendor_id = read_int(os.path.join(card_path, "device", "vendor"))
    device_id = read_int(os.path.join(card_path, "device", "device"))
    vendor = _vendor_key(vendor_id)
    if vendor_id is not None and device_id is not None:
        name = _DEVICE_NAMES.get((vendor_id, device_id))
        if name is not None:
            return name
    if device_id is not None:
        return f"{vendor} GPU 0x{device_id:04x}"
    return f"{vendor} GPU"
