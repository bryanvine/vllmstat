from __future__ import annotations

from pathlib import Path

from vllmstat.providers.gpu_intel import (
    pdev_for_card,
    read_gtidle_util,
    read_intel_sysfs,
    read_pci_vram_total,
)


def _make_intel_card(
    tmp_path: Path,
    *,
    hwmon_name: str = "xe",
    energy_uj: int = 84023575462341,
    temp2: int | None = 57000,
    temp1: int | None = None,
) -> Path:
    """Build a fake Intel xe sysfs tree mirroring the real Battlemage layout."""
    card = tmp_path / "card0"
    dev = card / "device"
    # freq: tile0/gt0/freq0/cur_freq
    freq = dev / "tile0" / "gt0" / "freq0"
    freq.mkdir(parents=True)
    (freq / "cur_freq").write_text("2800\n")
    (freq / "max_freq").write_text("2800\n")
    # hwmon
    hw = dev / "hwmon" / "hwmon0"
    hw.mkdir(parents=True)
    (hw / "name").write_text(hwmon_name + "\n")
    if temp2 is not None:
        (hw / "temp2_input").write_text(f"{temp2}\n")
        (hw / "temp2_label").write_text("pkg\n")
    if temp1 is not None:
        (hw / "temp1_input").write_text(f"{temp1}\n")
    (hw / "temp3_input").write_text("52000\n")
    (hw / "temp3_label").write_text("vram\n")
    (hw / "energy1_input").write_text(f"{energy_uj}\n")
    (hw / "energy1_label").write_text("card\n")
    (hw / "power1_cap").write_text("275000000\n")  # micro-W -> 275 W
    (hw / "fan1_input").write_text("1060\n")  # RPM
    return card


def test_read_intel_sysfs_temp_freq_fan_limit(tmp_path: Path):
    card = _make_intel_card(tmp_path)
    g, energy = read_intel_sysfs(str(card), prev_energy=None, now=100.0)
    assert g.vendor == "intel"
    assert g.clock_sm_mhz == 2800
    assert g.temp_c == 57.0  # temp2_input (pkg) / 1000
    assert g.power_limit_w == 275.0
    assert g.fan_rpm == 1060
    # util and VRAM are not available on the xe driver
    assert g.util_gpu is None
    assert g.mem_used is None and g.mem_total is None
    # energy carried forward for the next power delta
    assert energy == (84023575462341, 100.0)


def test_read_intel_sysfs_power_none_on_first_call(tmp_path: Path):
    card = _make_intel_card(tmp_path)
    g, _ = read_intel_sysfs(str(card), prev_energy=None, now=100.0)
    assert g.power_w is None  # need two samples for a delta


def test_read_intel_sysfs_power_from_energy_delta(tmp_path: Path):
    # 60 J consumed over 2 s -> 30 W.  60 J == 60_000_000 µJ.
    e1 = 84_000_000_000_000
    card = _make_intel_card(tmp_path, energy_uj=e1 + 60_000_000)
    g, energy = read_intel_sysfs(str(card), prev_energy=(e1, 100.0), now=102.0)
    assert g.power_w is not None
    assert abs(g.power_w - 30.0) < 1e-6
    assert energy == (e1 + 60_000_000, 102.0)


def test_read_intel_sysfs_temp1_fallback(tmp_path: Path):
    # No temp2_input -> fall back to temp1_input.
    card = _make_intel_card(tmp_path, temp2=None, temp1=49000)
    g, _ = read_intel_sysfs(str(card), prev_energy=None, now=1.0)
    assert g.temp_c == 49.0


def test_read_intel_sysfs_zero_dt_yields_no_power(tmp_path: Path):
    e1 = 84_000_000_000_000
    card = _make_intel_card(tmp_path, energy_uj=e1 + 1_000_000)
    g, _ = read_intel_sysfs(str(card), prev_energy=(e1, 100.0), now=100.0)
    assert g.power_w is None  # dt == 0 must not divide-by-zero


def test_read_intel_sysfs_missing_energy_keeps_prev(tmp_path: Path):
    """If energy can't be read, power is None and carry stays None (no crash)."""
    card = tmp_path / "card0"
    (card / "device").mkdir(parents=True)
    g, energy = read_intel_sysfs(str(card), prev_energy=(123, 1.0), now=2.0)
    assert g.power_w is None
    assert energy is None


def test_pdev_for_card_resolves_pci_address(tmp_path: Path):
    # card0/device -> symlink into the PCI tree; basename of its realpath is pdev.
    pci = tmp_path / "sys" / "devices" / "pci0000:00" / "0000:06:00.0"
    pci.mkdir(parents=True)
    card = tmp_path / "sys" / "class" / "drm" / "card0"
    card.mkdir(parents=True)
    (card / "device").symlink_to(pci)
    assert pdev_for_card(str(card)) == "0000:06:00.0"


def test_pdev_for_card_missing_device_returns_none_or_str(tmp_path: Path):
    # A card dir with no device link must not raise.
    card = tmp_path / "card0"
    card.mkdir()
    # realpath of a non-existent path yields its basename ("device"); never raises.
    assert pdev_for_card(str(card)) in ("device", None)


def _make_gtidle_card(tmp_path: Path, gts: dict[str, int]) -> Path:
    """Build a fake card tree with ``device/<gt>/gtidle/idle_residency_ms``.

    ``gts`` maps a relative GT subpath (e.g. ``"tile0/gt0"``) to its initial
    cumulative idle-residency in ms.
    """
    card = tmp_path / "card0"
    for rel, idle_ms in gts.items():
        gtidle = card / "device" / Path(rel) / "gtidle"
        gtidle.mkdir(parents=True)
        (gtidle / "idle_residency_ms").write_text(f"{idle_ms}\n")
    return card


def _set_gtidle(card: Path, rel: str, idle_ms: int) -> None:
    (card / "device" / rel / "gtidle" / "idle_residency_ms").write_text(f"{idle_ms}\n")


def test_read_gtidle_util_none_on_first_call(tmp_path: Path):
    # No previous sample -> util is None, but the idle dict is captured.
    card = _make_gtidle_card(tmp_path, {"tile0/gt0": 5000})
    util, new_idle, t = read_gtidle_util(str(card), prev_idle=None, now=100.0, prev_now=None)
    assert util is None
    assert t == 100.0
    # the gt path was recorded so the next call has a baseline
    assert len(new_idle) == 1
    assert next(iter(new_idle.values())) == 5000


def test_read_gtidle_util_partial_idle_gives_load(tmp_path: Path):
    # idle advanced 300ms over a 1000ms wall window -> 70% busy.
    card = _make_gtidle_card(tmp_path, {"tile0/gt0": 5000})
    _, idle1, t1 = read_gtidle_util(str(card), prev_idle=None, now=100.0, prev_now=None)
    _set_gtidle(card, "tile0/gt0", 5300)
    util, _, t2 = read_gtidle_util(str(card), prev_idle=idle1, now=101.0, prev_now=t1)
    assert util == 70.0
    assert t2 == 101.0


def test_read_gtidle_util_fully_idle_gives_zero(tmp_path: Path):
    # idle advances by >= wall -> clamped to 0% busy.
    card = _make_gtidle_card(tmp_path, {"tile0/gt0": 5000})
    _, idle1, t1 = read_gtidle_util(str(card), prev_idle=None, now=100.0, prev_now=None)
    _set_gtidle(card, "tile0/gt0", 6200)  # +1200ms idle over a 1000ms wall window
    util, _, _ = read_gtidle_util(str(card), prev_idle=idle1, now=101.0, prev_now=t1)
    assert util == 0.0


def test_read_gtidle_util_no_idle_progress_gives_full(tmp_path: Path):
    # idle unchanged across the window -> 100% busy.
    card = _make_gtidle_card(tmp_path, {"tile0/gt0": 5000})
    _, idle1, t1 = read_gtidle_util(str(card), prev_idle=None, now=100.0, prev_now=None)
    _set_gtidle(card, "tile0/gt0", 5000)  # no change
    util, _, _ = read_gtidle_util(str(card), prev_idle=idle1, now=101.0, prev_now=t1)
    assert util == 100.0


def test_read_gtidle_util_multi_gt_takes_busiest(tmp_path: Path):
    # gt0 fully idle (0% busy), gt1 30% idle (70% busy) -> max wins -> 70.0.
    card = _make_gtidle_card(tmp_path, {"tile0/gt0": 5000, "tile0/gt1": 8000})
    _, idle1, t1 = read_gtidle_util(str(card), prev_idle=None, now=100.0, prev_now=None)
    _set_gtidle(card, "tile0/gt0", 6000)  # +1000ms idle / 1000ms wall -> 0% busy
    _set_gtidle(card, "tile0/gt1", 8300)  # +300ms idle / 1000ms wall -> 70% busy
    util, _, _ = read_gtidle_util(str(card), prev_idle=idle1, now=101.0, prev_now=t1)
    assert util == 70.0


def test_read_gtidle_util_no_gt_paths_returns_none(tmp_path: Path):
    # A card with no gtidle counters must not raise; util stays None.
    card = tmp_path / "card0"
    (card / "device").mkdir(parents=True)
    util, new_idle, t = read_gtidle_util(str(card), prev_idle=None, now=100.0, prev_now=None)
    assert util is None
    assert new_idle == {}
    assert t == 100.0


def test_read_gtidle_util_zero_wall_returns_none(tmp_path: Path):
    # dwall == 0 must not divide-by-zero; util is None.
    card = _make_gtidle_card(tmp_path, {"tile0/gt0": 5000})
    _, idle1, t1 = read_gtidle_util(str(card), prev_idle=None, now=100.0, prev_now=None)
    _set_gtidle(card, "tile0/gt0", 5000)
    util, _, _ = read_gtidle_util(str(card), prev_idle=idle1, now=100.0, prev_now=t1)
    assert util is None


def _write_resource(card: Path, lines: list[str]) -> None:
    dev = card / "device"
    dev.mkdir(parents=True, exist_ok=True)
    (dev / "resource").write_text("\n".join(lines) + "\n")


def test_read_pci_vram_total_largest_prefetchable_bar(tmp_path: Path):
    # A 32 GiB prefetchable BAR (flags bit 0x2000 set) plus smaller / non-pref
    # BARs. The 32 GiB region's size is end - start + 1 == 32 * 2**30.
    card = tmp_path / "card0"
    size = 32 * 2**30
    start = 0x3800000000
    end = start + size - 1
    lines = [
        # mmio control BAR, prefetchable but tiny
        "0x00000000d0000000 0x00000000d0ffffff 0x0000000000040200",
        # the big prefetchable VRAM BAR (0x...0200 has the 0x0200 + 0x2000 bits)
        f"0x{start:016x} 0x{end:016x} 0x000000000014220c",
        # expansion ROM, non-prefetchable, must be ignored
        "0x00000000d1000000 0x00000000d107ffff 0x0000000000000000",
        # an empty BAR (start==end==0)
        "0x0000000000000000 0x0000000000000000 0x0000000000000000",
    ]
    _write_resource(card, lines)
    total = read_pci_vram_total(str(card))
    assert total == size
    assert total is not None
    # sanity: matches the documented hardware figure ~34.36 GB
    assert abs(total / 1e9 - 34.36) < 0.01


def test_read_pci_vram_total_ignores_non_prefetchable(tmp_path: Path):
    # Largest region is NOT prefetchable -> ignored; only the small pref BAR counts.
    card = tmp_path / "card0"
    lines = [
        # huge non-prefetchable region (no 0x2000 bit)
        "0x0000000000000000 0x00000000ffffffff 0x0000000000040200",
        # small prefetchable region (0x2000 set) -> 0x1000 bytes
        "0x00000000d0000000 0x00000000d0000fff 0x000000000014220c",
    ]
    _write_resource(card, lines)
    assert read_pci_vram_total(str(card)) == 0x1000


def test_read_pci_vram_total_none_when_no_prefetchable(tmp_path: Path):
    card = tmp_path / "card0"
    lines = [
        "0x00000000d0000000 0x00000000d0ffffff 0x0000000000040200",  # no 0x2000 bit
    ]
    _write_resource(card, lines)
    assert read_pci_vram_total(str(card)) is None


def test_read_pci_vram_total_missing_file_returns_none(tmp_path: Path):
    card = tmp_path / "card0"
    (card / "device").mkdir(parents=True)
    # no resource file at all -> None, never raises
    assert read_pci_vram_total(str(card)) is None


def test_read_pci_vram_total_malformed_lines_skipped(tmp_path: Path):
    card = tmp_path / "card0"
    lines = [
        "garbage not three hex fields",
        "0x1 0x2",  # too few columns
        "0xNOTHEX 0xVALUES 0x2000",  # unparseable
        # one valid 4 KiB prefetchable region
        "0x00000000d0000000 0x00000000d0000fff 0x0000000000002000",
    ]
    _write_resource(card, lines)
    assert read_pci_vram_total(str(card)) == 0x1000
