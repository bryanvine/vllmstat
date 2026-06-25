from __future__ import annotations

from pathlib import Path

from vllmstat.providers.gpu_sysfs import (
    Card,
    detect_cards,
    pci_name,
    read_int,
    read_text,
)


def _make_card(drm_root: Path, name: str, *, vendor: str, device: str | None = None) -> Path:
    card = drm_root / name
    dev = card / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor + "\n")
    if device is not None:
        (dev / "device").write_text(device + "\n")
    return card


def test_read_text_and_int_happy_path(tmp_path: Path):
    f = tmp_path / "val"
    f.write_text("  42 \n")
    assert read_text(str(f)) == "42"
    assert read_int(str(f)) == 42


def test_read_text_missing_returns_none(tmp_path: Path):
    assert read_text(str(tmp_path / "nope")) is None
    assert read_int(str(tmp_path / "nope")) is None


def test_read_text_swallows_non_oserror_from_read(tmp_path: Path, monkeypatch):
    """A sysfs read that raises a non-OSError must degrade to None, not crash.

    Reading some hwmon attributes on real hardware (e.g. Intel ``fan1_input``)
    makes the kernel feed ``None`` into the text decoder mid-read, surfacing as
    ``TypeError: can't concat NoneType to bytes`` from ``fh.read()`` — not an
    ``OSError``. ``read_text`` promises "None on any error", so it must catch it.
    """
    target = tmp_path / "fan1_input"
    target.write_text("0\n")
    real_open = open

    class _ExplodingFile:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise TypeError("can't concat NoneType to bytes")

    def fake_open(path, *args, **kwargs):
        if str(path) == str(target):
            return _ExplodingFile()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert read_text(str(target)) is None
    assert read_int(str(target)) is None


def test_read_int_non_numeric_returns_none(tmp_path: Path):
    f = tmp_path / "val"
    f.write_text("xe\n")
    assert read_int(str(f)) is None


def test_read_int_handles_hex(tmp_path: Path):
    f = tmp_path / "val"
    f.write_text("0x8086\n")
    assert read_int(str(f)) == 0x8086


def test_detect_single_intel_card(tmp_path: Path):
    drm = tmp_path / "drm"
    drm.mkdir()
    _make_card(drm, "card0", vendor="0x8086", device="0xe223")
    cards = detect_cards(str(drm))
    assert len(cards) == 1
    assert cards[0] == Card(index=0, path=str(drm / "card0"), vendor="intel")


def test_detect_mixed_vendors_sorted_by_index(tmp_path: Path):
    drm = tmp_path / "drm"
    drm.mkdir()
    _make_card(drm, "card1", vendor="0x1002")  # amd
    _make_card(drm, "card0", vendor="0x10de")  # nvidia
    _make_card(drm, "card2", vendor="0x8086")  # intel
    cards = detect_cards(str(drm))
    assert [(c.index, c.vendor) for c in cards] == [
        (0, "nvidia"),
        (1, "amd"),
        (2, "intel"),
    ]


def test_detect_ignores_connectors_and_render_nodes(tmp_path: Path):
    drm = tmp_path / "drm"
    drm.mkdir()
    _make_card(drm, "card0", vendor="0x8086")
    # connector dirs (no device/vendor) and a render node must be ignored
    (drm / "card0-DP-1").mkdir()
    (drm / "renderD128").mkdir()
    (drm / "version").write_text("drm 1.1.0\n")
    cards = detect_cards(str(drm))
    assert len(cards) == 1
    assert cards[0].index == 0


def test_detect_unknown_vendor_is_other(tmp_path: Path):
    drm = tmp_path / "drm"
    drm.mkdir()
    _make_card(drm, "card0", vendor="0x1234")
    cards = detect_cards(str(drm))
    assert cards[0].vendor == "other"


def test_detect_missing_drm_root_returns_empty(tmp_path: Path):
    assert detect_cards(str(tmp_path / "does-not-exist")) == []


def test_detect_card_without_vendor_file_skipped(tmp_path: Path):
    drm = tmp_path / "drm"
    drm.mkdir()
    (drm / "card0" / "device").mkdir(parents=True)  # no vendor file
    assert detect_cards(str(drm)) == []


def test_pci_name_known_intel_battlemage(tmp_path: Path):
    card = _make_card(tmp_path, "card0", vendor="0x8086", device="0xe223")
    name = pci_name(str(card))
    assert "Intel" in name and "Arc" in name


def test_pci_name_unknown_id_has_fallback(tmp_path: Path):
    card = _make_card(tmp_path, "card0", vendor="0x8086", device="0xffff")
    name = pci_name(str(card))
    # falls back to "<vendor> GPU <id>"
    assert "intel" in name.lower()
    assert "ffff" in name.lower()


def test_pci_name_missing_files_does_not_raise(tmp_path: Path):
    card = tmp_path / "card0"
    card.mkdir()
    name = pci_name(str(card))  # no device/ at all
    assert isinstance(name, str) and name  # non-empty, no crash
