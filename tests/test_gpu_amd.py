from __future__ import annotations

import json
from pathlib import Path

from vllmstat.providers.gpu_amd import parse_amd_smi_json, read_amd_sysfs


def _make_amd_card(tmp_path: Path) -> Path:
    card = tmp_path / "card0"
    dev = card / "device"
    dev.mkdir(parents=True)
    (dev / "gpu_busy_percent").write_text("42\n")
    (dev / "mem_info_vram_used").write_text("8000000000\n")
    (dev / "mem_info_vram_total").write_text("17163091968\n")
    hw = dev / "hwmon" / "hwmon0"
    hw.mkdir(parents=True)
    (hw / "name").write_text("amdgpu\n")
    (hw / "temp1_input").write_text("48000\n")  # milli-degC -> 48 C
    (hw / "power1_average").write_text("123000000\n")  # micro-W -> 123 W
    (hw / "power1_cap").write_text("250000000\n")  # micro-W -> 250 W
    (hw / "fan1_input").write_text("1800\n")  # RPM
    (hw / "freq1_input").write_text("2100000000\n")  # Hz -> 2100 MHz
    return card


def test_read_amd_sysfs_full(tmp_path: Path):
    card = _make_amd_card(tmp_path)
    g = read_amd_sysfs(str(card))
    assert g.vendor == "amd"
    assert g.util_gpu == 42.0
    assert g.mem_used == 8000000000
    assert g.mem_total == 17163091968
    assert g.temp_c == 48.0
    assert g.power_w == 123.0
    assert g.power_limit_w == 250.0
    assert g.fan_rpm == 1800
    assert g.clock_sm_mhz == 2100


def test_read_amd_sysfs_missing_fields_degrade(tmp_path: Path):
    card = tmp_path / "card0"
    dev = card / "device"
    dev.mkdir(parents=True)
    (dev / "gpu_busy_percent").write_text("7\n")  # only util present
    g = read_amd_sysfs(str(card))  # must not raise
    assert g.vendor == "amd"
    assert g.util_gpu == 7.0
    assert g.mem_used is None
    assert g.mem_total is None
    assert g.temp_c is None
    assert g.power_w is None
    assert g.fan_rpm is None
    assert g.clock_sm_mhz is None


def test_read_amd_sysfs_freq_fallback_to_pp_dpm_sclk(tmp_path: Path):
    card = tmp_path / "card0"
    dev = card / "device"
    dev.mkdir(parents=True)
    # No hwmon freq1_input; sclk comes from pp_dpm_sclk active line ("*").
    (dev / "pp_dpm_sclk").write_text("0: 500Mhz\n1: 1500Mhz *\n2: 2200Mhz\n")
    g = read_amd_sysfs(str(card))
    assert g.clock_sm_mhz == 1500


def test_read_amd_sysfs_picks_amdgpu_hwmon_by_name(tmp_path: Path):
    """A card may have several hwmon nodes; we read temp/power from any of them."""
    card = tmp_path / "card0"
    dev = card / "device"
    dev.mkdir(parents=True)
    # an unrelated hwmon (e.g. a fan controller) plus the amdgpu one
    other = dev / "hwmon" / "hwmon3"
    other.mkdir(parents=True)
    (other / "name").write_text("nvme\n")
    amd = dev / "hwmon" / "hwmon5"
    amd.mkdir(parents=True)
    (amd / "name").write_text("amdgpu\n")
    (amd / "temp1_input").write_text("55000\n")
    g = read_amd_sysfs(str(card))
    assert g.temp_c == 55.0


def test_parse_amd_smi_json_value_unit_shape():
    payload = [
        {
            "gpu": 0,
            "usage": {"gfx_activity": {"value": 73, "unit": "%"}},
            "mem_usage": {
                "used_vram": {"value": 8192, "unit": "MB"},
                "total_vram": {"value": 16368, "unit": "MB"},
            },
            "temperature": {
                "edge": {"value": 49, "unit": "C"},
                "hotspot": {"value": 61, "unit": "C"},
            },
            "power": {
                "socket_power": {"value": 142, "unit": "W"},
                "power_cap": {"value": 300, "unit": "W"},
            },
        }
    ]
    gpus = parse_amd_smi_json(json.dumps(payload))
    assert len(gpus) == 1
    g = gpus[0]
    assert g.index == 0
    assert g.vendor == "amd"
    assert g.util_gpu == 73.0
    assert g.mem_used == 8192 * 1024 * 1024
    assert g.mem_total == 16368 * 1024 * 1024
    assert g.temp_c == 49.0
    assert g.power_w == 142.0
    assert g.power_limit_w == 300.0


def test_parse_amd_smi_json_flat_scalar_shape():
    """Older/rocm-smi-ish shape: plain numbers, no {value, unit} wrapper."""
    payload = {
        "card0": {
            "GPU use (%)": "55",
            "VRAM Total Memory (B)": "17163091968",
            "VRAM Total Used Memory (B)": "9000000000",
            "Temperature (Sensor edge) (C)": "44.0",
            "Average Graphics Package Power (W)": "98.0",
        }
    }
    gpus = parse_amd_smi_json(json.dumps(payload))
    assert len(gpus) == 1
    g = gpus[0]
    assert g.util_gpu == 55.0
    assert g.mem_total == 17163091968
    assert g.mem_used == 9000000000
    assert g.temp_c == 44.0
    assert g.power_w == 98.0


def test_parse_amd_smi_json_garbage_returns_empty():
    assert parse_amd_smi_json("not json") == []
    assert parse_amd_smi_json("") == []
    assert parse_amd_smi_json("123") == []
