from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from vllmstat.core.energy import (
    EnergyConfig,
    GpuEnergy,
    InstanceEnergy,
    TouRule,
    integrate_kwh,
    parse_energy_config,
    rate_at,
)


def test_integrate_kwh_trapezoidal():
    assert integrate_kwh(800.0, 1200.0, 3600.0) == pytest.approx(1.0)
    assert integrate_kwh(500.0, 500.0, 10.0) == pytest.approx(500.0 * 10 / 3600 / 1000)
    assert integrate_kwh(0.0, 0.0, 10.0) == 0.0


def test_parse_energy_config_flat_and_tou():
    cfg = parse_energy_config(
        {
            "currency": "£",
            "store": "/var/lib/vllmstat/x.db",
            "interval": 15,
            "retention_days": 14,
            "tou": [
                {"days": "mon-fri", "from": "16:00", "to": "21:00", "rate": 0.42, "label": "peak"},
                {"default": True, "rate": 0.12, "label": "off-peak"},
            ],
        }
    )
    assert cfg.currency == "£" and cfg.store.endswith("x.db")
    assert cfg.interval == 15.0 and cfg.retention_days == 14
    assert len(cfg.tou) == 2 and cfg.tou[1].default is True


def test_parse_energy_config_defaults():
    cfg = parse_energy_config({})
    assert cfg.currency == "$" and cfg.store is None
    assert cfg.interval == 10.0 and cfg.retention_days == 7 and cfg.tou == ()


def test_parse_energy_config_requires_default_when_tou_present():
    with pytest.raises(ValueError, match="default"):
        parse_energy_config(
            {"tou": [{"days": "mon-fri", "from": "9:00", "to": "17:00", "rate": 0.3}]}
        )


def test_parse_energy_config_rejects_bad_time_and_negative_rate():
    with pytest.raises(ValueError):
        parse_energy_config({"tou": [{"default": True, "rate": -1.0}]})
    with pytest.raises(ValueError):
        parse_energy_config(
            {"tou": [{"days": "mon-fri", "from": "25:00", "to": "9:00", "rate": 0.3},
                     {"default": True, "rate": 0.1}]}
        )


def test_rate_at_picks_window_then_default():
    cfg = parse_energy_config(
        {"tou": [
            {"days": "mon-fri", "from": "16:00", "to": "21:00", "rate": 0.42, "label": "peak"},
            {"default": True, "rate": 0.12, "label": "off-peak"},
        ]}
    )
    assert rate_at(cfg, datetime(2026, 6, 24, 18, 0)) == (0.42, "peak")
    assert rate_at(cfg, datetime(2026, 6, 24, 9, 0)) == (0.12, "off-peak")
    assert rate_at(cfg, datetime(2026, 6, 28, 18, 0)) == (0.12, "off-peak")


def test_rate_at_overnight_window_wraps_midnight():
    cfg = parse_energy_config(
        {"tou": [
            {"days": "mon-sun", "from": "21:00", "to": "07:00", "rate": 0.08, "label": "night"},
            {"default": True, "rate": 0.20, "label": "day"},
        ]}
    )
    assert rate_at(cfg, datetime(2026, 6, 24, 23, 30)) == (0.08, "night")
    assert rate_at(cfg, datetime(2026, 6, 24, 6, 0)) == (0.08, "night")
    assert rate_at(cfg, datetime(2026, 6, 24, 12, 0)) == (0.20, "day")


def test_rate_at_no_config_returns_none():
    assert rate_at(parse_energy_config({}), datetime(2026, 6, 24, 12, 0)) == (None, "")


def test_energy_carriers_are_frozen_dataclasses():
    g = GpuEnergy(gpu_idx=0, watts=200.0, kwh=0.01, cost=0.002)
    i = InstanceEnergy(instance="a", kwh=0.01, cost=0.002, tokens=50.0)
    assert g.gpu_idx == 0 and i.instance == "a"
    assert isinstance(parse_energy_config({}), EnergyConfig)
    assert TouRule(rate=0.1).default is False
    with pytest.raises(FrozenInstanceError):
        g.kwh = 1.0  # frozen
