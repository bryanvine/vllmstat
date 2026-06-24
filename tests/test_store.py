import pytest

from vllmstat.core.energy import GpuEnergy, InstanceEnergy
from vllmstat.core.store import Store


def _open(tmp_path):
    return Store.open(str(tmp_path / "e.db"))


def test_record_accumulates_totals(tmp_path):
    s = _open(tmp_path)
    ts = 1782648000.0
    s.record(ts, [GpuEnergy(0, 200.0, 0.01, 0.002)], [InstanceEnergy("a", 0.01, 0.002, 50.0)])
    s.record(ts + 10, [GpuEnergy(0, 220.0, 0.02, 0.004)], [InstanceEnergy("a", 0.02, 0.004, 60.0)])
    g = {r["gpu_idx"]: r for r in s.totals_gpu()}
    assert g[0]["kwh"] == pytest.approx(0.03)
    assert g[0]["cost"] == pytest.approx(0.006)
    i = {r["instance"]: r for r in s.totals_instance()}
    assert i["a"]["kwh"] == pytest.approx(0.03) and i["a"]["tokens"] == pytest.approx(110.0)
    s.close()


def test_read_view_today_and_alltime(tmp_path):
    s = _open(tmp_path)
    ts = 1782648000.0
    s.record(ts, [GpuEnergy(0, 200.0, 1.0, 0.10)], [InstanceEnergy("a", 1.0, 0.10)])
    view = s.read_view(now=ts, currency="$")
    assert view.available is True
    assert view.today_kwh == pytest.approx(1.0) and view.today_cost == pytest.approx(0.10)
    assert view.alltime_kwh == pytest.approx(1.0) and view.alltime_cost == pytest.approx(0.10)
    s.close()


def test_prune_drops_old_samples_keeps_daily_and_totals(tmp_path):
    s = _open(tmp_path)
    old = 1000.0
    s.record(old, [GpuEnergy(0, 100.0, 0.5, 0.05)], [InstanceEnergy("a", 0.5, 0.05)])
    s.record(1782648000.0, [GpuEnergy(0, 100.0, 0.5, 0.05)], [InstanceEnergy("a", 0.5, 0.05)])
    removed = s.prune(before_ts=1782000000.0)
    assert removed == 1
    assert s.sample_count() == 1
    assert s.totals_gpu()[0]["kwh"] == pytest.approx(1.0)
    assert s.daily_count() >= 1
    s.close()


def test_cost_none_when_rate_unset(tmp_path):
    s = _open(tmp_path)
    s.record(1782648000.0, [GpuEnergy(0, 100.0, 0.5, None)], [InstanceEnergy("a", 0.5, None)])
    assert s.totals_gpu()[0]["cost"] is None
    s.close()


def test_open_enables_wal(tmp_path):
    s = _open(tmp_path)
    mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    s.close()


def test_multi_gpu_accumulate_separately(tmp_path):
    s = _open(tmp_path)
    ts = 1782648000.0
    s.record(ts, [GpuEnergy(0, 100.0, 0.5, 0.05), GpuEnergy(1, 200.0, 1.0, 0.10)],
             [InstanceEnergy("a", 1.5, 0.15)])
    g = {r["gpu_idx"]: r for r in s.totals_gpu()}
    assert g[0]["kwh"] == pytest.approx(0.5) and g[1]["kwh"] == pytest.approx(1.0)
    s.close()


def test_concurrent_readonly_open(tmp_path):
    w = _open(tmp_path)
    w.record(1782648000.0, [GpuEnergy(0, 100.0, 0.5, 0.05)], [InstanceEnergy("a", 0.5, 0.05)])
    r = Store.open(str(tmp_path / "e.db"), read_only=True)
    assert r.totals_gpu()[0]["kwh"] == pytest.approx(0.5)
    r.close()
    w.close()
