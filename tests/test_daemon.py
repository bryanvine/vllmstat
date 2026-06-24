import pytest

from vllmstat.core.energy import EnergyConfig, parse_energy_config
from vllmstat.core.state import GpuSample, GpuSnapshot, Instance, Snapshot
from vllmstat.core.store import Store
from vllmstat.daemon import Collector


def _snap(running=1.0, gen_tokens=0.0, gpus=()):
    s = Snapshot(ts=0.0, connected=True, running=running)
    s.session_gen_tokens = gen_tokens
    s.gpu = GpuSnapshot(available=True, source="test",
                        gpus=[GpuSample(index=i, name="x", power_w=p) for i, p in gpus])
    return s


def test_collector_integrates_between_steps(tmp_path):
    store = Store.open(str(tmp_path / "e.db"))
    cfg = parse_energy_config({"tou": [{"default": True, "rate": 0.10}]})
    col = Collector(store, cfg)
    inst = Instance("a", "http://x", gpus=(0,), locality="local")
    col.step(1000.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    assert store.sample_count() == 0  # baseline only
    col.step(1000.0 + 3600.0, [(inst, _snap(gen_tokens=500.0, gpus=[(0, 1000.0)]))])
    g = store.totals_gpu()[0]
    assert g["kwh"] == pytest.approx(1.0) and g["cost"] == pytest.approx(0.10)
    i = store.totals_instance()[0]
    assert i["instance"] == "a" and i["kwh"] == pytest.approx(1.0)
    assert i["tokens"] == pytest.approx(500.0)
    store.close()


def test_collector_restart_does_not_integrate_gap(tmp_path):
    store = Store.open(str(tmp_path / "e.db"))
    cfg = parse_energy_config({"tou": [{"default": True, "rate": 0.10}]})
    inst = Instance("a", "http://x", gpus=(0,), locality="local")
    col1 = Collector(store, cfg)
    col1.step(1000.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    col2 = Collector(store, cfg)  # "restart": no baseline, long gap not integrated
    col2.step(100000.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    assert store.sample_count() == 0
    col2.step(100000.0 + 3600.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    assert store.totals_gpu()[0]["kwh"] == pytest.approx(1.0)
    store.close()


def test_collector_cost_none_when_no_schedule(tmp_path):
    store = Store.open(str(tmp_path / "e.db"))
    col = Collector(store, EnergyConfig())  # no TOU -> rate unset
    inst = Instance("a", "http://x", gpus=(0,), locality="local")
    col.step(0.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    col.step(3600.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    assert store.totals_gpu()[0]["cost"] is None
    store.close()


def test_collector_attributes_only_mapped_gpus_to_instance(tmp_path):
    store = Store.open(str(tmp_path / "e.db"))
    cfg = parse_energy_config({"tou": [{"default": True, "rate": 0.10}]})
    col = Collector(store, cfg)
    inst = Instance("a", "http://x", gpus=(1,), locality="local")  # only GPU 1
    snap = _snap(gpus=[(0, 1000.0), (1, 1000.0)])
    col.step(0.0, [(inst, snap)])
    col.step(3600.0, [(inst, _snap(gpus=[(0, 1000.0), (1, 1000.0)]))])
    # host has 2 GPUs at 1 kWh each; instance "a" maps only GPU 1 -> 1.0 kWh
    assert store.totals_instance()[0]["kwh"] == pytest.approx(1.0)
    gpu_kwh = {r["gpu_idx"]: r["kwh"] for r in store.totals_gpu()}
    assert gpu_kwh[0] == pytest.approx(1.0) and gpu_kwh[1] == pytest.approx(1.0)
    store.close()


def test_collector_clamps_token_counter_reset(tmp_path):
    store = Store.open(str(tmp_path / "e.db"))
    cfg = parse_energy_config({"tou": [{"default": True, "rate": 0.10}]})
    col = Collector(store, cfg)
    inst = Instance("a", "http://x", gpus=(0,), locality="local")
    col.step(0.0, [(inst, _snap(gen_tokens=1000.0, gpus=[(0, 1000.0)]))])
    # counter resets to a lower value (server restarted) -> token delta clamped to 0, not negative
    col.step(3600.0, [(inst, _snap(gen_tokens=5.0, gpus=[(0, 1000.0)]))])
    assert store.totals_instance()[0]["tokens"] == pytest.approx(0.0)
    store.close()


def test_collector_skips_remote_instances(tmp_path):
    store = Store.open(str(tmp_path / "e.db"))
    cfg = parse_energy_config({"tou": [{"default": True, "rate": 0.10}]})
    col = Collector(store, cfg)
    remote = Instance("r", "http://remote", locality="remote")
    rsnap = Snapshot(ts=0.0, connected=True)
    rsnap.gpu = GpuSnapshot(available=False, source="remote")
    col.step(0.0, [(remote, rsnap)])
    col.step(3600.0, [(remote, rsnap)])
    assert store.totals_instance() == []  # no energy for remote instances
    store.close()
