from vllmstat.core.state import GpuSample, GpuSnapshot, Quantiles, Snapshot


def test_quantiles_defaults_none():
    q = Quantiles()
    assert q.p50 is None and q.p90 is None and q.p99 is None and q.mean is None


def test_gpu_snapshot_unavailable():
    snap = GpuSnapshot(available=False, source="none", gpus=[], error="no nvml")
    assert snap.available is False
    assert snap.gpus == []


def test_snapshot_minimal_construction():
    s = Snapshot(ts=1.0, connected=True)
    assert s.connected is True
    assert s.running == 0.0
    assert s.spec_active is False
    assert s.gpu.available is False


def test_gpu_sample_vendor_and_fan_rpm_defaults():
    g = GpuSample(index=0, name="GPU")
    assert g.vendor == ""
    assert g.fan_rpm is None
    assert g.fan_pct is None


def test_gpu_sample_accepts_vendor_and_fan_rpm():
    g = GpuSample(index=1, name="Intel Arc", vendor="intel", fan_rpm=1060)
    assert g.vendor == "intel"
    assert g.fan_rpm == 1060


def test_instance_defaults():
    from vllmstat.core.state import Instance

    i = Instance(name="a", url="http://localhost:8000")
    assert (i.metrics_path, i.api_key, i.gpus, i.locality) == ("/metrics", None, (), "local")


def test_fleet_snapshot_defaults():
    from vllmstat.core.state import FleetSnapshot

    fs = FleetSnapshot(ts=1.0)
    assert fs.items == [] and fs.gpu.available is False


def test_instance_logs_default_none():
    from vllmstat.core.state import Instance

    assert Instance(name="a", url="http://x").logs is None


def test_snapshot_v06_metric_defaults():
    from vllmstat.core.state import Quantiles, Snapshot

    s = Snapshot(ts=0.0, connected=True)
    assert s.prefill == Quantiles() and s.decode == Quantiles()
    assert s.prompt_len == Quantiles() and s.gen_len == Quantiles()
    assert s.finish_reasons == {}
    assert s.goodput_ttft is None and s.goodput_tpot is None
    assert s.ttft_slo_s == 1.0 and s.tpot_slo_s == 0.05
