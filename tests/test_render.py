from vllmstat import render
from vllmstat.core.history import History
from vllmstat.core.state import GpuSample, GpuSnapshot, Snapshot


def _snap(**kw) -> Snapshot:
    base = dict(
        ts=0.0,
        connected=True,
        model_names=["m"],
        engine_count=1,
        gen_tps=142.0,
        prompt_tps=318.0,
        running=2.0,
        waiting=0.0,
        kv_dtype="turboquant_k3v4_nc",
        kv_capacity_tokens=444608,
        kv_ratio=32 / 7,
        kv_ratio_kind="nominal",
        kv_usage=0.0009,
        prefix_hit_lifetime=0.315,
        prefix_hit_window=0.381,
        src_compute=0.69,
        src_cache_hit=0.31,
        src_external=0.0,
        spec_active=True,
        spec_acceptance=0.398,
        spec_accepted_per_draft=2.16,
    )
    base.update(kw)
    return Snapshot(**base)  # type: ignore[arg-type]


def test_cache_kv_panel_shows_dtype_and_ratio():
    text = render.cache_kv(_snap(), History())
    assert "turboquant_k3v4_nc" in text
    assert "4.6x" in text or "4.57x" in text
    assert "444" in text  # capacity shown (e.g. 445k or 444,608)


def test_throughput_panel_shows_tps():
    text = render.throughput(_snap(), History())
    assert "142" in text


def test_gpu_panel_unavailable_message():
    s = _snap(gpu=GpuSnapshot(available=False, source="none", error="no NVML"))
    text = render.gpu(s)
    assert "unavailable" in text.lower() or "no nvml" in text.lower()


def test_gpu_panel_renders_device():
    s = _snap(
        gpu=GpuSnapshot(
            available=True,
            source="nvml",
            gpus=[
                GpuSample(
                    index=0,
                    name="NVIDIA Test",
                    util_gpu=81,
                    mem_used=23_100_000_000,
                    mem_total=24_000_000_000,
                    temp_c=61,
                    power_w=142,
                    power_limit_w=200,
                    clock_sm_mhz=2520,
                    clock_mem_mhz=9501,
                    fan_pct=45,
                )
            ],
        )
    )
    text = render.gpu(s)
    assert "NVIDIA Test" in text and "81" in text and "61" in text


def test_specdecode_hidden_when_inactive():
    assert render.specdecode(_snap(spec_active=False)) == ""


def test_gpu_panel_handles_all_none_optional_fields():
    from vllmstat.core.state import GpuSample, GpuSnapshot

    s = _snap(
        gpu=GpuSnapshot(
            available=True,
            source="nvidia-smi",
            gpus=[
                GpuSample(
                    index=0,
                    name="GPU X",
                    util_gpu=None,
                    mem_used=None,
                    mem_total=None,
                    temp_c=None,
                    power_w=None,
                    power_limit_w=None,
                    clock_sm_mhz=None,
                    clock_mem_mhz=None,
                    fan_pct=None,
                )
            ],
        )
    )
    text = render.gpu(s)  # must NOT raise
    assert "GPU X" in text
    assert "—" in text  # missing values shown as em dash


def test_specdecode_handles_none_accepted_per_draft():
    s = _snap(spec_active=True, spec_acceptance=0.4, spec_accepted_per_draft=None)
    text = render.specdecode(s)  # must NOT raise
    assert "acceptance" in text


def test_gpu_panel_intel_style_sample_shows_name_temp_and_hint():
    s = _snap(
        gpu=GpuSnapshot(
            available=True,
            source="intel-sysfs",
            gpus=[
                GpuSample(
                    index=0,
                    name="Intel Arc B-series (Battlemage)",
                    vendor="intel",
                    util_gpu=None,
                    mem_used=None,
                    mem_total=None,
                    temp_c=57.0,
                    power_w=116.0,
                    power_limit_w=275.0,
                    fan_rpm=1060,
                    clock_sm_mhz=2800,
                )
            ],
        )
    )
    text = render.gpu(s)  # must NOT raise
    assert "Intel Arc B-series (Battlemage)" in text
    assert "intel" in text.lower()
    assert "57" in text  # temperature shown
    assert "2800" in text  # clock shown
    assert "1060" in text and "rpm" in text.lower()  # fan as RPM
    assert "—" in text  # util / VRAM shown as em dash
    # both util and VRAM missing (no root) -> the root hint is appended
    assert "need root" in text.lower()
    assert "readme" in text.lower()


def test_gpu_panel_intel_with_fdinfo_util_vram_no_hint():
    # Intel sample with real fdinfo util/VRAM but unknown total -> no hint,
    # VRAM rendered as "<used>/—" (None-safe total).
    s = _snap(
        gpu=GpuSnapshot(
            available=True,
            source="intel-sysfs",
            gpus=[
                GpuSample(
                    index=0,
                    name="Intel Arc B-series (Battlemage)",
                    vendor="intel",
                    util_gpu=37.0,
                    mem_used=31_645_832 * 1024,
                    mem_total=None,
                    temp_c=57.0,
                    power_w=116.0,
                    power_limit_w=275.0,
                    fan_rpm=1060,
                    clock_sm_mhz=2800,
                )
            ],
        )
    )
    text = render.gpu(s)
    assert "37" in text  # util% shown
    assert "32.4 GB" in text  # VRAM used shown (31_645_832 KiB ~= 32.4 GB)
    assert "need root" not in text.lower()  # util/VRAM present -> no hint
    # total unknown -> shown as em dash on the right of the slash
    assert "32.4 GB/—" in text


def test_gpu_panel_amd_rpm_fan_and_no_hint_when_util_present():
    s = _snap(
        gpu=GpuSnapshot(
            available=True,
            source="amdgpu-sysfs",
            gpus=[
                GpuSample(
                    index=0,
                    name="amd GPU 0x744c",
                    vendor="amd",
                    util_gpu=42.0,
                    mem_used=8_000_000_000,
                    mem_total=17_000_000_000,
                    temp_c=48.0,
                    power_w=123.0,
                    power_limit_w=250.0,
                    fan_rpm=1800,
                    clock_sm_mhz=2100,
                )
            ],
        )
    )
    text = render.gpu(s)
    assert "1800" in text and "RPM" in text
    assert "42" in text
    assert "prereq" not in text.lower()  # util+VRAM present -> no hint
