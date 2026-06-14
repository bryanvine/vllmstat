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


def _has_braille(text: str) -> bool:
    return any(0x2800 <= ord(c) <= 0x28FF for c in text)


def test_throughput_panel_shows_tps_and_braille_plot():
    h = History()
    for v in [10, 50, 90, 142, 142]:
        h.push("gen_tps", v)
    for v in [20, 80, 200, 318, 318]:
        h.push("prompt_tps", v)
    text = render.throughput(_snap(), h, width=40)
    assert "142" in text  # numeric gen tok/s still shown
    assert "318" in text  # prompt still shown as text
    assert _has_braille(text)  # braille area plots present
    # both time series now have their own plot + caption
    assert "gen tok/s" in text
    assert "prompt tok/s" in text


def test_concurrency_panel_shows_counts_and_braille_plot():
    h = History()
    for v in [0, 1, 2, 3, 2]:
        h.push("running", v)
    for v in [0, 0, 1, 2, 0]:
        h.push("waiting", v)
    text = render.concurrency(_snap(running=2.0, waiting=0.0), h, width=40)
    assert "running 2" in text
    assert "waiting 0" in text
    assert _has_braille(text)
    assert "preempt" in text
    # both time series now have their own plot + caption
    assert "running" in text
    assert "waiting" in text


def test_timeseries_panels_no_braille_when_empty_history():
    # No samples yet -> braille rows are blank; must not raise.
    text_t = render.throughput(_snap(), History(), width=40)
    text_c = render.concurrency(_snap(), History(), width=40)
    assert "142" in text_t and "running" in text_c


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


def test_session_panel_shows_decode_prefill_active():
    s = _snap(
        avg_decode_tps=142.0,
        avg_prefill_tps=318.0,
        session_active_frac=0.8,
        session_active_s=723.0,  # 12m03s
        session_idle_s=42.0,
        session_requests=128,
        session_gen_tokens=48000.0,
        session_prompt_tokens=96000.0,
        avg_gen_tokens_per_req=375.0,
    )
    text = render.session(s)
    assert "SESSION" in text
    assert "142" in text  # decode avg
    assert "318" in text  # prefill/pp avg
    assert "prefill" in text.lower() and "pp" in text.lower()
    assert "80.0%" in text  # active fraction
    assert "12m03s" in text  # busy duration
    assert "42s" in text  # idle duration
    assert "128 reqs" in text
    assert "375" in text  # gen tok/req
    assert "48.0k" in text  # session gen totals (fmt_si)


def test_session_panel_none_safe_em_dash():
    # Fresh session: no averages yet -> em dashes, never raises.
    s = _snap(
        avg_decode_tps=None,
        avg_prefill_tps=None,
        session_active_frac=None,
        session_active_s=0.0,
        session_idle_s=0.0,
        session_requests=0,
        session_gen_tokens=0.0,
        session_prompt_tokens=0.0,
        avg_gen_tokens_per_req=None,
    )
    text = render.session(s)  # must NOT raise
    assert "SESSION" in text
    assert "—" in text


def test_gpu_panel_intel_mem_clock_none_shows_no_slash():
    # Intel xe has no mem clock -> show "clk 2800 MHz" (no "/—").
    s = _snap(
        gpu=GpuSnapshot(
            available=True,
            source="intel-sysfs",
            gpus=[
                GpuSample(
                    index=0,
                    name="Intel Arc B-series (Battlemage)",
                    vendor="intel",
                    util_gpu=70.0,
                    mem_used=20_000_000_000,
                    mem_total=34_359_738_368,  # 32 GiB total now known
                    temp_c=57.0,
                    power_w=116.0,
                    power_limit_w=275.0,
                    fan_rpm=1060,
                    clock_sm_mhz=2800,
                    clock_mem_mhz=None,
                )
            ],
        )
    )
    text = render.gpu(s)
    assert "clk 2800 MHz" in text
    assert "clk 2800/" not in text  # no slash when mem clock absent
    assert "2800/—" not in text
    # mem_total now set -> used/total + percent rendered (no em dash on total)
    assert "GB/—" not in text
    assert "%" in text  # mem percent present


def test_gpu_panel_both_clocks_shows_slash():
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
                    clock_sm_mhz=2520,
                    clock_mem_mhz=9501,
                )
            ],
        )
    )
    text = render.gpu(s)
    assert "clk 2520/9501 MHz" in text


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
    # both util and VRAM missing -> generic GPU-stats hint pointing at the README
    assert "gpu stats" in text.lower()
    assert "readme" in text.lower()


def test_gpu_panel_intel_util_present_vram_none_shows_vram_hint():
    # gtidle gives util without root; VRAM still root-gated -> only flag VRAM.
    s = _snap(
        gpu=GpuSnapshot(
            available=True,
            source="intel-sysfs",
            gpus=[
                GpuSample(
                    index=0,
                    name="Intel Arc B-series (Battlemage)",
                    vendor="intel",
                    util_gpu=70.0,
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
    text = render.gpu(s)
    assert "70" in text  # util present (gtidle, non-root)
    assert "vram needs root" in text.lower()  # only VRAM flagged
    assert "gpu stats" not in text.lower()  # not the both-missing hint


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
