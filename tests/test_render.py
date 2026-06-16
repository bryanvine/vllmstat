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


def test_concurrency_panel_shows_peak_high_water_mark():
    text = render.concurrency(
        _snap(running=2.0, waiting=1.0, peak_running=8.0, peak_waiting=3.0), History(), width=40
    )
    assert "running 2 (peak 8)" in text
    assert "waiting 1 (peak 3)" in text


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


def test_advisor_renders_markers_and_empty():
    from vllmstat.core.advisor import Issue

    assert render.advisor([]) == ""  # all clear -> hidden
    out = render.advisor(
        [
            Issue("error", "kv_preemption", "KV cache exhausted — 2.3 preemptions/s."),
            Issue("warn", "prefix_caching_off", "Prefix caching is disabled."),
        ]
    )
    assert "CONFIG ADVISOR" in out
    assert "✖" in out and "⚠" in out
    assert "KV cache exhausted" in out and "Prefix caching is disabled" in out


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


def _fleet_fixture():
    from vllmstat.core.state import (
        FleetSnapshot,
        GpuSample,
        GpuSnapshot,
        Instance,
        Quantiles,
        Snapshot,
    )

    up = Snapshot(
        ts=1.0,
        connected=True,
        running=12,
        waiting=3,
        gen_tps=1400,
        kv_usage=0.63,
        ttft=Quantiles(p50=0.142),
        gpu=GpuSnapshot(
            available=True,
            source="x",
            gpus=[GpuSample(index=0, name="Arc", vendor="intel", util_gpu=100.0)],
        ),
    )
    down = Snapshot(ts=1.0, connected=False)
    return FleetSnapshot(
        ts=1.0,
        items=[
            (Instance("qwen-30b", "http://localhost:8000", gpus=(0,), locality="local"), up),
            (Instance("remote-a", "http://gpu-box:8000", locality="remote"), down),
        ],
    )


def test_fleet_overview_rows_and_cursor():
    from vllmstat import render

    out = render.fleet_overview(_fleet_fixture(), 0, width=80, uptime="0h03m")
    lines = out.splitlines()
    assert "fleet" in lines[0]
    assert "▸" in next(ln for ln in lines if "qwen-30b" in ln)  # selected cursor
    remote_line = next(ln for ln in lines if "remote-a" in ln)
    assert "✗" in remote_line and "(remote)" in remote_line  # down + remote gpu cell
    assert "intel" in next(ln for ln in lines if "qwen-30b" in ln)  # local gpu cell


def test_fleet_overview_none_safe_selection_out_of_range():
    from vllmstat import render

    render.fleet_overview(_fleet_fixture(), 99, width=80)  # must not raise


def test_detail_header_breadcrumb():
    from vllmstat import render
    from vllmstat.core.state import Instance, Snapshot

    h = render.detail_header(
        Instance("qwen-30b", "http://localhost:8000"),
        Snapshot(ts=1.0, connected=True),
        interval=1.0,
        uptime="0h01m",
    )
    assert "qwen-30b" in h and "esc back" in h


def test_render_tee_http_and_empty():
    from vllmstat import render
    from vllmstat.core.tee import TeeEvent

    assert "waiting" in render.tee([], width=60, source_desc="docker:x")
    out = render.tee(
        [TeeEvent(ts=0.0, kind="http", method="POST", path="/v1/chat/completions", status=200)],
        width=60,
        source_desc="docker:x",
    )
    assert "TEE" in out and "POST" in out and "/v1/chat/completions" in out and "200" in out


def test_render_tee_marks_errors_and_exchange():
    from vllmstat import render
    from vllmstat.core.tee import TeeEvent

    err = render.tee([TeeEvent(ts=0.0, kind="http", method="GET", path="/x", status=503)], width=40)
    assert "!" in err
    ex = render.tee(
        [TeeEvent(ts=0.0, kind="exchange", prompt="hi there", response="hello", done=False)],
        width=40,
    )
    assert "▶" in ex and "◀" in ex


def test_latency_includes_phases():
    from vllmstat import render
    from vllmstat.core.state import Quantiles, Snapshot

    s = Snapshot(
        ts=0.0,
        connected=True,
        prefill=Quantiles(p50=0.09, p90=0.26),
        decode=Quantiles(p50=0.21, p90=0.54),
    )
    out = render.latency(s)
    assert "prefill" in out and "decode" in out


def test_request_shape_and_empty():
    from vllmstat import render
    from vllmstat.core.state import Quantiles, Snapshot

    assert render.request_shape(Snapshot(ts=0.0, connected=True)) == ""
    s = Snapshot(
        ts=0.0,
        connected=True,
        prompt_len=Quantiles(p50=1400, p90=6800, mean=2100),
        gen_len=Quantiles(p50=256, p90=900, mean=320),
    )
    out = render.request_shape(s)
    assert "REQUEST SHAPE" in out and "prompt" in out and "avg" in out and "gen" in out


def test_context_window_full_with_headroom():
    s = _snap(max_prompt_tokens=4096, max_output_tokens=512, max_model_len=32768)
    out = render.context_window(s)
    assert "MAX CONTEXT" in out
    assert "prompt ≤4096" in out and "output ≤512" in out
    assert "total ≤4608" in out  # prompt + output upper bound
    assert "32768" in out and "14.1%" in out  # headroom vs configured max-model-len


def test_context_window_no_max_model_len_omits_headroom():
    out = render.context_window(_snap(max_prompt_tokens=4096, max_output_tokens=512))
    assert "total ≤4608" in out
    assert "%" not in out  # no configured cap -> no headroom percentage


def test_context_window_empty_when_no_data():
    assert render.context_window(_snap(max_prompt_tokens=None, max_output_tokens=None)) == ""


def test_outcomes_and_empty():
    from vllmstat import render
    from vllmstat.core.state import Snapshot

    assert render.outcomes(Snapshot(ts=0.0, connected=True)) == ""
    s = Snapshot(
        ts=0.0,
        connected=True,
        finish_reasons={"stop": 0.92, "length": 0.08},
        goodput_ttft=0.88,
        goodput_tpot=0.94,
    )
    out = render.outcomes(s)
    assert "stop" in out and "length" in out and "goodput" in out and "TTFT<" in out


def test_efficiency_shows_session_averages():
    from vllmstat import render
    from vllmstat.core.state import Snapshot

    out = render.efficiency(
        Snapshot(ts=0.0, connected=True, tokens_per_watt=5.9, joules_per_token=0.17)
    )
    assert "5.9 tok/W" in out and "0.17 J/tok" in out


def test_efficiency_holds_average_when_idle():
    # The held session means stay visible (frozen) when the server goes idle: gen_tps has
    # decayed to an EWMA residual, but tok/W and J/tok must NOT recompute (no blowup) or vanish.
    from vllmstat import render
    from vllmstat.core.state import Snapshot

    s = Snapshot(
        ts=0.0,
        connected=True,
        gen_tps=0.002,  # idle residual
        eff_active=False,
        tokens_per_watt=6.0,
        joules_per_token=0.17,
    )
    out = render.efficiency(s)
    assert "6.0 tok/W" in out and "0.17 J/tok" in out  # shown, frozen at the session mean


def test_efficiency_hidden_when_no_average_yet():
    # No active sample recorded yet and nothing else to show -> panel stays empty.
    from vllmstat import render
    from vllmstat.core.state import Snapshot

    assert render.efficiency(Snapshot(ts=0.0, connected=True, gen_tps=0.002)) == ""


def test_efficiency_shows_idle_watts():
    from vllmstat import render
    from vllmstat.core.state import Snapshot

    out = render.efficiency(Snapshot(ts=0.0, connected=True, idle_watts_avg=32.0))
    assert "idle 32 W" in out
