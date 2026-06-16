from pathlib import Path

from vllmstat.core.metrics import MetricsEngine
from vllmstat.core.parse import parse_metrics

FIX = Path(__file__).parent / "fixtures" / "metrics_qwen3.txt"
DIMS = {"layers": 48, "kv_heads": 4, "head_dim": 128}


def test_derive_from_golden_two_samples_for_rates():
    text = FIX.read_text()
    fam = parse_metrics(text)
    eng = MetricsEngine(dims=DIMS, max_model_len=262144)
    # First derive establishes baselines (rates 0)
    s0 = eng.derive(fam, now=0.0)
    assert s0.connected is True
    assert s0.gen_tps == 0.0
    # Second derive: simulate +1420 generation tokens over 10s -> ~142 tok/s eventually
    fam2 = dict(fam)
    # bump generation_tokens_total by 1420
    base = fam["vllm:generation_tokens_total"][0][1]
    fam2["vllm:generation_tokens_total"] = [
        (fam["vllm:generation_tokens_total"][0][0], base + 1420)
    ]
    s1 = eng.derive(fam2, now=10.0)
    assert s1.gen_tps > 0.0


def test_kv_and_cache_fields_present():
    fam = parse_metrics(FIX.read_text())
    eng = MetricsEngine(dims=DIMS, max_model_len=262144)
    s = eng.derive(fam, now=0.0)
    assert s.kv_dtype == "turboquant_k3v4_nc"
    assert s.kv_capacity_tokens == 6947 * 64
    assert s.kv_ratio_kind == "nominal"  # memory_bytes is None on fixture
    assert s.prefix_hit_lifetime is not None and 0.0 <= s.prefix_hit_lifetime <= 1.0
    # token sources: local_cache_hit + local_compute fractions sum ~1
    assert s.src_cache_hit is not None
    assert s.spec_active is True  # fixture has spec decode
    assert s.spec_accepted_per_draft and s.spec_accepted_per_draft > 1.0


def test_efficiency_hidden_when_zero():
    fam = parse_metrics(FIX.read_text())
    eng = MetricsEngine(dims=DIMS, max_model_len=262144)
    s = eng.derive(fam, now=0.0)
    assert s.eff_active is False  # estimated_* are 0 on fixture


def test_latency_quantiles_computed():
    fam = parse_metrics(FIX.read_text())
    eng = MetricsEngine(dims=DIMS, max_model_len=262144)
    eng.derive(fam, now=0.0)
    s = eng.derive(fam, now=1.0)  # same fixture -> windowed delta 0; falls back to lifetime
    # With zero window delta, engine uses cumulative buckets so p50 is defined
    assert s.ttft.p50 is not None


# --- session averages (accumulated while serving) -----------------------------


def _fam(*, gen: float, prompt: float, req: float, running: float):
    """Minimal synthetic Families with just the session-relevant counters."""
    e = {"engine": "0", "model_name": "m"}
    return {
        "vllm:generation_tokens_total": [(e, gen)],
        "vllm:prompt_tokens_total": [(e, prompt)],
        "vllm:request_success_total": [(e, req)],
        "vllm:num_requests_running": [(e, running)],
    }


def test_session_accumulates_active_and_idle():
    eng = MetricsEngine()
    # t=0: baseline (first sample, no accumulation yet)
    s0 = eng.derive(_fam(gen=1000.0, prompt=5000.0, req=10.0, running=2.0), now=0.0)
    assert s0.session_active_s == 0.0
    assert s0.session_idle_s == 0.0
    assert s0.avg_decode_tps is None  # no active time yet

    # t=10: serving (running>0); +1000 gen, +2000 prompt over 10s
    s1 = eng.derive(_fam(gen=2000.0, prompt=7000.0, req=12.0, running=3.0), now=10.0)
    assert s1.session_active_s == 10.0
    assert s1.session_idle_s == 0.0
    assert s1.avg_decode_tps == 100.0  # 1000 gen / 10s active
    assert s1.avg_prefill_tps == 200.0  # 2000 prompt / 10s active

    # t=15: idle (running==0); gen/prompt do not advance, no decode added
    s2 = eng.derive(_fam(gen=2000.0, prompt=7000.0, req=12.0, running=0.0), now=15.0)
    assert s2.session_active_s == 10.0  # unchanged (idle window)
    assert s2.session_idle_s == 5.0
    assert s2.avg_decode_tps == 100.0  # still 1000/10 (idle added no tokens)

    # t=25: serving again; +500 gen, +1000 prompt over 10s active
    s3 = eng.derive(_fam(gen=2500.0, prompt=8000.0, req=14.0, running=1.0), now=25.0)
    assert s3.session_active_s == 20.0
    assert s3.session_idle_s == 5.0
    assert s3.avg_decode_tps == (1000.0 + 500.0) / 20.0  # 75.0
    assert s3.avg_prefill_tps == (2000.0 + 1000.0) / 20.0  # 150.0
    # active fraction = 20 / (20 + 5)
    assert s3.session_active_frac == 20.0 / 25.0
    # session totals/requests are baselined at the first sample
    assert s3.session_requests == 14 - 10  # 4
    assert s3.session_gen_tokens == 2500.0 - 1000.0  # 1500
    assert s3.session_prompt_tokens == 8000.0 - 5000.0  # 3000
    assert s3.avg_gen_tokens_per_req == 1500.0 / 4  # 375.0


def test_session_reset_zeroes_accumulators():
    eng = MetricsEngine()
    eng.derive(_fam(gen=1000.0, prompt=5000.0, req=10.0, running=2.0), now=0.0)
    s1 = eng.derive(_fam(gen=2000.0, prompt=7000.0, req=12.0, running=2.0), now=10.0)
    assert s1.session_active_s > 0.0 and s1.avg_decode_tps is not None

    eng.reset_session()
    # First derive after reset re-baselines: no accumulation, totals zero.
    s2 = eng.derive(_fam(gen=2000.0, prompt=7000.0, req=12.0, running=2.0), now=20.0)
    assert s2.session_active_s == 0.0
    assert s2.session_idle_s == 0.0
    assert s2.avg_decode_tps is None
    assert s2.avg_prefill_tps is None
    assert s2.session_requests == 0
    assert s2.session_gen_tokens == 0.0
    assert s2.avg_gen_tokens_per_req is None


def test_session_rebaselines_on_counter_reset():
    eng = MetricsEngine()
    eng.derive(_fam(gen=5000.0, prompt=9000.0, req=20.0, running=2.0), now=0.0)
    eng.derive(_fam(gen=6000.0, prompt=11000.0, req=22.0, running=2.0), now=10.0)
    # Server restarts: gen_total drops below the session baseline -> re-baseline.
    s = eng.derive(_fam(gen=100.0, prompt=200.0, req=1.0, running=2.0), now=20.0)
    assert s.session_active_s == 0.0
    assert s.session_idle_s == 0.0
    assert s.session_gen_tokens == 0.0
    assert s.session_requests == 0
    assert s.avg_decode_tps is None


# ---------------------------------------------------------------------------
# v0.6 metrics: prefill latency, gen_len, finish_reasons, goodput
# ---------------------------------------------------------------------------

# Prometheus text helpers
_PREFILL_T0 = """\
# HELP vllm:request_prefill_time_seconds Prefill time
# TYPE vllm:request_prefill_time_seconds histogram
vllm:request_prefill_time_seconds_bucket{le="0.1"} 2.0
vllm:request_prefill_time_seconds_bucket{le="0.5"} 5.0
vllm:request_prefill_time_seconds_bucket{le="1.0"} 8.0
vllm:request_prefill_time_seconds_bucket{le="+Inf"} 10.0
vllm:request_prefill_time_seconds_count 10.0
vllm:request_prefill_time_seconds_sum 4.2
"""

_PREFILL_T1 = """\
# HELP vllm:request_prefill_time_seconds Prefill time
# TYPE vllm:request_prefill_time_seconds histogram
vllm:request_prefill_time_seconds_bucket{le="0.1"} 4.0
vllm:request_prefill_time_seconds_bucket{le="0.5"} 9.0
vllm:request_prefill_time_seconds_bucket{le="1.0"} 14.0
vllm:request_prefill_time_seconds_bucket{le="+Inf"} 20.0
vllm:request_prefill_time_seconds_count 20.0
vllm:request_prefill_time_seconds_sum 8.4
"""

_GEN_LEN_T0 = """\
# HELP vllm:request_generation_tokens Generation token counts
# TYPE vllm:request_generation_tokens histogram
vllm:request_generation_tokens_bucket{le="64"} 3.0
vllm:request_generation_tokens_bucket{le="256"} 7.0
vllm:request_generation_tokens_bucket{le="512"} 9.0
vllm:request_generation_tokens_bucket{le="+Inf"} 10.0
vllm:request_generation_tokens_count 10.0
vllm:request_generation_tokens_sum 2000.0
"""

_GEN_LEN_T1 = """\
# HELP vllm:request_generation_tokens Generation token counts
# TYPE vllm:request_generation_tokens histogram
vllm:request_generation_tokens_bucket{le="64"} 5.0
vllm:request_generation_tokens_bucket{le="256"} 12.0
vllm:request_generation_tokens_bucket{le="512"} 16.0
vllm:request_generation_tokens_bucket{le="+Inf"} 20.0
vllm:request_generation_tokens_count 20.0
vllm:request_generation_tokens_sum 5500.0
"""

_SUCCESS_T0 = """\
# HELP vllm:request_success_total Finished requests by reason
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop"} 8.0
vllm:request_success_total{finished_reason="length"} 2.0
"""

_SUCCESS_T1 = """\
# HELP vllm:request_success_total Finished requests by reason
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop"} 14.0
vllm:request_success_total{finished_reason="length"} 6.0
"""

_TTFT_T0 = """\
# HELP vllm:time_to_first_token_seconds TTFT
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.5"} 3.0
vllm:time_to_first_token_seconds_bucket{le="1.0"} 7.0
vllm:time_to_first_token_seconds_bucket{le="2.0"} 9.0
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 10.0
vllm:time_to_first_token_seconds_count 10.0
vllm:time_to_first_token_seconds_sum 9.5
"""

_TTFT_T1 = """\
# HELP vllm:time_to_first_token_seconds TTFT
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.5"} 7.0
vllm:time_to_first_token_seconds_bucket{le="1.0"} 13.0
vllm:time_to_first_token_seconds_bucket{le="2.0"} 18.0
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 20.0
vllm:time_to_first_token_seconds_count 20.0
vllm:time_to_first_token_seconds_sum 18.5
"""


def _combined(prefill: str, gen: str, success: str, ttft: str) -> str:
    return prefill + gen + success + ttft


def test_v06_prefill_quantiles_and_gen_len_mean_and_finish_reasons_and_goodput():
    """Two-sample derive: windowed metrics for prefill, gen_len, finish_reasons, goodput_ttft."""
    t0 = _combined(_PREFILL_T0, _GEN_LEN_T0, _SUCCESS_T0, _TTFT_T0)
    t1 = _combined(_PREFILL_T1, _GEN_LEN_T1, _SUCCESS_T1, _TTFT_T1)

    eng = MetricsEngine()
    eng.derive(parse_metrics(t0), now=0.0)
    snap = eng.derive(parse_metrics(t1), now=1.0)

    # prefill p50 should be a positive float (windowed buckets have observations)
    assert snap.prefill.p50 is not None
    assert snap.prefill.p50 > 0.0

    # gen_len mean: delta sum / delta count = (5500 - 2000) / (20 - 10) = 350.0
    expected_mean = (5500.0 - 2000.0) / (20.0 - 10.0)
    assert snap.gen_len.mean is not None
    assert abs(snap.gen_len.mean - expected_mean) < 1e-9

    # finish_reasons: deltas are stop+6, length+4; total 10; fractions must sum to 1
    assert abs(sum(snap.finish_reasons.values()) - 1.0) < 1e-9
    delta_stop = 14.0 - 8.0  # 6
    delta_length = 6.0 - 2.0  # 4
    total_delta = delta_stop + delta_length  # 10
    assert abs(snap.finish_reasons.get("stop", 0.0) - delta_stop / total_delta) < 1e-9
    assert abs(snap.finish_reasons.get("length", 0.0) - delta_length / total_delta) < 1e-9

    # goodput_ttft: fraction of requests completing ttft within SLO (1.0s default)
    assert snap.goodput_ttft is not None
    assert 0.0 <= snap.goodput_ttft <= 1.0

    # SLO params propagated through
    assert snap.ttft_slo_s == 1.0
    assert snap.tpot_slo_s == 0.05


def test_v06_engine_slo_params_propagated():
    """Custom SLO params are stored on the engine and reflected in Snapshot."""
    eng = MetricsEngine(ttft_slo_s=0.5, tpot_slo_s=0.02)
    t = _combined(_PREFILL_T0, _GEN_LEN_T0, _SUCCESS_T0, _TTFT_T0)
    snap = eng.derive(parse_metrics(t), now=0.0)
    assert snap.ttft_slo_s == 0.5
    assert snap.tpot_slo_s == 0.02


def test_v06_finish_reasons_empty_when_no_success_metric():
    """Engine returns empty dict when vllm:request_success_total is absent."""
    eng = MetricsEngine()
    snap = eng.derive(parse_metrics(_PREFILL_T0 + _GEN_LEN_T0 + _TTFT_T0), now=0.0)
    assert snap.finish_reasons == {}
