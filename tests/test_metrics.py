from pathlib import Path

from vllmtop.core.metrics import MetricsEngine
from vllmtop.core.parse import parse_metrics

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
    fam2["vllm:generation_tokens_total"] = [(fam["vllm:generation_tokens_total"][0][0], base + 1420)]
    s1 = eng.derive(fam2, now=10.0)
    assert s1.gen_tps > 0.0


def test_kv_and_cache_fields_present():
    fam = parse_metrics(FIX.read_text())
    eng = MetricsEngine(dims=DIMS, max_model_len=262144)
    s = eng.derive(fam, now=0.0)
    assert s.kv_dtype == "turboquant_k3v4_nc"
    assert s.kv_capacity_tokens == 6947 * 64
    assert s.kv_ratio_kind == "nominal"          # memory_bytes is None on fixture
    assert s.prefix_hit_lifetime is not None and 0.0 <= s.prefix_hit_lifetime <= 1.0
    # token sources: local_cache_hit + local_compute fractions sum ~1
    assert s.src_cache_hit is not None
    assert s.spec_active is True                  # fixture has spec decode
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
