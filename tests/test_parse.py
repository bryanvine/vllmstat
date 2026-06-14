from pathlib import Path

from vllmtop.core.parse import (
    parse_metrics, sum_value, first_value, info_labels, get_buckets,
)

FIX = Path(__file__).parent / "fixtures" / "metrics_qwen3.txt"

SMALL = """\
# HELP vllm:num_requests_running Number of requests running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 2.0
# HELP vllm:generation_tokens_total gen.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="m"} 100.0
# HELP vllm:ttft seconds
# TYPE vllm:ttft histogram
vllm:ttft_bucket{le="0.1"} 1.0
vllm:ttft_bucket{le="0.5"} 3.0
vllm:ttft_bucket{le="+Inf"} 4.0
vllm:ttft_count 4.0
vllm:ttft_sum 1.2
"""


def test_parse_small_gauge_and_counter():
    fam = parse_metrics(SMALL)
    assert sum_value(fam, "vllm:num_requests_running") == 2.0
    assert first_value(fam, "vllm:generation_tokens_total") == 100.0
    assert sum_value(fam, "vllm:does_not_exist") is None


def test_get_buckets_sorted_with_inf():
    fam = parse_metrics(SMALL)
    buckets = get_buckets(fam, "vllm:ttft")
    assert buckets[0] == (0.1, 1.0)
    assert buckets[-1][0] == float("inf")
    assert buckets[-1][1] == 4.0


def test_info_labels():
    text = 'vllm:cache_config_info{cache_dtype="turboquant_k3v4_nc",num_gpu_blocks="6947"} 1.0\n'
    fam = parse_metrics(text)
    labels = info_labels(fam, "vllm:cache_config_info")
    assert labels["cache_dtype"] == "turboquant_k3v4_nc"
    assert labels["num_gpu_blocks"] == "6947"


def test_golden_fixture_has_core_metrics():
    fam = parse_metrics(FIX.read_text())
    for name in (
        "vllm:num_requests_running",
        "vllm:generation_tokens_total",
        "vllm:prefix_cache_hits_total",
        "vllm:kv_cache_usage_perc",
        "vllm:cache_config_info",
    ):
        assert name in fam, name
    labels = info_labels(fam, "vllm:cache_config_info")
    assert labels["cache_dtype"] == "turboquant_k3v4_nc"
    assert labels["num_gpu_blocks"] == "6947"
