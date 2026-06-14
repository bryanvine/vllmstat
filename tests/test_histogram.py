import math

from vllmtop.core.histogram import histogram_quantile, windowed_buckets


def test_quantile_linear_interpolation():
    buckets = [(0.1, 1.0), (0.5, 3.0), (float("inf"), 4.0)]
    # total=4, p50 target=2.0 falls in bucket (0.1,0.5]: prev_count=1, count=3
    # frac=(2-1)/(3-1)=0.5 -> 0.1 + 0.5*(0.5-0.1)=0.3
    assert math.isclose(histogram_quantile(buckets, 0.5), 0.3, rel_tol=1e-9)


def test_quantile_empty_or_zero_total():
    assert histogram_quantile([], 0.5) is None
    assert histogram_quantile([(0.1, 0.0), (float("inf"), 0.0)], 0.5) is None


def test_quantile_in_inf_bucket_returns_prev_le():
    buckets = [(0.1, 1.0), (float("inf"), 10.0)]
    # p99 target=9.9 > 1.0 so crosses in +Inf bucket -> returns prev finite le
    assert histogram_quantile(buckets, 0.99) == 0.1


def test_windowed_buckets_subtracts_prev():
    prev = [(0.1, 5.0), (float("inf"), 8.0)]
    cur = [(0.1, 6.0), (float("inf"), 12.0)]
    assert windowed_buckets(prev, cur) == [(0.1, 1.0), (float("inf"), 4.0)]


def test_windowed_buckets_handles_reset():
    prev = [(0.1, 50.0), (float("inf"), 80.0)]
    cur = [(0.1, 1.0), (float("inf"), 2.0)]  # counters reset (smaller)
    # falls back to current (treat prev as zero)
    assert windowed_buckets(prev, cur) == cur
