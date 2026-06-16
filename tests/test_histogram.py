import math

from vllmstat.core.histogram import histogram_max_le, histogram_quantile, windowed_buckets


def test_quantile_linear_interpolation():
    buckets = [(0.1, 1.0), (0.5, 3.0), (float("inf"), 4.0)]
    # total=4, p50 target=2.0 falls in bucket (0.1,0.5]: prev_count=1, count=3
    # frac=(2-1)/(3-1)=0.5 -> 0.1 + 0.5*(0.5-0.1)=0.3
    result = histogram_quantile(buckets, 0.5)
    assert result is not None
    assert math.isclose(result, 0.3, rel_tol=1e-9)


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


def test_histogram_fraction_below():
    from vllmstat.core.histogram import histogram_fraction_below

    b = [(0.1, 10.0), (0.5, 50.0), (1.0, 90.0), (float("inf"), 100.0)]
    assert histogram_fraction_below(b, 0.5) == 0.5
    result_03 = histogram_fraction_below(b, 0.3)
    assert result_03 is not None and abs(result_03 - 0.30) < 1e-9
    assert histogram_fraction_below(b, 5.0) == 0.9
    assert histogram_fraction_below([], 1.0) is None
    assert histogram_fraction_below([(1.0, 0.0)], 1.0) is None


def test_histogram_max_le():
    # max falls in the highest populated finite bucket (le=100 first reaches total 5)
    assert histogram_max_le([(10.0, 2.0), (100.0, 5.0), (float("inf"), 5.0)]) == 100.0
    # all observations in the lowest bucket -> that bucket's le
    assert histogram_max_le([(10.0, 5.0), (100.0, 5.0), (float("inf"), 5.0)]) == 10.0
    # no observations
    assert histogram_max_le([]) is None
    assert histogram_max_le([(10.0, 0.0), (float("inf"), 0.0)]) is None
    # max exceeds every finite bucket (only +Inf reaches the total) -> None
    assert histogram_max_le([(10.0, 2.0), (100.0, 4.0), (float("inf"), 5.0)]) is None
