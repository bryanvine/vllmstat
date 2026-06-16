from __future__ import annotations


def histogram_quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    """Prometheus-style quantile from cumulative buckets [(le, cum_count), ...] sorted asc."""
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = q * total
    prev_le = 0.0
    prev_count = 0.0
    for le, count in buckets:
        if count >= target:
            if le == float("inf"):
                return prev_le
            bucket_count = count - prev_count
            if bucket_count <= 0:
                return le
            frac = (target - prev_count) / bucket_count
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return buckets[-1][0]


def histogram_fraction_below(buckets: list[tuple[float, float]], threshold: float) -> float | None:
    """Fraction of observations with value <= threshold, interpolated within the bucket.

    `buckets` are cumulative [(le, cum_count), ...] sorted ascending (Prometheus style).
    Returns None when there are no observations.
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if threshold < le:
            if le == float("inf"):
                return prev_count / total
            bucket_count = count - prev_count
            if bucket_count <= 0:
                return prev_count / total
            frac = (threshold - prev_le) / (le - prev_le)
            return min(1.0, (prev_count + frac * bucket_count) / total)
        prev_le, prev_count = le, count
    return 1.0


def histogram_max_le(buckets: list[tuple[float, float]]) -> float | None:
    """Upper bound on the largest observed value from cumulative buckets.

    `buckets` are cumulative [(le, cum_count), ...] sorted ascending, ending in a
    `+Inf` bucket (Prometheus style). Returns the smallest *finite* `le` whose
    cumulative count reaches the total — the bucket the maximum falls in, so the
    true max is `<= le`. Returns None when there are no observations, or when the
    maximum exceeds every finite boundary (only the `+Inf` bucket reaches total).
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    for le, count in buckets:
        if le == float("inf"):
            return None  # max exceeds the largest finite bucket
        if count >= total:
            return le
    return None


def windowed_buckets(
    prev: list[tuple[float, float]], cur: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Per-`le` delta cur-prev. If any delta is negative (counter reset), return cur."""
    prev_map = dict(prev)
    out: list[tuple[float, float]] = []
    for le, count in cur:
        delta = count - prev_map.get(le, 0.0)
        if delta < 0:
            return cur
        out.append((le, delta))
    return out
