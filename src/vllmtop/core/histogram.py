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
