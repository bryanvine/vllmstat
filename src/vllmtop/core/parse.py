from __future__ import annotations

from prometheus_client.parser import text_string_to_metric_families

Families = dict[str, list[tuple[dict[str, str], float]]]


def parse_metrics(text: str) -> Families:
    """Parse Prometheus exposition text into {sample_name: [(labels, value), ...]}."""
    families: Families = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            families.setdefault(sample.name, []).append((dict(sample.labels), sample.value))
    return families


def sum_value(families: Families, name: str) -> float | None:
    rows = families.get(name)
    if not rows:
        return None
    return sum(v for _, v in rows)


def first_value(families: Families, name: str) -> float | None:
    rows = families.get(name)
    if not rows:
        return None
    return rows[0][1]


def info_labels(families: Families, name: str) -> dict[str, str]:
    rows = families.get(name)
    return rows[0][0] if rows else {}


def get_buckets(families: Families, base: str) -> list[tuple[float, float]]:
    """Aggregate `<base>_bucket` samples across labels, summing counts per `le`."""
    rows = families.get(base + "_bucket", [])
    agg: dict[float, float] = {}
    for labels, value in rows:
        le = float(labels["le"])
        agg[le] = agg.get(le, 0.0) + value
    return sorted(agg.items())


def hist_count(families: Families, base: str) -> float | None:
    return sum_value(families, base + "_count")


def hist_sum(families: Families, base: str) -> float | None:
    return sum_value(families, base + "_sum")
