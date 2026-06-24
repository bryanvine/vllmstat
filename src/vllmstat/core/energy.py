from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

_DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


@dataclass(frozen=True)
class TouRule:
    rate: float
    days: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)  # 0=Mon .. 6=Sun
    start_min: int | None = None  # minutes from local midnight; None = all day
    end_min: int | None = None
    label: str = ""
    default: bool = False


@dataclass(frozen=True)
class EnergyConfig:
    currency: str = "$"
    store: str | None = None
    interval: float = 10.0
    retention_days: int = 7
    tou: tuple[TouRule, ...] = ()


@dataclass(frozen=True)
class GpuEnergy:
    gpu_idx: int
    watts: float
    kwh: float
    cost: float | None


@dataclass(frozen=True)
class InstanceEnergy:
    instance: str
    kwh: float
    cost: float | None
    tokens: float = 0.0


def integrate_kwh(p0: float, p1: float, dt_s: float) -> float:
    """Trapezoidal energy in kWh from two power readings (W) dt_s seconds apart."""
    if dt_s <= 0:
        return 0.0
    return (p0 + p1) / 2.0 * dt_s / 3600.0 / 1000.0


def _parse_days(spec: str) -> tuple[int, ...]:
    spec = spec.strip().lower()
    if not spec:
        return tuple(range(7))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            ai, bi = _DAYS[a.strip()], _DAYS[b.strip()]
            i = ai
            while True:
                out.add(i)
                if i == bi:
                    break
                i = (i + 1) % 7
        else:
            out.add(_DAYS[part])
    return tuple(sorted(out))


def _parse_hhmm(s: str) -> int:
    h, _, m = s.strip().partition(":")
    hi, mi = int(h), int(m)
    if not (0 <= hi <= 23 and 0 <= mi <= 59):
        raise ValueError(f"invalid time {s!r}")
    return hi * 60 + mi


def parse_energy_config(table: dict) -> EnergyConfig:
    currency = str(table.get("currency", "$"))
    store = table.get("store")
    store = str(store) if store is not None else None
    interval = float(table.get("interval", 10.0))
    retention_days = int(table.get("retention_days", 7))
    rules: list[TouRule] = []
    raw = table.get("tou", [])
    if not isinstance(raw, list):
        raise ValueError("'energy.tou' must be an array of tables")
    for r in raw:
        rate = float(r["rate"])
        if rate < 0:
            raise ValueError("energy rate must be >= 0")
        if r.get("default"):
            rules.append(TouRule(rate=rate, label=str(r.get("label", "")), default=True))
            continue
        try:
            days = _parse_days(str(r.get("days", "mon-sun")))
        except KeyError as e:
            raise ValueError(f"invalid day in TOU rule: {e}") from e
        start = _parse_hhmm(r["from"]) if "from" in r else None
        end = _parse_hhmm(r["to"]) if "to" in r else None
        rules.append(
            TouRule(rate=rate, days=days, start_min=start, end_min=end,
                    label=str(r.get("label", "")))
        )
    if rules and not any(x.default for x in rules):
        raise ValueError("a TOU schedule needs exactly one rule with default = true")
    return EnergyConfig(
        currency=currency, store=store, interval=interval,
        retention_days=retention_days, tou=tuple(rules),
    )


def _in_window(rule: TouRule, minute: int) -> bool:
    if rule.start_min is None or rule.end_min is None:
        return True
    if rule.start_min <= rule.end_min:
        return rule.start_min <= minute < rule.end_min
    # overnight window wraps midnight
    return minute >= rule.start_min or minute < rule.end_min


def rate_at(cfg: EnergyConfig, when: datetime) -> tuple[float | None, str]:
    """Return (rate, label) for a local datetime, or (None, '') if no schedule."""
    if not cfg.tou:
        return None, ""
    minute = when.hour * 60 + when.minute
    weekday = when.weekday()  # Mon=0
    default: TouRule | None = None
    for rule in cfg.tou:
        if rule.default:
            default = rule
            continue
        if weekday in rule.days and _in_window(rule, minute):
            return rule.rate, rule.label
    if default is not None:
        return default.rate, default.label
    return None, ""


def replace_store(cfg: EnergyConfig, store: str) -> EnergyConfig:
    from dataclasses import replace
    return replace(cfg, store=store)
