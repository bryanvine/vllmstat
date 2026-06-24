# Energy Accounting + Long-Term Daemon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add total-energy counters (kWh + cost at adjustable time-of-use electricity rates) and an installable headless daemon that records GPU/instance energy long term in a SQLite store the TUI reads.

**Architecture:** A new `vllmstat daemon` builds the same `Fleet` as the TUI, polls on its own interval, integrates per-GPU watts into energy (trapezoidal), computes cost from a time-of-use schedule, and writes to a SQLite store (WAL). The TUI opens that store read-only and renders an ENERGY panel; with no store it degrades to a session estimate.

**Tech Stack:** Python ≥3.10, stdlib `sqlite3`, stdlib `tomllib`/`tomli`, existing Textual TUI + `Fleet`/provider/GPU stack. No new dependencies.

**Conventions (match existing code):**
- Renderers return plain monochrome strings — **no Rich markup**.
- `.venv/bin/python -m pytest` to run tests; `.venv/bin/python -m pytest tests/test_X.py::test_Y -v` for one.
- Gate before finishing: `.venv/bin/ruff check src tests && .venv/bin/python -m pytest`.
- Time: the daemon stores **wall-clock epoch seconds** (`time.time()`) for sample timestamps and derives the integration interval from successive wall times; the TUI loop keeps using `time.monotonic()` and is unaffected.

---

## File Structure

- `src/vllmstat/core/energy.py` (new) — pure: `TouRule`, `EnergyConfig`, `parse_energy_config`, `rate_at`, `integrate_kwh`, energy-delta carriers `GpuEnergy`/`InstanceEnergy`.
- `src/vllmstat/core/store.py` (new) — `Store`: SQLite schema/migration, `record`, `prune`, `read_view`, `totals_gpu`/`totals_instance`.
- `src/vllmstat/core/state.py` (modify) — add `EnergyView` read struct.
- `src/vllmstat/core/config_file.py` (modify) — already returns globals; no change needed beyond passing `[energy]` through (verify).
- `src/vllmstat/daemon.py` (new) — `Collector` loop: build fleet, poll, integrate, write; restart-safe.
- `src/vllmstat/core/service.py` (new) — systemd unit-file generation + install/uninstall/paths.
- `src/vllmstat/render.py` (modify) — `energy_panel`.
- `src/vllmstat/cli.py` (modify) — `daemon` subcommand routing; keep no-subcommand → TUI.
- `src/vllmstat/app.py` (modify) — open store read-only, periodic read, wire ENERGY panel.
- `src/vllmstat/__init__.py`, `pyproject.toml`, `README.md` — version bump + docs at finish.
- Tests: `tests/test_energy.py`, `tests/test_store.py`, `tests/test_service.py`, `tests/test_daemon.py`, plus additions to `tests/test_render.py`, `tests/test_cli.py`.

---

### Task 1: Pure energy math — config, TOU rate lookup, integration

**Files:**
- Create: `src/vllmstat/core/energy.py`
- Test: `tests/test_energy.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_energy.py
from datetime import datetime

import pytest

from vllmstat.core.energy import (
    EnergyConfig,
    GpuEnergy,
    InstanceEnergy,
    TouRule,
    integrate_kwh,
    parse_energy_config,
    rate_at,
)


def test_integrate_kwh_trapezoidal():
    # 1000 W avg for 3600 s = 1.0 kWh; trapezoid of 800->1200 over 1 h
    assert integrate_kwh(800.0, 1200.0, 3600.0) == pytest.approx(1.0)
    # 500 W for 10 s
    assert integrate_kwh(500.0, 500.0, 10.0) == pytest.approx(500.0 * 10 / 3600 / 1000)
    assert integrate_kwh(0.0, 0.0, 10.0) == 0.0


def test_parse_energy_config_flat_and_tou():
    cfg = parse_energy_config(
        {
            "currency": "£",
            "store": "/var/lib/vllmstat/x.db",
            "interval": 15,
            "retention_days": 14,
            "tou": [
                {"days": "mon-fri", "from": "16:00", "to": "21:00", "rate": 0.42, "label": "peak"},
                {"default": True, "rate": 0.12, "label": "off-peak"},
            ],
        }
    )
    assert cfg.currency == "£" and cfg.store.endswith("x.db")
    assert cfg.interval == 15.0 and cfg.retention_days == 14
    assert len(cfg.tou) == 2 and cfg.tou[1].default is True


def test_parse_energy_config_defaults():
    cfg = parse_energy_config({})
    assert cfg.currency == "$" and cfg.store is None
    assert cfg.interval == 10.0 and cfg.retention_days == 7 and cfg.tou == ()


def test_parse_energy_config_requires_default_when_tou_present():
    with pytest.raises(ValueError, match="default"):
        parse_energy_config({"tou": [{"days": "mon-fri", "from": "9:00", "to": "17:00", "rate": 0.3}]})


def test_parse_energy_config_rejects_bad_time_and_negative_rate():
    with pytest.raises(ValueError):
        parse_energy_config({"tou": [{"default": True, "rate": -1.0}]})
    with pytest.raises(ValueError):
        parse_energy_config(
            {"tou": [{"days": "mon-fri", "from": "25:00", "to": "9:00", "rate": 0.3},
                     {"default": True, "rate": 0.1}]}
        )


def test_rate_at_picks_window_then_default():
    cfg = parse_energy_config(
        {"tou": [
            {"days": "mon-fri", "from": "16:00", "to": "21:00", "rate": 0.42, "label": "peak"},
            {"default": True, "rate": 0.12, "label": "off-peak"},
        ]}
    )
    # Wednesday 18:00 -> peak
    assert rate_at(cfg, datetime(2026, 6, 24, 18, 0)) == (0.42, "peak")
    # Wednesday 09:00 -> default off-peak
    assert rate_at(cfg, datetime(2026, 6, 24, 9, 0)) == (0.12, "off-peak")
    # Sunday 18:00 -> not a weekday -> default
    assert rate_at(cfg, datetime(2026, 6, 28, 18, 0)) == (0.12, "off-peak")


def test_rate_at_overnight_window_wraps_midnight():
    cfg = parse_energy_config(
        {"tou": [
            {"days": "mon-sun", "from": "21:00", "to": "07:00", "rate": 0.08, "label": "night"},
            {"default": True, "rate": 0.20, "label": "day"},
        ]}
    )
    assert rate_at(cfg, datetime(2026, 6, 24, 23, 30)) == (0.08, "night")
    assert rate_at(cfg, datetime(2026, 6, 24, 6, 0)) == (0.08, "night")
    assert rate_at(cfg, datetime(2026, 6, 24, 12, 0)) == (0.20, "day")


def test_rate_at_no_config_returns_none():
    assert rate_at(parse_energy_config({}), datetime(2026, 6, 24, 12, 0)) == (None, "")


def test_energy_carriers_are_frozen_dataclasses():
    g = GpuEnergy(gpu_idx=0, watts=200.0, kwh=0.01, cost=0.002)
    i = InstanceEnergy(instance="a", kwh=0.01, cost=0.002, tokens=50.0)
    assert g.gpu_idx == 0 and i.instance == "a"
    with pytest.raises(Exception):
        g.kwh = 1.0  # frozen
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_energy.py -q`
Expected: FAIL with `ModuleNotFoundError: vllmstat.core.energy`.

- [ ] **Step 3: Implement `core/energy.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_energy.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/vllmstat/core/energy.py tests/test_energy.py
git commit -m "feat(energy): pure TOU rate lookup + kWh integration"
```

---

### Task 2: SQLite store — schema, record, prune, read_view

**Files:**
- Create: `src/vllmstat/core/store.py`
- Modify: `src/vllmstat/core/state.py` (add `EnergyView`)
- Test: `tests/test_store.py`

- [ ] **Step 1: Add `EnergyView` to `core/state.py`**

Append after the `Snapshot` class:

```python
@dataclass
class EnergyView:
    """Read-only energy figures for the TUI (assembled from the store + live data)."""
    available: bool = False
    currency: str = "$"
    today_kwh: float = 0.0
    today_cost: float | None = None
    alltime_kwh: float = 0.0
    alltime_cost: float | None = None
    now_w: float | None = None
    rate: float | None = None
    rate_label: str = ""
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_store.py
from vllmstat.core.energy import GpuEnergy, InstanceEnergy
from vllmstat.core.store import Store


def _open(tmp_path):
    return Store.open(str(tmp_path / "e.db"))


def test_record_accumulates_totals(tmp_path):
    s = _open(tmp_path)
    # ts is wall-clock epoch seconds; 2026-06-24 12:00:00 UTC ~ 1782648000
    ts = 1782648000.0
    s.record(ts, [GpuEnergy(0, 200.0, 0.01, 0.002)], [InstanceEnergy("a", 0.01, 0.002, 50.0)])
    s.record(ts + 10, [GpuEnergy(0, 220.0, 0.02, 0.004)], [InstanceEnergy("a", 0.02, 0.004, 60.0)])
    g = {r["gpu_idx"]: r for r in s.totals_gpu()}
    assert g[0]["kwh"] == pytest.approx(0.03)
    assert g[0]["cost"] == pytest.approx(0.006)
    i = {r["instance"]: r for r in s.totals_instance()}
    assert i["a"]["kwh"] == pytest.approx(0.03) and i["a"]["tokens"] == pytest.approx(110.0)
    s.close()


def test_read_view_today_and_alltime(tmp_path):
    s = _open(tmp_path)
    ts = 1782648000.0  # treat as "today" for the read
    s.record(ts, [GpuEnergy(0, 200.0, 1.0, 0.10)], [InstanceEnergy("a", 1.0, 0.10)])
    view = s.read_view(now=ts, currency="$")
    assert view.available is True
    assert view.today_kwh == pytest.approx(1.0) and view.today_cost == pytest.approx(0.10)
    assert view.alltime_kwh == pytest.approx(1.0) and view.alltime_cost == pytest.approx(0.10)
    s.close()


def test_prune_drops_old_samples_keeps_daily_and_totals(tmp_path):
    s = _open(tmp_path)
    old = 1000.0
    s.record(old, [GpuEnergy(0, 100.0, 0.5, 0.05)], [InstanceEnergy("a", 0.5, 0.05)])
    s.record(1782648000.0, [GpuEnergy(0, 100.0, 0.5, 0.05)], [InstanceEnergy("a", 0.5, 0.05)])
    removed = s.prune(before_ts=1782000000.0)
    assert removed == 1
    assert s.sample_count() == 1                      # old raw sample gone
    assert s.totals_gpu()[0]["kwh"] == pytest.approx(1.0)  # totals untouched
    assert s.daily_count() >= 1                       # daily rollup retained
    s.close()


def test_cost_none_when_rate_unset(tmp_path):
    s = _open(tmp_path)
    s.record(1782648000.0, [GpuEnergy(0, 100.0, 0.5, None)], [InstanceEnergy("a", 0.5, None)])
    assert s.totals_gpu()[0]["cost"] is None
    s.close()


def test_concurrent_readonly_open(tmp_path):
    w = _open(tmp_path)
    w.record(1782648000.0, [GpuEnergy(0, 100.0, 0.5, 0.05)], [InstanceEnergy("a", 0.5, 0.05)])
    r = Store.open(str(tmp_path / "e.db"), read_only=True)
    assert r.totals_gpu()[0]["kwh"] == pytest.approx(0.5)
    r.close()
    w.close()
```

Add `import pytest` at the top of the test file.

- [ ] **Step 3: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_store.py -q`
Expected: FAIL with `ModuleNotFoundError: vllmstat.core.store`.

- [ ] **Step 4: Implement `core/store.py`**

```python
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from vllmstat.core.energy import GpuEnergy, InstanceEnergy
from vllmstat.core.state import EnergyView

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts REAL NOT NULL, gpu_idx INTEGER NOT NULL,
    watts REAL NOT NULL, kwh REAL NOT NULL, cost REAL
);
CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts);
CREATE TABLE IF NOT EXISTS daily (
    date TEXT NOT NULL, scope TEXT NOT NULL, key TEXT NOT NULL,
    kwh REAL NOT NULL DEFAULT 0, cost REAL, tokens REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (date, scope, key)
);
CREATE TABLE IF NOT EXISTS totals_gpu (
    gpu_idx INTEGER PRIMARY KEY, kwh REAL NOT NULL DEFAULT 0, cost REAL,
    tokens REAL NOT NULL DEFAULT 0, since_ts REAL, updated_ts REAL
);
CREATE TABLE IF NOT EXISTS totals_instance (
    instance TEXT PRIMARY KEY, kwh REAL NOT NULL DEFAULT 0, cost REAL,
    tokens REAL NOT NULL DEFAULT 0, since_ts REAL, updated_ts REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _local_date(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _add_cost(old, delta):
    """Cost addition that propagates 'unknown' (None) without poisoning known sums."""
    if delta is None:
        return old
    return (old or 0.0) + delta


class Store:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    @classmethod
    def open(cls, path: str, *, read_only: bool = False) -> "Store":
        p = Path(path).expanduser()
        if read_only:
            conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2.0)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(p), timeout=5.0)
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
        return cls(conn)

    def record(self, ts: float, gpus: list[GpuEnergy], instances: list[InstanceEnergy]) -> None:
        date = _local_date(ts)
        c = self._conn
        for g in gpus:
            c.execute(
                "INSERT INTO samples(ts, gpu_idx, watts, kwh, cost) VALUES (?,?,?,?,?)",
                (ts, g.gpu_idx, g.watts, g.kwh, g.cost),
            )
            self._bump_totals("totals_gpu", "gpu_idx", str(g.gpu_idx), g.gpu_idx,
                              ts, g.kwh, g.cost, 0.0)
            self._bump_daily(date, "gpu", str(g.gpu_idx), g.kwh, g.cost, 0.0)
        for i in instances:
            self._bump_totals("totals_instance", "instance", i.instance, i.instance,
                              ts, i.kwh, i.cost, i.tokens)
            self._bump_daily(date, "instance", i.instance, i.kwh, i.cost, i.tokens)
        c.commit()

    def _bump_totals(self, table, keycol, key, keyval, ts, kwh, cost, tokens) -> None:
        row = self._conn.execute(
            f"SELECT kwh, cost, tokens FROM {table} WHERE {keycol}=?", (keyval,)
        ).fetchone()
        if row is None:
            self._conn.execute(
                f"INSERT INTO {table}({keycol}, kwh, cost, tokens, since_ts, updated_ts) "
                f"VALUES (?,?,?,?,?,?)",
                (keyval, kwh, cost, tokens, ts, ts),
            )
        else:
            self._conn.execute(
                f"UPDATE {table} SET kwh=?, cost=?, tokens=?, updated_ts=? WHERE {keycol}=?",
                (row["kwh"] + kwh, _add_cost(row["cost"], cost), row["tokens"] + tokens, ts, keyval),
            )

    def _bump_daily(self, date, scope, key, kwh, cost, tokens) -> None:
        row = self._conn.execute(
            "SELECT kwh, cost, tokens FROM daily WHERE date=? AND scope=? AND key=?",
            (date, scope, key),
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO daily(date, scope, key, kwh, cost, tokens) VALUES (?,?,?,?,?,?)",
                (date, scope, key, kwh, cost, tokens),
            )
        else:
            self._conn.execute(
                "UPDATE daily SET kwh=?, cost=?, tokens=? WHERE date=? AND scope=? AND key=?",
                (row["kwh"] + kwh, _add_cost(row["cost"], cost), row["tokens"] + tokens,
                 date, scope, key),
            )

    def prune(self, before_ts: float) -> int:
        cur = self._conn.execute("DELETE FROM samples WHERE ts < ?", (before_ts,))
        self._conn.commit()
        return cur.rowcount

    def totals_gpu(self) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM totals_gpu ORDER BY gpu_idx").fetchall()]

    def totals_instance(self) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM totals_instance ORDER BY instance").fetchall()]

    def sample_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    def daily_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]

    def read_view(self, *, now: float, currency: str = "$") -> EnergyView:
        date = _local_date(now)
        today = self._conn.execute(
            "SELECT COALESCE(SUM(kwh),0) k, SUM(cost) c FROM daily WHERE scope='gpu' AND date=?",
            (date,),
        ).fetchone()
        alltime = self._conn.execute(
            "SELECT COALESCE(SUM(kwh),0) k, SUM(cost) c FROM totals_gpu"
        ).fetchone()
        has_rows = self._conn.execute("SELECT COUNT(*) FROM totals_gpu").fetchone()[0] > 0
        return EnergyView(
            available=has_rows,
            currency=currency,
            today_kwh=today["k"] or 0.0,
            today_cost=today["c"],
            alltime_kwh=alltime["k"] or 0.0,
            alltime_cost=alltime["c"],
        )

    def close(self) -> None:
        self._conn.close()
```

> Note: `read_view` uses UTC-free local date via `_local_date`. The test timestamps are treated as local; assertions only compare today==the single recorded day, so they hold regardless of host tz.

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_store.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/vllmstat/core/store.py src/vllmstat/core/state.py tests/test_store.py
git commit -m "feat(energy): SQLite store with totals, daily rollup, retention prune"
```

---

### Task 3: ENERGY panel renderer

**Files:**
- Modify: `src/vllmstat/render.py`
- Test: `tests/test_render.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_render.py`)

```python
from vllmstat.core.state import EnergyView


def test_energy_panel_full():
    v = EnergyView(available=True, currency="$", today_kwh=2.4, today_cost=0.43,
                   alltime_kwh=318.0, alltime_cost=57.2, now_w=412.0, rate=0.18,
                   rate_label="off-peak")
    out = render.energy_panel(v)
    assert "ENERGY" in out
    assert "today 2.4 kWh ($0.43)" in out
    assert "all-time 318.0 kWh ($57.20)" in out
    assert "now 412 W" in out and "$0.18/kWh" in out and "off-peak" in out


def test_energy_panel_rate_unset_hides_cost():
    v = EnergyView(available=True, currency="$", today_kwh=2.4, today_cost=None,
                   alltime_kwh=318.0, alltime_cost=None, now_w=400.0, rate=None)
    out = render.energy_panel(v)
    assert "today 2.4 kWh" in out and "$" not in out.split("now")[0].replace("ENERGY", "")
    assert "rate unset" in out


def test_energy_panel_empty_when_unavailable():
    assert render.energy_panel(EnergyView(available=False)) == ""
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_render.py -k energy_panel -q`
Expected: FAIL (`energy_panel` undefined).

- [ ] **Step 3: Implement `energy_panel`** (add to `render.py`, and import `EnergyView` in the existing state import line)

Change the import line:
```python
from vllmstat.core.state import EnergyView, FleetSnapshot, Instance, Snapshot
```

Add the renderer:
```python
def _kwh(v: float) -> str:
    return f"{v:.1f} kWh"


def energy_panel(v: "EnergyView") -> str:
    if not v.available:
        return ""
    cur = v.currency

    def money(x: float | None) -> str:
        return f" ({cur}{x:.2f})" if x is not None else ""

    line1 = f"today {_kwh(v.today_kwh)}{money(v.today_cost)}  ·  all-time {_kwh(v.alltime_kwh)}{money(v.alltime_cost)}"
    bits = []
    if v.now_w is not None:
        bits.append(f"now {v.now_w:.0f} W")
    if v.rate is not None:
        label = f" ({v.rate_label})" if v.rate_label else ""
        bits.append(f"rate {cur}{v.rate:.2f}/kWh{label}")
    else:
        bits.append("rate unset")
    line2 = " · ".join(bits)
    return f"ENERGY  {line1}\n        {line2}"
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_render.py -k energy_panel -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vllmstat/render.py tests/test_render.py
git commit -m "feat(energy): ENERGY panel renderer"
```

---

### Task 4: Energy config wiring (config file → EnergyConfig)

**Files:**
- Modify: `src/vllmstat/config.py`
- Test: `tests/test_config.py` (create if absent; otherwise append)

The config-file loader (`core/config_file.py`) already returns non-`instance` keys in `globals_`, so `[energy]` arrives as `config_globals["energy"]` (a dict; TOML `[[energy.tou]]` becomes `energy["tou"]`, a list). Add an `energy: EnergyConfig` field to `Config` and populate it from those globals.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from vllmstat.config import Config
from vllmstat.core.energy import EnergyConfig


def test_config_has_energy_default():
    cfg = Config.from_sources([], {})
    assert isinstance(cfg.energy, EnergyConfig)
    assert cfg.energy.currency == "$"
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_config_has_energy_default -q`
Expected: FAIL (`Config` has no attribute `energy`).

- [ ] **Step 3: Add the field to `config.py`**

Add import and field:
```python
from vllmstat.core.energy import EnergyConfig
```
In the `Config` dataclass, after `proxy: str | None = None`:
```python
    energy: EnergyConfig = field(default_factory=EnergyConfig)
```
(`from_sources` needs no change — energy comes from the config file, applied in `resolve_instances`.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_config_has_energy_default -q`
Expected: PASS.

- [ ] **Step 5: Wire config-file `[energy]` into `resolve_instances`** (in `cli.py`)

In `resolve_instances`, after the block that reads `config_globals` (right after the `gpu = config_globals.get("gpu")` handling), add:
```python
    energy_tbl = config_globals.get("energy")
    if isinstance(energy_tbl, dict):
        from vllmstat.core.energy import parse_energy_config
        try:
            cfg.energy = parse_energy_config(energy_tbl)
        except ValueError as e:
            print(f"vllmstat: ignoring [energy] config: {e}", file=sys.stderr)
```

- [ ] **Step 6: Write + run a wiring test**

```python
# tests/test_config.py  (append)
def test_energy_config_loaded_from_file(tmp_path):
    import os
    from vllmstat.cli import resolve_instances

    p = tmp_path / "vllmstat.toml"
    p.write_text(
        '[energy]\ncurrency = "£"\n'
        '[[energy.tou]]\ndefault = true\nrate = 0.15\n'
    )
    cfg = Config.from_sources(["--config", str(p)], {})
    resolve_instances(cfg, {})
    assert cfg.energy.currency == "£"
    assert cfg.energy.tou[0].rate == 0.15
```

Run: `.venv/bin/python -m pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/vllmstat/config.py src/vllmstat/cli.py tests/test_config.py
git commit -m "feat(energy): load [energy] config into Config.energy"
```

---

### Task 5: Daemon collector loop

**Files:**
- Create: `src/vllmstat/daemon.py`
- Test: `tests/test_daemon.py`

The `Collector` is the testable core (no sleeping, injectable clock). It holds the previous per-GPU power+timestamp, builds energy deltas on each `step`, and writes to the store. A thin `run()` wraps it with a real poll loop and signal handling.

Per-instance attribution: for each local instance, sum the kWh of the GPUs in its `instance.gpus` mapping (empty mapping → all host GPUs); attribute its generation-token delta (from `Snapshot.session_gen_tokens`) to the instance row.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_daemon.py
from vllmstat.core.energy import EnergyConfig, parse_energy_config
from vllmstat.core.state import GpuSample, GpuSnapshot, Instance, Snapshot
from vllmstat.core.store import Store
from vllmstat.daemon import Collector


def _gpu(idx, power):
    return GpuSnapshot(available=True, source="test",
                       gpus=[GpuSample(index=idx, name="x", power_w=power)])


def _snap(running=1.0, gen_tokens=0.0, gpus=()):
    s = Snapshot(ts=0.0, connected=True, running=running)
    s.session_gen_tokens = gen_tokens
    g = GpuSnapshot(available=True, source="test",
                    gpus=[GpuSample(index=i, name="x", power_w=p) for i, p in gpus])
    s.gpu = g
    return s


def test_collector_integrates_between_steps(tmp_path):
    store = Store.open(str(tmp_path / "e.db"))
    cfg = parse_energy_config({"tou": [{"default": True, "rate": 0.10}]})
    col = Collector(store, cfg)
    inst = Instance("a", "http://x", gpus=(0,), locality="local")
    # first step: baseline only, no energy yet
    col.step(1000.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    assert store.sample_count() == 0
    # second step 3600 s later at 1000 W -> 1.0 kWh, cost 0.10
    col.step(1000.0 + 3600.0, [(inst, _snap(gen_tokens=500.0, gpus=[(0, 1000.0)]))])
    g = store.totals_gpu()[0]
    assert g["kwh"] == pytest.approx(1.0) and g["cost"] == pytest.approx(0.10)
    i = store.totals_instance()[0]
    assert i["instance"] == "a" and i["kwh"] == pytest.approx(1.0)
    assert i["tokens"] == pytest.approx(500.0)
    store.close()


def test_collector_restart_does_not_integrate_gap(tmp_path):
    store = Store.open(str(tmp_path / "e.db"))
    cfg = parse_energy_config({"tou": [{"default": True, "rate": 0.10}]})
    inst = Instance("a", "http://x", gpus=(0,), locality="local")
    col1 = Collector(store, cfg)
    col1.step(1000.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    # "restart": a fresh Collector has no previous reading, so the long gap is not integrated
    col2 = Collector(store, cfg)
    col2.step(100000.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    assert store.sample_count() == 0
    col2.step(100000.0 + 3600.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    assert store.totals_gpu()[0]["kwh"] == pytest.approx(1.0)
    store.close()


def test_collector_cost_none_when_no_schedule(tmp_path):
    store = Store.open(str(tmp_path / "e.db"))
    col = Collector(store, EnergyConfig())  # no TOU -> rate unset
    inst = Instance("a", "http://x", gpus=(0,), locality="local")
    col.step(0.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    col.step(3600.0, [(inst, _snap(gpus=[(0, 1000.0)]))])
    assert store.totals_gpu()[0]["cost"] is None
    store.close()
```

Add `import pytest` at top.

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_daemon.py -q`
Expected: FAIL (`vllmstat.daemon` missing).

- [ ] **Step 3: Implement `daemon.py`**

```python
from __future__ import annotations

import asyncio
import signal
import sys
import time
from datetime import datetime

from vllmstat.config import Config
from vllmstat.core.energy import EnergyConfig, GpuEnergy, InstanceEnergy, integrate_kwh, rate_at
from vllmstat.core.state import Instance, Snapshot
from vllmstat.core.store import Store


class Collector:
    """Turns successive fleet polls into energy deltas written to the store.

    Pure of I/O timing: call ``step(now, items)`` with wall-clock ``now`` and the
    fleet's ``[(Instance, Snapshot)]``. Keeps the previous per-GPU power reading in
    memory; a fresh Collector (process restart) starts with no baseline, so downtime
    gaps are never integrated.
    """

    def __init__(self, store: Store, energy: EnergyConfig) -> None:
        self._store = store
        self._energy = energy
        self._prev_ts: float | None = None
        self._prev_power: dict[int, float] = {}
        self._prev_tokens: dict[str, float] = {}

    def step(self, now: float, items: list[tuple[Instance, Snapshot]]) -> None:
        # current per-GPU power from the host snapshot (any item carries the host slice;
        # collect the union across local instances)
        cur_power: dict[int, float] = {}
        for _inst, snap in items:
            for g in snap.gpu.gpus:
                if g.power_w is not None:
                    cur_power[g.index] = g.power_w

        rate, _label = rate_at(self._energy, datetime.fromtimestamp(now))

        if self._prev_ts is None:
            self._prev_ts, self._prev_power = now, cur_power
            self._capture_tokens(items)
            return
        dt = now - self._prev_ts
        gpu_kwh: dict[int, float] = {}
        gpu_rows: list[GpuEnergy] = []
        for idx, p1 in cur_power.items():
            p0 = self._prev_power.get(idx, p1)
            kwh = integrate_kwh(p0, p1, dt)
            cost = kwh * rate if rate is not None else None
            gpu_kwh[idx] = kwh
            gpu_rows.append(GpuEnergy(gpu_idx=idx, watts=p1, kwh=kwh, cost=cost))

        inst_rows: list[InstanceEnergy] = []
        for inst, snap in items:
            if inst.locality != "local" or not snap.gpu.gpus:
                continue
            want = set(inst.gpus) if inst.gpus else {g.index for g in snap.gpu.gpus}
            kwh = sum(gpu_kwh.get(i, 0.0) for i in want)
            cost = kwh * rate if rate is not None else None
            tok = max(0.0, snap.session_gen_tokens - self._prev_tokens.get(inst.name, snap.session_gen_tokens))
            inst_rows.append(InstanceEnergy(instance=inst.name, kwh=kwh, cost=cost, tokens=tok))

        if gpu_rows or inst_rows:
            self._store.record(now, gpu_rows, inst_rows)
        self._prev_ts, self._prev_power = now, cur_power
        self._capture_tokens(items)

    def _capture_tokens(self, items: list[tuple[Instance, Snapshot]]) -> None:
        for inst, snap in items:
            self._prev_tokens[inst.name] = snap.session_gen_tokens


async def _run_loop(cfg: Config, store: Store) -> int:
    from vllmstat.core.fleet import Fleet, InstanceRuntime
    from vllmstat.providers.gpu import GpuProvider

    runtimes = [InstanceRuntime(i) for i in cfg.instances]
    fleet = Fleet([], runtimes=runtimes)
    gpu = GpuProvider(enabled=cfg.gpu)
    col = Collector(store, cfg.energy)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - non-unix
            pass

    interval = cfg.energy.interval
    retention_s = cfg.energy.retention_days * 86400
    last_prune = 0.0
    print(f"vllmstat daemon: polling {len(runtimes)} instance(s) every {interval:g}s", flush=True)
    while not stop.is_set():
        now = time.time()
        host_gpu = gpu.sample()
        fs = await fleet.poll(host_gpu, now)
        col.step(now, fs.items)
        if now - last_prune > 3600:
            store.prune(before_ts=now - retention_s)
            last_prune = now
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
    await fleet.aclose()
    store.close()
    return 0


def run(cfg: Config) -> int:
    from vllmstat.core.service import resolve_store_path

    path = resolve_store_path(cfg, for_write=True)
    store = Store.open(path)
    try:
        return asyncio.run(_run_loop(cfg, store))
    except KeyboardInterrupt:  # pragma: no cover
        store.close()
        return 0
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_daemon.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vllmstat/daemon.py tests/test_daemon.py
git commit -m "feat(daemon): energy Collector loop (restart-safe integration)"
```

---

### Task 6: systemd unit generation + store-path resolution

**Files:**
- Create: `src/vllmstat/core/service.py`
- Test: `tests/test_service.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_service.py
import pytest

from vllmstat.config import Config
from vllmstat.core.service import (
    SYSTEM_STORE,
    USER_STORE,
    resolve_store_path,
    systemd_unit,
    unit_path,
)


def test_systemd_unit_contains_exec_and_install():
    unit = systemd_unit(exec_path="/usr/local/bin/vllmstat", system=True)
    assert "ExecStart=/usr/local/bin/vllmstat daemon run" in unit
    assert "[Service]" in unit and "WantedBy=multi-user.target" in unit
    assert "Restart=on-failure" in unit


def test_systemd_unit_user_target():
    unit = systemd_unit(exec_path="vllmstat", system=False)
    assert "WantedBy=default.target" in unit


def test_unit_path_system_vs_user():
    assert unit_path(system=True) == "/etc/systemd/system/vllmstat.service"
    assert unit_path(system=False).endswith("/.config/systemd/user/vllmstat.service")


def test_resolve_store_path_precedence(tmp_path):
    cfg = Config.from_sources([], {})
    # default (no override) -> system path for write
    assert resolve_store_path(cfg, for_write=True) == SYSTEM_STORE or USER_STORE
    # explicit config wins
    from vllmstat.core.energy import EnergyConfig
    cfg.energy = EnergyConfig(store=str(tmp_path / "x.db"))
    assert resolve_store_path(cfg, for_write=True) == str(tmp_path / "x.db")
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_service.py -q`
Expected: FAIL (`vllmstat.core.service` missing).

- [ ] **Step 3: Implement `core/service.py`**

```python
from __future__ import annotations

import os
from pathlib import Path

from vllmstat.config import Config

SYSTEM_STORE = "/var/lib/vllmstat/vllmstat.db"


def _user_store() -> str:
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return str(Path(base) / "vllmstat" / "vllmstat.db")


USER_STORE = _user_store()


def resolve_store_path(cfg: Config, *, for_write: bool) -> str:
    """`--store`/config override, else system path if usable, else user path.

    For reads the TUI tries the same order and falls back to the user path so a
    user-run TUI still finds a user-run daemon's store.
    """
    if cfg.energy.store:
        return cfg.energy.store
    # Prefer the system store if its directory exists / is writable; else user store.
    sys_dir = Path(SYSTEM_STORE).parent
    if for_write:
        if os.access(sys_dir, os.W_OK) or (not sys_dir.exists() and os.access("/var/lib", os.W_OK)):
            return SYSTEM_STORE
        return USER_STORE
    return SYSTEM_STORE if Path(SYSTEM_STORE).exists() else USER_STORE


def unit_path(*, system: bool) -> str:
    if system:
        return "/etc/systemd/system/vllmstat.service"
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return str(Path(base) / "systemd" / "user" / "vllmstat.service")


def systemd_unit(*, exec_path: str, system: bool) -> str:
    target = "multi-user.target" if system else "default.target"
    return (
        "[Unit]\n"
        "Description=vllmstat energy/stats collector\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={exec_path} daemon run\n"
        "Restart=on-failure\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        f"WantedBy={target}\n"
    )


def install_unit(*, system: bool, exec_path: str | None = None) -> str:
    """Write the unit file and return its path. Raises PermissionError without rights."""
    import shutil

    exec_path = exec_path or shutil.which("vllmstat") or "vllmstat"
    path = unit_path(system=system)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(systemd_unit(exec_path=exec_path, system=system))
    return path


def uninstall_unit(*, system: bool) -> bool:
    path = Path(unit_path(system=system))
    if path.exists():
        path.unlink()
        return True
    return False
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_service.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vllmstat/core/service.py tests/test_service.py
git commit -m "feat(daemon): systemd unit generation + store-path resolution"
```

---

### Task 7: CLI `daemon` subcommand routing

**Files:**
- Modify: `src/vllmstat/cli.py`
- Test: `tests/test_cli.py` (create if absent; otherwise append)

Keep the default (no subcommand) → TUI path untouched. Detect a leading `daemon` token in `argv` before the existing flat parser runs.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
from vllmstat.cli import main


def test_daemon_install_writes_unit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    rc = main(["daemon", "install", "--user"])
    assert rc == 0
    unit = tmp_path / "systemd" / "user" / "vllmstat.service"
    assert unit.exists() and "daemon run" in unit.read_text()
    out = capsys.readouterr().out
    assert "systemctl" in out


def test_daemon_uninstall(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    main(["daemon", "install", "--user"])
    rc = main(["daemon", "uninstall", "--user"])
    assert rc == 0
    assert not (tmp_path / "systemd" / "user" / "vllmstat.service").exists()


def test_daemon_status_no_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))  # empty -> no store
    rc = main(["daemon", "status"])
    out = capsys.readouterr().out
    assert rc == 0 and "no energy store" in out.lower()
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`
Expected: FAIL (`daemon` parsed as a URL / no routing).

- [ ] **Step 3: Add routing to `cli.py`**

At the top of `main`, before `cfg = Config.from_sources(...)`:
```python
    if argv and argv[0] == "daemon":
        return _daemon_main(argv[1:], env)
```

Add the daemon dispatcher (place above `main`):
```python
def _daemon_main(argv: list[str], env: dict[str, str]) -> int:
    import argparse

    from vllmstat.core.service import install_unit, resolve_store_path, uninstall_unit, unit_path
    from vllmstat.core.store import Store

    p = argparse.ArgumentParser(prog="vllmstat daemon")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    for name in ("install", "uninstall", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("--system", dest="system", action="store_true", default=False)
        sp.add_argument("--user", dest="user", action="store_true", default=False)
    p.add_argument("--config", dest="config_path", default=None)
    p.add_argument("--store", dest="store", default=None)
    p.add_argument("--json", dest="json", action="store_true", default=False)
    ns = p.parse_args(argv)

    cfg = Config.from_sources(
        (["--config", ns.config_path] if ns.config_path else []), env
    )
    resolve_instances(cfg, env)
    if ns.store:
        from vllmstat.core.energy import replace_store  # tiny helper, see below
        cfg.energy = replace_store(cfg.energy, ns.store)

    if ns.cmd == "run":
        from vllmstat.daemon import run
        return run(cfg)

    # install/uninstall default to --system unless --user given
    system = not ns.user if hasattr(ns, "user") else True
    if getattr(ns, "user", False):
        system = False
    elif getattr(ns, "system", False):
        system = True

    if ns.cmd == "install":
        try:
            path = install_unit(system=system)
        except PermissionError:
            print("vllmstat: need root to install a system unit (try --user or sudo)",
                  file=sys.stderr)
            return 1
        scope = "" if system else "--user "
        print(f"wrote {path}\nenable with:\n  sudo systemctl {scope}daemon-reload\n"
              f"  sudo systemctl {scope}enable --now vllmstat")
        return 0
    if ns.cmd == "uninstall":
        removed = uninstall_unit(system=system)
        print(f"removed {unit_path(system=system)}" if removed else "no unit installed")
        return 0
    # status
    path = resolve_store_path(cfg, for_write=False)
    from pathlib import Path
    if not Path(path).expanduser().exists():
        print("no energy store yet (start the daemon: vllmstat daemon run)")
        return 0
    store = Store.open(path, read_only=True)
    import time as _t
    view = store.read_view(now=_t.time(), currency=cfg.energy.currency)
    gpus = store.totals_gpu()
    store.close()
    if ns.json:
        print(json.dumps({
            "today_kwh": view.today_kwh, "today_cost": view.today_cost,
            "alltime_kwh": view.alltime_kwh, "alltime_cost": view.alltime_cost,
            "gpus": gpus,
        }, default=str))
    else:
        cur = cfg.energy.currency
        tc = f" ({cur}{view.today_cost:.2f})" if view.today_cost is not None else ""
        ac = f" ({cur}{view.alltime_cost:.2f})" if view.alltime_cost is not None else ""
        print(f"today:    {view.today_kwh:.2f} kWh{tc}")
        print(f"all-time: {view.alltime_kwh:.2f} kWh{ac}")
        for g in gpus:
            print(f"  GPU{g['gpu_idx']}: {g['kwh']:.2f} kWh")
    return 0
```

Add the tiny helper to `core/energy.py`:
```python
def replace_store(cfg: EnergyConfig, store: str) -> EnergyConfig:
    from dataclasses import replace
    return replace(cfg, store=store)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vllmstat/cli.py src/vllmstat/core/energy.py tests/test_cli.py
git commit -m "feat(daemon): vllmstat daemon run/install/uninstall/status CLI"
```

---

### Task 8: TUI integration — open store read-only, wire ENERGY panel

**Files:**
- Modify: `src/vllmstat/app.py`

This task is integration glue; verify by running the TUI against the mock provider and a pre-populated store. No new unit test is required, but a smoke assertion is added.

- [ ] **Step 1: Open the store and add the panel in `compose`/`__init__`**

In `App.__init__` (near `self._gpu = GpuProvider(...)`):
```python
        from vllmstat.core.service import resolve_store_path
        from vllmstat.core.store import Store
        from vllmstat.core.state import EnergyView
        self._energy_view = EnergyView()
        self._energy_store = None
        try:
            path = resolve_store_path(cfg, for_write=False)
            from pathlib import Path
            if Path(path).expanduser().exists():
                self._energy_store = Store.open(path, read_only=True)
        except Exception:
            self._energy_store = None
```

In `compose`, add a panel (after `self.p_eff`):
```python
        self.p_energy = Panel(id="energy")
```
and `yield self.p_energy` in the same place the other detail panels are yielded (next to `p_eff`).

- [ ] **Step 2: Refresh the view each tick**

In `_tick_body`, after `fs = await self.fleet.poll(host_gpu, now)`:
```python
        self._refresh_energy(host_gpu)
```

Add the method:
```python
    def _refresh_energy(self, host_gpu) -> None:
        from datetime import datetime
        from vllmstat.core.energy import rate_at
        from vllmstat.core.state import EnergyView
        if self._energy_store is None:
            self._energy_view = EnergyView(available=False)
            return
        import time as _t
        try:
            view = self._energy_store.read_view(now=_t.time(), currency=self.cfg.energy.currency)
        except Exception:
            view = EnergyView(available=False)
        now_w = sum(g.power_w for g in host_gpu.gpus if g.power_w) or None
        rate, label = rate_at(self.cfg.energy, datetime.now())
        view.now_w, view.rate, view.rate_label = now_w, rate, label
        self._energy_view = view
```

- [ ] **Step 3: Render the panel in `_refresh_detail`**

After the `eff` panel block:
```python
        energy = render.energy_panel(self._energy_view)
        self.p_energy.display = bool(energy)
        self.p_energy.update(energy)
```

- [ ] **Step 4: Close the store on unmount**

In `on_unmount`, add:
```python
        if self._energy_store is not None:
            self._energy_store.close()
```

- [ ] **Step 5: Smoke-verify**

Run a one-shot import + render smoke test:
```bash
.venv/bin/python -c "
from vllmstat.render import energy_panel
from vllmstat.core.state import EnergyView
print(repr(energy_panel(EnergyView(available=True, today_kwh=1.0, alltime_kwh=2.0, now_w=300, rate=0.2, rate_label='peak'))))
"
```
Expected: a two-line `ENERGY ...` string.

Then run the full suite:
```bash
.venv/bin/ruff check src tests && .venv/bin/python -m pytest -q
```
Expected: all green, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/vllmstat/app.py
git commit -m "feat(energy): wire ENERGY panel into the TUI (store read-only)"
```

---

### Task 9: Docs + version bump

**Files:**
- Modify: `src/vllmstat/__init__.py`, `pyproject.toml`, `README.md`

- [ ] **Step 1: Bump version** to `0.9.0` in `src/vllmstat/__init__.py` and `pyproject.toml` (match current style; replace the existing `0.8.0`).

- [ ] **Step 2: README** — add an "Energy & cost" section documenting: the ENERGY panel, the `[energy]` config block with a TOU example (copy from the spec), the `vllmstat daemon run/install/uninstall/status` subcommands, `--system` vs `--user`, and the store locations.

- [ ] **Step 3: Final gate**

```bash
.venv/bin/ruff check src tests && .venv/bin/python -m pytest -q
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add src/vllmstat/__init__.py pyproject.toml README.md
git commit -m "docs: energy accounting + daemon (v0.9.0)"
```

---

## Self-Review notes

- **Spec coverage:** daemon-writes/TUI-reads (Tasks 5,8); time-series+totals+daily (Task 2); TOU rate model (Task 1); per-GPU+per-instance attribution (Task 5); subcommand + system/user unit (Tasks 6,7); ENERGY panel today+all-time+live (Tasks 3,8); `[energy]` config (Task 4); restart-safe / no gap integration (Task 5 test); graceful degrade (panel empty when no store — Tasks 3,8). All covered.
- **Cost-unknown handling:** `_add_cost` keeps `None` from poisoning known sums; `cost is None` ⇒ "rate unset" in panel and `(None)` cost in totals. Consistent across store/daemon/render.
- **Type consistency:** `GpuEnergy(gpu_idx, watts, kwh, cost)` and `InstanceEnergy(instance, kwh, cost, tokens)` used identically in energy/store/daemon. `EnergyView` fields match between state/store/render/app. `resolve_store_path(cfg, for_write=...)` signature consistent across service/daemon/cli/app.
- **No new deps:** sqlite3 + tomllib are stdlib.
