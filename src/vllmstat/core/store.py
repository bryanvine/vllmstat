from __future__ import annotations

import sqlite3
from datetime import datetime
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
    def open(cls, path: str, *, read_only: bool = False) -> Store:
        """Open the energy store.

        When ``read_only=True`` the database file must already exist; callers
        guard with an existence check before opening. Opening a missing file in
        read-only mode raises ``sqlite3.OperationalError``. This method does not
        add fallback logic for an absent file — that is handled at the call site.
        """
        p = Path(path).expanduser()
        if read_only:
            conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2.0)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(p), timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
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
            self._bump_totals("totals_gpu", "gpu_idx", g.gpu_idx, ts, g.kwh, g.cost, 0.0)
            self._bump_daily(date, "gpu", str(g.gpu_idx), g.kwh, g.cost, 0.0)
        for i in instances:
            self._bump_totals(
                "totals_instance", "instance", i.instance, ts, i.kwh, i.cost, i.tokens
            )
            self._bump_daily(date, "instance", i.instance, i.kwh, i.cost, i.tokens)
        c.commit()

    def _bump_totals(self, table, keycol, keyval, ts, kwh, cost, tokens) -> None:
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
                (row["kwh"] + kwh, _add_cost(row["cost"], cost), row["tokens"] + tokens,
                 ts, keyval),
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
