# Energy accounting + long-term daemon — Design

**Date:** 2026-06-24
**Status:** Approved

## Goal

Add total-energy counters (kWh + cost at adjustable, time-of-use electricity rates)
and an installable headless daemon that records vLLM/GPU stats long term so the
totals survive restarts and accrue even when the TUI is closed.

## Decisions (from brainstorming)

- **Architecture:** Daemon writes, TUI reads. A headless `vllmstat daemon` polls on
  its own interval and writes to a local SQLite store; the TUI opens that store
  read-only. The daemon is optional for the TUI (degrades to a session-only estimate).
- **Data model:** Time-series samples + rolled-up lifetime totals, plus a daily
  downsample that survives sample retention and powers the "today" figures.
- **Rate model:** Time-of-use (TOU) schedule. Samples store rate-agnostic kWh; cost is
  computed per-sample from the schedule and is recomputable if the schedule changes.
- **Granularity:** Per-GPU measured truth, rolled up to each local instance via its
  GPU mapping. Remote instances (no local GPU data) get no energy figure.
- **Install:** `vllmstat daemon` subcommand group; `install` generates a systemd unit,
  default `--system` (root, `/etc/systemd/system`), with a `--user` alternative.
- **TUI display:** An ENERGY panel showing today + all-time kWh & cost, live draw, and
  the current effective rate/label.

## Architecture & data flow

```
systemd ─▶ vllmstat daemon ──poll(Fleet)──▶ integrate ──▶ vllmstat.db (WAL)
                                                              ▲ read-only
                                            vllmstat (TUI) ───┘
```

The daemon builds the same `Fleet`/provider/GPU stack as the TUI (no duplicate
sampling logic), polls every `interval` seconds (default 10), integrates per-GPU
watts into energy, computes cost from the TOU schedule, and writes to SQLite.

## Energy & cost math (`core/energy.py`, pure)

- Between polls at t0,t1 with GPU power p0,p1:
  `kWh = ((p0+p1)/2) · (t1−t0)/3600/1000` (trapezoidal).
- On daemon restart the gap is **not** integrated (power while down is unknown);
  energy only accrues while the daemon runs.
- **TOU schedule:** ordered rules, each with `days` (`mon-fri`, `sat-sun`, `mon-sun`,
  single days), `from`/`to` (`HH:MM`; overnight windows where from>to wrap midnight),
  `rate`, optional `label`. Exactly one rule must be `default = true` (fallback). The
  rate for a sample is chosen by the sample's timestamp in the **daemon host's local
  timezone** (DST handled automatically). If no `[energy]` config exists, kWh is still
  recorded and cost is left null (TUI shows kWh + "rate unset").

## Store schema (`core/store.py`, SQLite WAL)

```
samples(ts, gpu_idx, watts, kwh, cost)            -- per poll per GPU; pruned after retention_days (default 7)
daily(date, scope, key, kwh, cost, tokens)        -- downsample target; kept forever; powers "today"/history
totals_gpu(gpu_idx, kwh, cost, tokens, since_ts, updated_ts)
totals_instance(instance, kwh, cost, tokens, since_ts, updated_ts)
meta(key, value)                                  -- schema version, last-poll state
```

- `scope` in `daily` is `"gpu"` or `"instance"`; `key` is the GPU index or instance name.
- Raw samples give full resolution within the retention window; `daily` is the
  downsample that survives pruning; `totals_*` are lifetime counters.
- Per-GPU is measured; per-instance is derived via each local instance's GPU mapping.
- WAL mode lets the daemon write while the TUI reads.

## CLI / daemon management (`cli.py`, `daemon.py`, `core/service.py`)

`vllmstat` with no subcommand → TUI (unchanged). New `daemon` subcommand group:

- `vllmstat daemon run` — foreground collector (any host; used for testing/non-systemd).
- `vllmstat daemon install [--system|--user]` — generate a systemd unit (default
  `--system`, `/etc/systemd/system/vllmstat.service`, needs root; `--user` →
  `~/.config/systemd/user/`) and print enable/start commands.
- `vllmstat daemon uninstall` — remove the generated unit.
- `vllmstat daemon status` — totals + last poll + per-GPU breakdown; honors `--json`.

**Store path resolution** (daemon writes, TUI reads, same order): `--store` flag ›
`[energy].store` › system `/var/lib/vllmstat/vllmstat.db` › user
`~/.local/state/vllmstat/vllmstat.db`. A system DB is made world-readable so a
user-run TUI can read it.

## Config (`[energy]` in the existing config file)

```toml
[energy]
currency = "$"
store = "/var/lib/vllmstat/vllmstat.db"   # optional
interval = 10                              # daemon poll seconds, optional
retention_days = 7                         # raw sample retention, optional

[[energy.tou]]
days = "mon-fri"
from = "16:00"
to   = "21:00"
rate = 0.42
label = "peak"

[[energy.tou]]
default = true
rate = 0.12
label = "off-peak"
```

Validation: rates ≥ 0; if any `[[energy.tou]]` rules are present, exactly one must be
`default = true`; `from`/`to` must be `HH:MM`. Invalid config is reported and ignored
(consistent with existing config-file handling), leaving cost unset rather than crashing.

## TUI panel (`render.py`, `app.py`, `state.py`)

A new `EnergyView` read struct + `energy_panel` renderer, refreshed from the store
every few seconds:

```
ENERGY  today 2.4 kWh ($0.43)  ·  all-time 318 kWh ($57.2)
        now 412 W  ·  rate $0.18/kWh (off-peak)
```

Degrades gracefully: no store → session-only kWh estimate from live power;
rate unset → kWh only (no cost).

## Components / files

- `core/energy.py` (new) — pure: power→energy integration; TOU schedule parse + rate
  lookup; cost calc; `EnergyConfig` dataclass.
- `core/store.py` (new) — SQLite schema/migrations, `record_sample`, `upsert_totals`,
  daily rollup, `prune`, read queries (`totals`, `today`, per-GPU breakdown).
- `daemon.py` (new) — headless collector loop (build fleet, poll, integrate, write;
  signal handling; restart-safe resume).
- `core/service.py` (new) — systemd unit-file generation (system/user), install/uninstall.
- `cli.py` (modify) — `daemon` subcommand routing while keeping no-subcommand → TUI.
- `core/config_file.py` (modify) — parse `[energy]` table into `EnergyConfig`.
- `render.py` (modify) — `energy_panel`.
- `app.py` (modify) — wire ENERGY panel, periodic store read.
- `core/state.py` (modify) — `EnergyView` read struct.

## Testing

- `energy.py`: trapezoid integration, kWh conversion, TOU lookup (day-ranges, overnight
  windows, default fallback, DST), cost calc. Pure → TDD.
- `store.py`: tmp SQLite — sample→totals accumulation, daily rollup, retention prune,
  read queries, concurrent read under WAL.
- `daemon.py`: `FakeProvider` + injected GPU snapshot + injected clock — energy accrues;
  restart resumes without integrating the downtime gap.
- `core/service.py`: assert generated unit-file contents (no real `systemctl`).
- `render.py` / `cli.py`: panel formatting (incl. degrade paths); `daemon` routing,
  install dry-run.
