from __future__ import annotations

import asyncio
import signal
import time
from datetime import datetime

from vllmstat.config import Config
from vllmstat.core.energy import EnergyConfig, GpuEnergy, InstanceEnergy, integrate_kwh, rate_at
from vllmstat.core.state import Instance, Snapshot
from vllmstat.core.store import Store


class Collector:
    """Turns successive fleet polls into energy deltas written to the store.

    Call ``step(now, items)`` with wall-clock ``now`` (epoch seconds) and the fleet's
    ``[(Instance, Snapshot)]``. Keeps the previous per-GPU power reading in memory; a
    fresh Collector (process restart) starts with no baseline, so downtime gaps are
    never integrated.
    """

    def __init__(self, store: Store, energy: EnergyConfig) -> None:
        self._store = store
        self._energy = energy
        self._prev_ts: float | None = None
        self._prev_power: dict[int, float] = {}
        self._prev_tokens: dict[str, float] = {}

    def step(self, now: float, items: list[tuple[Instance, Snapshot]]) -> None:
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
            prev_tok = self._prev_tokens.get(inst.name, snap.session_gen_tokens)
            tok = max(0.0, snap.session_gen_tokens - prev_tok)
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
