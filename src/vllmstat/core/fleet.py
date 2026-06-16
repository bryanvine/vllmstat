from __future__ import annotations

import asyncio
from typing import Any

from vllmstat.core.history import History
from vllmstat.core.metrics import MetricsEngine
from vllmstat.core.parse import parse_metrics
from vllmstat.core.state import FleetSnapshot, GpuSnapshot, Instance, Snapshot
from vllmstat.core.tee import TeeBuffer
from vllmstat.model_dims import load_model_dims
from vllmstat.providers.vllm import VllmProvider

# Generation rate (tok/s) above which the server counts as "actively generating".
# Below it, gen_tps is just an EWMA residual after inference stops, so per-token
# energy figures would be meaningless — efficiency is held, not recomputed.
_EFF_MIN_TPS = 1.0


def slice_gpu(host: GpuSnapshot, gpus: tuple[int, ...]) -> GpuSnapshot:
    """Return a GpuSnapshot restricted to the GPU indices a local instance uses.

    An empty *gpus* means "no explicit mapping" → show the whole host (matches
    single-instance behaviour, where every GPU is shown). When *gpus* is given,
    return only those indices. An unavailable host stays unavailable.
    """
    if not host.available:
        return GpuSnapshot(available=False, source=host.source)
    if not gpus:
        return host
    want = set(gpus)
    sub = [g for g in host.gpus if g.index in want]
    return GpuSnapshot(available=bool(sub), source=host.source, gpus=sub)


class InstanceRuntime:
    """Wraps one vLLM instance with its own metrics engine, history, and provider."""

    def __init__(self, instance: Instance, *, provider: Any = None) -> None:
        self.instance = instance
        self._provider: Any = provider or VllmProvider(
            base_url=instance.url,
            metrics_path=instance.metrics_path,
            api_key=instance.api_key,
        )
        self._engine = MetricsEngine()
        self.history: History = History()
        self.tee: TeeBuffer = TeeBuffer()
        self.snapshot: Snapshot | None = None
        self.model_names: list[str] = []
        self._dims_loaded = False
        self._idle_w_sum = 0.0
        self._idle_w_n = 0
        self._eff_tokw_sum = 0.0
        self._eff_jpt_sum = 0.0
        self._eff_n = 0

    async def _ensure_dims(self) -> None:
        if self._dims_loaded:
            return
        self._dims_loaded = True
        info = await self._provider.fetch_model_info()
        md = load_model_dims(info.root, info.max_model_len)
        self._engine = MetricsEngine(dims=md.dims, max_model_len=md.max_model_len)
        self.model_names = info.model_names

    async def poll(self, now: float) -> Snapshot:
        """Fetch metrics, derive a Snapshot, and push to history.  Never raises."""
        await self._ensure_dims()
        raw = await self._provider.fetch_metrics()
        if raw.fetched_ok and raw.text:
            snap = self._engine.derive(parse_metrics(raw.text), now=now)
        else:
            prev = self.snapshot
            snap = prev if prev is not None else Snapshot(ts=now, connected=False, error=raw.error)
            snap.connected = False
            snap.error = raw.error
        self.snapshot = snap
        self._push_history(snap)
        return snap

    def _push_history(self, s: Snapshot) -> None:
        self.history.push("running", s.running)
        self.history.push("waiting", s.waiting)
        self.history.push("gen_tps", s.gen_tps)
        self.history.push("prompt_tps", s.prompt_tps)
        if s.prefix_hit_window is not None:
            self.history.push("prefix_hit", s.prefix_hit_window)

    def record_idle_power(self, running: float, power_w: float | None) -> float | None:
        """Accumulate the mean GPU power while idle (running == 0); return the running mean."""
        if power_w and running == 0:
            self._idle_w_sum += power_w
            self._idle_w_n += 1
        return (self._idle_w_sum / self._idle_w_n) if self._idle_w_n else None

    def record_efficiency(
        self, gen_tps: float, power_w: float | None
    ) -> tuple[float | None, float | None]:
        """Accumulate session-mean tokens/W and J/token while actively generating.

        Samples count only while ``gen_tps >= _EFF_MIN_TPS`` and power is known, so
        the averages stop updating (but stay shown) once the server goes idle —
        rather than decaying toward an EWMA residual or dividing by ~0. Returns the
        current means, or ``(None, None)`` until the first active sample.
        """
        if power_w and gen_tps >= _EFF_MIN_TPS:
            self._eff_tokw_sum += gen_tps / power_w
            self._eff_jpt_sum += power_w / gen_tps
            self._eff_n += 1
        if self._eff_n == 0:
            return None, None
        return self._eff_tokw_sum / self._eff_n, self._eff_jpt_sum / self._eff_n

    def reset_session(self) -> None:
        self._engine.reset_session()
        self._idle_w_sum = 0.0
        self._idle_w_n = 0
        self._eff_tokw_sum = 0.0
        self._eff_jpt_sum = 0.0
        self._eff_n = 0

    async def aclose(self) -> None:
        await self._provider.aclose()


class Fleet:
    """A collection of InstanceRuntimes polled concurrently."""

    def __init__(
        self,
        instances: list[Instance],
        *,
        runtimes: list[InstanceRuntime] | None = None,
    ) -> None:
        self.runtimes: list[InstanceRuntime] = (
            runtimes if runtimes is not None else [InstanceRuntime(i) for i in instances]
        )

    async def poll(self, host_gpu: GpuSnapshot, now: float) -> FleetSnapshot:
        """Poll all runtimes concurrently; isolate failures; attach GPU slices."""
        results: list[Any] = list(
            await asyncio.gather(*(rt.poll(now) for rt in self.runtimes), return_exceptions=True)
        )
        items: list[tuple[Instance, Snapshot]] = []
        for rt, res in zip(self.runtimes, results, strict=True):
            if isinstance(res, BaseException):
                prev = rt.snapshot
                res = (
                    prev if prev is not None else Snapshot(ts=now, connected=False, error=str(res))
                )
                res.connected = False
            if rt.instance.locality == "local":
                res.gpu = slice_gpu(host_gpu, rt.instance.gpus)
            else:
                res.gpu = GpuSnapshot(available=False, source="remote")
            pw = sum(g.power_w for g in res.gpu.gpus if g.power_w) or None
            res.idle_watts_avg = rt.record_idle_power(res.running, pw)
            res.tokens_per_watt, res.joules_per_token = rt.record_efficiency(res.gen_tps, pw)
            items.append((rt.instance, res))
        return FleetSnapshot(ts=now, items=items, gpu=host_gpu)

    async def aclose(self) -> None:
        await asyncio.gather(*(rt.aclose() for rt in self.runtimes), return_exceptions=True)


def build_fleet(instances: list[Instance]) -> Fleet:
    """Construct a Fleet from a list of Instance configs."""
    return Fleet(instances)
