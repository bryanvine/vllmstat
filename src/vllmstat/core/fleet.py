from __future__ import annotations

import asyncio
from typing import Any

from vllmstat.core.history import History
from vllmstat.core.metrics import MetricsEngine
from vllmstat.core.parse import Families, parse_metrics
from vllmstat.core.state import FleetSnapshot, GpuSnapshot, Instance, Snapshot
from vllmstat.core.tee import TeeBuffer
from vllmstat.model_dims import load_model_dims
from vllmstat.providers.vllm import VllmProvider

# Generation rate (tok/s) above which the server counts as "actively generating".
# Below it, gen_tps is just an EWMA residual after inference stops, so per-token
# energy figures would be meaningless — efficiency is held, not recomputed.
_EFF_MIN_TPS = 1.0

COMBINED_NAME = "ALL"


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
        # Last successfully parsed /metrics families (None while down): feeds the combined view.
        self.families: Families | None = None
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
            self.families = parse_metrics(raw.text)
            snap = self._engine.derive(self.families, now=now)
        else:
            self.families = None
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


def merge_families(parts: list[tuple[str, Families]]) -> Families:
    """Concatenate several instances' families, tagging each sample with an ``instance`` label.

    Counters and histogram buckets then sum naturally in the engine (it already sums
    across ``model_name``/``engine`` labels), so rates, session totals, and latency
    quantiles come out fleet-wide. Ratio gauges and per-server config are fixed up
    afterwards by :meth:`CombinedRuntime.derive`.
    """
    merged: Families = {}
    for name, fam in parts:
        for sample, rows in fam.items():
            merged.setdefault(sample, []).extend(({**lbl, "instance": name}, v) for lbl, v in rows)
    return merged


def combined_instance(instances: list[Instance]) -> Instance:
    """The synthetic Instance that stands for every instance at once.

    GPUs are the union of the local instances' pins; any local instance without an
    explicit pin means "whole host" (matches :func:`slice_gpu`).
    """
    local = [i for i in instances if i.locality == "local"]
    gpus: tuple[int, ...] = ()
    if local and all(i.gpus for i in local):
        gpus = tuple(sorted({g for i in local for g in i.gpus}))
    return Instance(
        name=COMBINED_NAME,
        url="",
        gpus=gpus,
        locality="local" if local else "remote",
    )


class _NoProvider:
    async def fetch_model_info(self):  # pragma: no cover - never called
        raise RuntimeError("combined runtime has no provider")

    async def fetch_metrics(self):  # pragma: no cover - never called
        raise RuntimeError("combined runtime has no provider")

    async def aclose(self) -> None:
        return None


class CombinedRuntime(InstanceRuntime):
    """One MetricsEngine fed the merged families of every instance in the fleet.

    It is derived from the real runtimes' last poll rather than polled itself, so it
    carries its own history, session accumulators, and efficiency means like any
    other runtime and renders through the unchanged detail dashboard.
    """

    def __init__(self, instance: Instance) -> None:
        super().__init__(instance, provider=_NoProvider())
        self._dims_loaded = True

    async def poll(self, now: float) -> Snapshot:
        raise RuntimeError("CombinedRuntime is derived via derive(), not polled")

    def derive(
        self, parts: list[tuple[Instance, Families | None, Snapshot]], now: float
    ) -> Snapshot:
        live = [(inst, fam, snap) for inst, fam, snap in parts if fam is not None]
        down = [inst.name for inst, fam, _ in parts if fam is None]
        if not live:
            snap = Snapshot(ts=now, connected=False, error="no instances reachable")
        else:
            snap = self._engine.derive(merge_families([(i.name, f) for i, f, _ in live]), now=now)
            self._fixup(snap, [s for _, _, s in live])
            if down:
                snap.error = "down: " + ", ".join(down)
        self.snapshot = snap
        self._push_history(snap)
        return snap

    @staticmethod
    def _fixup(snap: Snapshot, snaps: list[Snapshot]) -> None:
        """Repair fields that don't sum: engine count, KV usage, and per-server config."""
        snap.engine_count = sum(s.engine_count for s in snaps)
        # Per-server config drives the advisor and context ratios; they only make sense
        # for one server, so the combined view leaves them unset.
        snap.max_model_len = None
        snap.gpu_memory_utilization = None
        snap.prefix_caching_enabled = None
        snap.kv_fp16_full_ctx_gb = None
        caps = [s.kv_capacity_tokens for s in snaps if s.kv_capacity_tokens]
        used = [s.kv_used_tokens for s in snaps if s.kv_used_tokens is not None]
        cap_total = sum(caps) if caps else None
        used_total = sum(used) if used else None
        snap.kv_capacity_tokens = cap_total
        snap.kv_used_tokens = used_total
        if cap_total and used_total is not None:
            snap.kv_usage = min(1.0, used_total / cap_total)
        else:
            snap.kv_usage = sum(s.kv_usage for s in snaps) / len(snaps)
        dtypes = {s.kv_dtype for s in snaps if s.kv_dtype}
        if len(dtypes) == 1:
            snap.kv_dtype = dtypes.pop()
            ratios = {(s.kv_ratio, s.kv_ratio_kind) for s in snaps}
            if len(ratios) != 1:
                snap.kv_ratio, snap.kv_ratio_kind = None, "none"
            eq = [s.kv_fp16_equiv_tokens for s in snaps if s.kv_fp16_equiv_tokens]
            snap.kv_fp16_equiv_tokens = sum(eq) if eq else None
        else:
            snap.kv_dtype = "mixed" if dtypes else None
            snap.kv_ratio, snap.kv_ratio_kind = None, "none"
            snap.kv_fp16_equiv_tokens = None


class Fleet:
    """A collection of InstanceRuntimes polled concurrently."""

    def __init__(
        self,
        instances: list[Instance],
        *,
        runtimes: list[InstanceRuntime] | None = None,
        combined: bool = False,
    ) -> None:
        self.runtimes: list[InstanceRuntime] = (
            runtimes if runtimes is not None else [InstanceRuntime(i) for i in instances]
        )
        self.combined: CombinedRuntime | None = (
            CombinedRuntime(combined_instance([rt.instance for rt in self.runtimes]))
            if combined
            else None
        )

    @property
    def views(self) -> list[InstanceRuntime]:
        """Runtimes in UI order: the combined one first (when enabled), then each instance."""
        return ([self.combined] if self.combined is not None else []) + self.runtimes

    @staticmethod
    def _attach_host(rt: InstanceRuntime, res: Snapshot, host_gpu: GpuSnapshot) -> None:
        """Slice host GPUs onto a snapshot and fold power into the runtime's session means."""
        if rt.instance.locality == "local":
            res.gpu = slice_gpu(host_gpu, rt.instance.gpus)
        else:
            res.gpu = GpuSnapshot(available=False, source="remote")
        pw = sum(g.power_w for g in res.gpu.gpus if g.power_w) or None
        res.idle_watts_avg = rt.record_idle_power(res.running, pw)
        res.tokens_per_watt, res.joules_per_token = rt.record_efficiency(res.gen_tps, pw)

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
                rt.families = None
            self._attach_host(rt, res, host_gpu)
            items.append((rt.instance, res))
        combined = None
        if self.combined is not None:
            parts = [
                (rt.instance, rt.families, snap)
                for rt, (_, snap) in zip(self.runtimes, items, strict=True)
            ]
            csnap = self.combined.derive(parts, now)
            self._attach_host(self.combined, csnap, host_gpu)
            combined = (self.combined.instance, csnap)
        return FleetSnapshot(ts=now, items=items, gpu=host_gpu, combined=combined)

    async def aclose(self) -> None:
        await asyncio.gather(*(rt.aclose() for rt in self.runtimes), return_exceptions=True)


def build_fleet(instances: list[Instance]) -> Fleet:
    """Construct a Fleet from a list of Instance configs."""
    return Fleet(instances)
