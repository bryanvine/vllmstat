from __future__ import annotations

import time

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.timer import Timer
from textual.widgets import Footer

from vllmtop import render
from vllmtop.config import Config
from vllmtop.core.history import History
from vllmtop.core.metrics import MetricsEngine
from vllmtop.core.parse import parse_metrics
from vllmtop.core.state import Snapshot
from vllmtop.model_dims import load_model_dims
from vllmtop.providers.gpu import GpuProvider
from vllmtop.providers.mock import MockProvider
from vllmtop.providers.vllm import VllmProvider
from vllmtop.widgets import Panel


class VllmTopApp(App):
    CSS = """
    Panel { border: round $primary; padding: 0 1; height: auto; }
    #row1 { height: auto; } #gpu { height: auto; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "toggle_pause", "Pause"),
        ("g", "toggle_gpu", "GPU"),
        ("plus,equals_sign", "faster", "Faster"),
        ("minus", "slower", "Slower"),
    ]

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.paused = False
        self.snapshot: Snapshot | None = None
        self._history = History()
        self._engine = MetricsEngine()
        self._gpu = GpuProvider(enabled=cfg.gpu)
        self._mock = MockProvider() if cfg.mock else None
        self._vllm = (
            None
            if cfg.mock
            else VllmProvider(base_url=cfg.url, metrics_path=cfg.metrics_path, api_key=cfg.api_key)
        )
        self._model_names: list[str] = []
        self._dims_loaded = cfg.mock  # mock keeps the plain engine; nothing to fetch
        self._start = time.monotonic()
        self._tick_n = 0
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        self.p_header = Panel(id="hdr")
        self.p_conc = Panel(id="conc")
        self.p_tput = Panel(id="tput")
        self.p_lat = Panel(id="lat")
        self.p_cache = Panel(id="cache")
        self.p_spec = Panel(id="spec")
        self.p_gpu = Panel(id="gpu")
        yield self.p_header
        with Horizontal(id="row1"):
            yield self.p_conc
            yield self.p_tput
            yield self.p_lat
        yield self.p_cache
        yield self.p_spec
        yield self.p_gpu
        yield Footer()

    def on_mount(self) -> None:
        self._timer = self.set_interval(self.cfg.interval, self.tick)
        self.call_later(self.tick)

    async def _ensure_dims(self) -> None:
        """Once, in vLLM mode, fetch model info and rebuild the engine with KV dims."""
        if self._dims_loaded or self._vllm is None:
            return
        self._dims_loaded = True  # set before await so we only attempt once
        info = await self._vllm.fetch_model_info()
        md = load_model_dims(info.root, info.max_model_len)
        self._engine = MetricsEngine(dims=md.dims, max_model_len=md.max_model_len)
        self._model_names = info.model_names

    async def _sample_text(self) -> tuple[str, bool, str | None]:
        if self._mock is not None:
            return self._mock.metrics_text(), True, None
        assert self._vllm is not None
        raw = await self._vllm.fetch_metrics()
        return raw.text, raw.fetched_ok, raw.error

    async def tick(self) -> None:
        if self.paused:
            return
        await self._ensure_dims()
        self._tick_n += 1
        text, ok, err = await self._sample_text()
        now = time.monotonic()
        if ok and text:
            fam = parse_metrics(text)
            snap = self._engine.derive(fam, now=now)
        else:
            snap = self.snapshot or Snapshot(ts=now, connected=False, error=err)
            snap.connected = False
            snap.error = err
        snap.gpu = self._gpu.sample()
        self._push_history(snap)
        self.snapshot = snap
        self._refresh(snap)

    def _push_history(self, s: Snapshot) -> None:
        self._history.push("running", s.running)
        self._history.push("waiting", s.waiting)
        self._history.push("gen_tps", s.gen_tps)
        self._history.push("prompt_tps", s.prompt_tps)
        if s.prefix_hit_window is not None:
            self._history.push("prefix_hit", s.prefix_hit_window)

    def _uptime(self) -> str:
        secs = int(time.monotonic() - self._start)
        h, rem = divmod(secs, 3600)
        m, _ = divmod(rem, 60)
        return f"{h}h{m:02d}m"

    def _refresh(self, s: Snapshot) -> None:
        self.p_header.update(
            render.header(s, url=self.cfg.url, interval=self.cfg.interval, uptime=self._uptime())
        )
        self.p_conc.update(render.concurrency(s, self._history))
        self.p_tput.update(render.throughput(s, self._history))
        self.p_lat.update(render.latency(s))
        self.p_cache.update(render.cache_kv(s, self._history))
        spec = render.specdecode(s)
        self.p_spec.display = bool(spec)
        self.p_spec.update(spec)
        self.p_gpu.update(render.gpu(s))

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused

    def action_toggle_gpu(self) -> None:
        self._gpu.enabled = not self._gpu.enabled

    def action_faster(self) -> None:
        self.cfg.interval = max(0.1, self.cfg.interval / 2)
        self._reschedule()

    def action_slower(self) -> None:
        self.cfg.interval = min(10.0, self.cfg.interval * 2)
        self._reschedule()

    def _reschedule(self) -> None:
        # restart the interval timer at the new cadence (Textual >=8: Timer.stop + recreate)
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(self.cfg.interval, self.tick)


def run_app(cfg: Config) -> int:
    VllmTopApp(cfg).run()
    return 0
