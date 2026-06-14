from __future__ import annotations

import time

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.timer import Timer
from textual.widgets import Footer

from vllmstat import render
from vllmstat.config import Config
from vllmstat.core.history import History
from vllmstat.core.metrics import MetricsEngine
from vllmstat.core.parse import parse_metrics
from vllmstat.core.state import Snapshot
from vllmstat.model_dims import load_model_dims
from vllmstat.providers.gpu import GpuProvider
from vllmstat.providers.mock import MockProvider, mock_gpu_snapshot
from vllmstat.providers.vllm import VllmProvider
from vllmstat.widgets import Panel


class VllmStatApp(App):
    CSS = """
    Panel { border: round $primary; padding: 0 1; height: auto; }
    #row1 { height: auto; }
    #row1 Panel { width: 1fr; }
    #gpu { height: auto; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "toggle_pause", "Pause"),
        ("g", "toggle_gpu", "GPU"),
        ("r", "reset_session", "Reset"),
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
        self._in_tick = False

    def compose(self) -> ComposeResult:
        self.p_header = Panel(id="hdr")
        self.p_conc = Panel(id="conc")
        self.p_tput = Panel(id="tput")
        self.p_lat = Panel(id="lat")
        self.p_cache = Panel(id="cache")
        self.p_session = Panel(id="session")
        self.p_eff = Panel(id="eff")
        self.p_spec = Panel(id="spec")
        self.p_gpu = Panel(id="gpu")
        yield self.p_header
        with Horizontal(id="row1"):
            yield self.p_conc
            yield self.p_tput
            yield self.p_lat
        yield self.p_cache
        yield self.p_session
        yield self.p_eff
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
        if self.paused or self._in_tick:
            return
        self._in_tick = True
        try:
            await self._tick_body()
        finally:
            self._in_tick = False

    async def _tick_body(self) -> None:
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
        if self._mock is not None and self._gpu.enabled:
            snap.gpu = mock_gpu_snapshot(self._tick_n)
        else:
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

    @staticmethod
    def _panel_width(panel: Panel) -> int | None:
        """Inner content width of a panel, or None if not laid out yet."""
        w = panel.content_size.width
        if not w:
            # Fallback before content_size is known: outer size minus border+pad.
            w = panel.size.width - 4
        return w if w > 0 else None

    def _refresh(self, s: Snapshot) -> None:
        self.p_header.update(
            render.header(s, url=self.cfg.url, interval=self.cfg.interval, uptime=self._uptime())
        )
        self.p_conc.update(
            render.concurrency(s, self._history, width=self._panel_width(self.p_conc))
        )
        self.p_tput.update(
            render.throughput(s, self._history, width=self._panel_width(self.p_tput))
        )
        self.p_lat.update(render.latency(s))
        self.p_cache.update(render.cache_kv(s, self._history))
        self.p_session.update(render.session(s))
        eff = render.efficiency(s)
        self.p_eff.display = bool(eff)
        self.p_eff.update(eff)
        spec = render.specdecode(s)
        self.p_spec.display = bool(spec)
        self.p_spec.update(spec)
        self.p_gpu.update(render.gpu(s))

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused

    def action_toggle_gpu(self) -> None:
        self._gpu.enabled = not self._gpu.enabled

    def action_reset_session(self) -> None:
        self._engine.reset_session()

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
    VllmStatApp(cfg).run()
    return 0
