from __future__ import annotations

import time

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Footer

from vllmstat import render
from vllmstat.config import Config
from vllmstat.core.fleet import Fleet, InstanceRuntime
from vllmstat.core.history import History
from vllmstat.core.resolve import derive_name
from vllmstat.core.state import FleetSnapshot, GpuSnapshot, Instance, Snapshot
from vllmstat.core.tee import TeeEvent
from vllmstat.providers.gpu import GpuProvider
from vllmstat.providers.logsource import LogTailer
from vllmstat.providers.mock import MockProvider, MockVllmProvider, mock_gpu_snapshot
from vllmstat.providers.proxy import TeeProxy, aiohttp_available, parse_proxy_addr
from vllmstat.widgets import Panel


class VllmStatApp(App):
    CSS = """
    Panel { border: round $primary; padding: 0 1; height: auto; }
    #row1 { height: auto; }
    #row1 Panel { width: 1fr; }
    #gpu { height: auto; }
    #overview { height: auto; }
    #tee { height: auto; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "toggle_pause", "Pause"),
        ("g", "toggle_gpu", "GPU"),
        ("t", "toggle_tee", "Tee"),
        ("r", "reset_session", "Reset"),
        ("up,k", "cursor_up", "Up"),
        ("down,j", "cursor_down", "Down"),
        ("enter", "drill_in", "Open"),
        ("escape", "back", "Back"),
        ("plus,equals_sign", "faster", "Faster"),
        ("minus", "slower", "Slower"),
    ]

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.paused = False
        self.selected = 0
        instances = cfg.instances or [
            Instance(
                name=derive_name(cfg.url),
                url=cfg.url,
                metrics_path=cfg.metrics_path,
                api_key=cfg.api_key,
                gpus=(),
                locality="local",
                logs=cfg.logs,
            )
        ]
        self.is_fleet = len(instances) > 1
        self.in_detail = not self.is_fleet
        self._gpu = GpuProvider(enabled=cfg.gpu)
        self._mock = cfg.mock
        if cfg.mock:
            runtimes = [
                InstanceRuntime(i, provider=MockVllmProvider(MockProvider())) for i in instances
            ]
        else:
            runtimes = [InstanceRuntime(i) for i in instances]
        self.fleet = Fleet([], runtimes=runtimes)
        self.fleet_snapshot: FleetSnapshot | None = None
        self.snapshot: Snapshot | None = None
        self._start = time.monotonic()
        self._tick_n = 0
        self._timer: Timer | None = None
        self._in_tick = False
        self.tee_visible = True
        self._tailers: list[LogTailer] = []
        self._proxy: TeeProxy | None = None
        self._proxy_desc = ""
        if cfg.proxy:
            host, port = parse_proxy_addr(cfg.proxy)
            rt0 = self.fleet.runtimes[0]
            self._proxy = TeeProxy(
                upstream_url=rt0.instance.url,
                host=host,
                port=port,
                on_event=rt0.tee.push,
                api_key=rt0.instance.api_key,
            )
            self._proxy_desc = f"proxy :{port} → {rt0.instance.url}"

    def compose(self) -> ComposeResult:
        self.p_overview = Panel(id="overview")
        yield self.p_overview
        self.p_header = Panel(id="hdr")
        self.p_conc = Panel(id="conc")
        self.p_tput = Panel(id="tput")
        self.p_lat = Panel(id="lat")
        self.p_cache = Panel(id="cache")
        self.p_shape = Panel(id="shape")
        self.p_context = Panel(id="context")
        self.p_session = Panel(id="session")
        self.p_outcomes = Panel(id="outcomes")
        self.p_eff = Panel(id="eff")
        self.p_spec = Panel(id="spec")
        self.p_gpu = Panel(id="gpu")
        self.p_tee = Panel(id="tee")
        with Vertical(id="detail"):
            yield self.p_header
            with Horizontal(id="row1"):
                yield self.p_conc
                yield self.p_tput
                yield self.p_lat
            yield self.p_cache
            yield self.p_shape
            yield self.p_context
            yield self.p_session
            yield self.p_outcomes
            yield self.p_eff
            yield self.p_spec
            yield self.p_gpu
            yield self.p_tee
        yield Footer()

    async def on_mount(self) -> None:
        self._apply_mode()
        self.p_tee.display = False
        for rt in self.fleet.runtimes:
            if rt.instance.logs:
                tailer = LogTailer(rt.instance.logs, on_event=rt.tee.push)
                tailer.start()
                self._tailers.append(tailer)
        self._timer = self.set_interval(self.cfg.interval, self.tick)
        self.call_later(self.tick)
        if self._proxy is not None:
            rt0 = self.fleet.runtimes[0]
            if not aiohttp_available():
                rt0.tee.push(
                    TeeEvent(
                        ts=time.time(),
                        kind="note",
                        text="proxy needs aiohttp — pip install 'vllmstat[proxy]'",
                    )
                )
                self._proxy = None
                self._proxy_desc = ""
            else:
                try:
                    await self._proxy.start()
                except Exception as e:  # noqa: BLE001
                    rt0.tee.push(TeeEvent(ts=time.time(), kind="note", text=f"proxy failed: {e}"))
                    await self._proxy.stop()  # release the httpx client on start failure
                    self._proxy = None
                    self._proxy_desc = ""

    async def on_unmount(self) -> None:
        for tailer in self._tailers:
            tailer.terminate()
        if self._proxy is not None:
            await self._proxy.stop()

    def _apply_mode(self) -> None:
        self.p_overview.display = self.is_fleet and not self.in_detail
        self.query_one("#detail").display = self.in_detail

    async def tick(self) -> None:
        if self.paused or self._in_tick:
            return
        self._in_tick = True
        try:
            await self._tick_body()
        finally:
            self._in_tick = False

    async def _tick_body(self) -> None:
        self._tick_n += 1
        now = time.monotonic()
        if self._mock and self._gpu.enabled:
            host_gpu = mock_gpu_snapshot(self._tick_n)
        elif self._gpu.enabled:
            host_gpu = self._gpu.sample()
        else:
            host_gpu = GpuSnapshot()
        fs = await self.fleet.poll(host_gpu, now)
        self.fleet_snapshot = fs
        if fs.items:
            idx = min(self.selected, len(fs.items) - 1)
            self.snapshot = fs.items[idx][1]
        self._refresh()

    def _refresh(self) -> None:
        if self.fleet_snapshot is None:
            return
        if self.is_fleet and not self.in_detail:
            self.p_overview.update(
                render.fleet_overview(
                    self.fleet_snapshot,
                    self.selected,
                    width=self._panel_width(self.p_overview),
                    uptime=self._uptime(),
                    interval=self.cfg.interval,
                    show_gpu=self._gpu.enabled,
                )
            )
        else:
            inst, snap, hist = self._current()
            self._refresh_detail(inst, snap, hist)

    def _current(self) -> tuple[Instance, Snapshot, History]:
        assert self.fleet_snapshot is not None
        idx = min(self.selected, len(self.fleet_snapshot.items) - 1)
        inst, snap = self.fleet_snapshot.items[idx]
        hist = self.fleet.runtimes[idx].history
        return inst, snap, hist

    def _refresh_detail(self, inst: Instance, snap: Snapshot, hist: History) -> None:
        if self.is_fleet:
            self.p_header.update(
                render.detail_header(inst, snap, interval=self.cfg.interval, uptime=self._uptime())
            )
        else:
            self.p_header.update(
                render.header(snap, url=inst.url, interval=self.cfg.interval, uptime=self._uptime())
            )
        self.p_conc.update(render.concurrency(snap, hist, width=self._panel_width(self.p_conc)))
        self.p_tput.update(render.throughput(snap, hist, width=self._panel_width(self.p_tput)))
        self.p_lat.update(render.latency(snap))
        self.p_cache.update(render.cache_kv(snap, hist))
        shape = render.request_shape(snap)
        self.p_shape.display = bool(shape)
        self.p_shape.update(shape)
        ctx = render.context_window(snap)
        self.p_context.display = bool(ctx)
        self.p_context.update(ctx)
        self.p_session.update(render.session(snap))
        outcomes = render.outcomes(snap)
        self.p_outcomes.display = bool(outcomes)
        self.p_outcomes.update(outcomes)
        eff = render.efficiency(snap)
        self.p_eff.display = bool(eff)
        self.p_eff.update(eff)
        spec = render.specdecode(snap)
        self.p_spec.display = bool(spec)
        self.p_spec.update(spec)
        self.p_gpu.update(render.gpu(snap))
        rt = self.fleet.runtimes[min(self.selected, len(self.fleet.runtimes) - 1)]
        has_tee = bool(inst.logs) or self._proxy is not None or len(rt.tee) > 0
        self.p_tee.display = has_tee and self.tee_visible
        if self.p_tee.display:
            source = self._proxy_desc or inst.logs or "—"
            self.p_tee.update(
                render.tee(
                    rt.tee.recent(40),
                    width=self._panel_width(self.p_tee),
                    source_desc=source,
                    height=12,
                )
            )

    def _uptime(self) -> str:
        secs = int(time.monotonic() - self._start)
        h, rem = divmod(secs, 3600)
        m, _ = divmod(rem, 60)
        return f"{h}h{m:02d}m"

    @staticmethod
    def _panel_width(panel: Panel) -> int | None:
        w = panel.content_size.width
        if not w:
            w = panel.size.width - 4
        return w if w > 0 else None

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused

    def action_toggle_gpu(self) -> None:
        self._gpu.enabled = not self._gpu.enabled
        self._refresh()

    def action_toggle_tee(self) -> None:
        self.tee_visible = not self.tee_visible
        self._refresh()

    def action_reset_session(self) -> None:
        idx = min(self.selected, len(self.fleet.runtimes) - 1)
        self.fleet.runtimes[idx].reset_session()

    def action_cursor_up(self) -> None:
        if self.is_fleet and not self.in_detail and self.selected > 0:
            self.selected -= 1
            self._refresh()

    def action_cursor_down(self) -> None:
        if self.is_fleet and not self.in_detail and self.selected < len(self.fleet.runtimes) - 1:
            self.selected += 1
            self._refresh()

    def action_drill_in(self) -> None:
        if self.is_fleet and not self.in_detail:
            self.in_detail = True
            self._apply_mode()
            self._refresh()

    def action_back(self) -> None:
        if self.is_fleet and self.in_detail:
            self.in_detail = False
            self._apply_mode()
            self._refresh()

    def action_faster(self) -> None:
        self.cfg.interval = max(0.1, self.cfg.interval / 2)
        self._reschedule()

    def action_slower(self) -> None:
        self.cfg.interval = min(10.0, self.cfg.interval * 2)
        self._reschedule()

    def _reschedule(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(self.cfg.interval, self.tick)


def run_app(cfg: Config) -> int:
    app = VllmStatApp(cfg)
    try:
        app.run()
    finally:
        for tailer in app._tailers:
            tailer.terminate()
    return 0
