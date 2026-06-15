import pytest

from vllmstat.app import VllmStatApp
from vllmstat.config import Config
from vllmstat.core.state import Instance


@pytest.mark.asyncio
async def test_app_boots_with_mock_and_renders_cache_panel():
    cfg = Config(mock=True, interval=0.1, gpu=False)
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.3)  # allow a couple of ticks
        # the latest snapshot has been derived from mock data
        assert app.snapshot is not None
        assert app.snapshot.kv_dtype is not None
        # cache panel widget content includes the dtype
        from vllmstat.widgets import Panel

        panels = app.query(Panel)
        text = " ".join(str(p.renderable) for p in panels)
        assert "CACHE & KV MEMORY" in text


@pytest.mark.asyncio
async def test_pause_binding_stops_updates():
    cfg = Config(mock=True, interval=0.05, gpu=False)
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        await pilot.press("p")  # pause
        snap_a = app.snapshot
        await pilot.pause(0.2)
        assert app.paused is True
        assert app.snapshot is snap_a  # unchanged while paused


@pytest.mark.asyncio
async def test_app_boots_with_session_panel_and_reset_key():
    cfg = Config(mock=True, interval=0.05, gpu=False)
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        from vllmstat.widgets import Panel

        panels = app.query(Panel)
        text = " ".join(str(p.renderable) for p in panels)
        assert "SESSION" in text
        # pressing "r" resets the session without crashing
        await pilot.press("r")
        await pilot.pause(0.15)
        assert app.snapshot is not None


@pytest.mark.asyncio
async def test_single_instance_mounts_detail():
    cfg = Config(mock=True, interval=0.1, gpu=False)
    cfg.instances = [Instance("a", "http://localhost:8000")]
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert app.is_fleet is False
        assert app.query_one("#detail").display is True
        assert app.p_overview.display is False


@pytest.mark.asyncio
async def test_fleet_mounts_overview_and_drills_in():
    cfg = Config(mock=True, interval=0.1, gpu=False)
    cfg.instances = [Instance("a", "http://localhost:8000"), Instance("b", "http://localhost:8001")]
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert app.is_fleet is True
        assert app.p_overview.display is True
        await pilot.press("down")
        assert app.selected == 1
        await pilot.press("enter")
        assert app.in_detail is True
        assert app.query_one("#detail").display is True
        await pilot.press("escape")
        assert app.in_detail is False
        assert app.p_overview.display is True


@pytest.mark.asyncio
async def test_single_instance_shows_all_host_gpus():
    # Regression: a default instance has no gpus mapping; it must still show
    # every host GPU (the pre-fleet behaviour), not an empty GPU panel.
    cfg = Config(mock=True, interval=0.1, gpu=True)
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert app.snapshot is not None
        assert app.snapshot.gpu.available is True
        assert len(app.snapshot.gpu.gpus) >= 1


@pytest.mark.asyncio
async def test_tee_panel_hidden_without_source():
    cfg = Config(mock=True, interval=0.1, gpu=False)
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert app.query_one("#tee").display is False


@pytest.mark.asyncio
async def test_tee_panel_shows_with_source_and_toggles():
    cfg = Config(mock=True, interval=0.1, gpu=False)
    cfg.logs = "docker:does-not-exist"  # tailer emits a note / exits; never crashes
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert app.query_one("#tee").display is True
        await pilot.press("t")
        assert app.query_one("#tee").display is False
        await pilot.press("t")
        assert app.query_one("#tee").display is True


@pytest.mark.asyncio
async def test_proxy_starts_and_shows_tee_panel():
    cfg = Config(mock=True, interval=0.1, gpu=False)
    cfg.proxy = "127.0.0.1:0"  # ephemeral port; no client needed
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert app._proxy is not None
        assert app.query_one("#tee").display is True


@pytest.mark.asyncio
async def test_tee_panel_with_logs_after_resolve_instances():
    from vllmstat.cli import resolve_instances

    cfg = Config.from_sources(["--mock", "--logs", "docker:does-not-exist"], {})
    resolve_instances(cfg, {})
    assert cfg.instances[0].logs == "docker:does-not-exist"
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert app.query_one("#tee").display is True
        assert len(app._tailers) == 1


@pytest.mark.asyncio
async def test_proxy_starts_after_resolve_instances():
    from vllmstat.cli import resolve_instances

    cfg = Config.from_sources(
        ["--mock", "--url", "http://localhost:8000", "--proxy", "127.0.0.1:0"],
        {},
    )
    resolve_instances(cfg, {})
    assert len(cfg.instances) == 1
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert app._proxy is not None
        assert app.fleet.runtimes[0].instance.url == "http://localhost:8000"
        assert app.query_one("#tee").display is True
