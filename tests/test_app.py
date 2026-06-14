import pytest

from vllmstat.app import VllmStatApp
from vllmstat.config import Config


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
