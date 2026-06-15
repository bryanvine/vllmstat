import pytest

from vllmstat.app import VllmStatApp
from vllmstat.config import Config


class FlakyVllm:
    """Returns ok once, then errors, simulating a disconnect."""

    def __init__(self):
        self.calls = 0

    async def fetch_metrics(self):
        from vllmstat.providers.vllm import RawText

        self.calls += 1
        if self.calls == 1:
            text = (
                'vllm:num_requests_running{engine="0",model_name="m"} 1.0\n'
                'vllm:generation_tokens_total{engine="0",model_name="m"} 10.0\n'
            )
            return RawText(text=text, fetched_ok=True)
        return RawText(text="", fetched_ok=False, error="connection refused")

    async def fetch_model_info(self):
        from vllmstat.providers.vllm import ModelInfo

        return ModelInfo(model_names=["m"], max_model_len=None, root=None)

    async def aclose(self): ...


@pytest.mark.asyncio
async def test_disconnect_marks_not_connected_but_keeps_running():
    cfg = Config(mock=False, interval=0.05, gpu=False)
    app = VllmStatApp(cfg)
    # Inject the flaky provider into the single instance's runtime (fleet architecture).
    app.fleet.runtimes[0]._provider = FlakyVllm()  # type: ignore[assignment]
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert app.snapshot is not None
        assert app.snapshot.connected is False  # later ticks failed
        assert app.snapshot.error is not None  # error surfaced, no crash
