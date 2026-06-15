from __future__ import annotations

import asyncio

from vllmstat.core.fleet import Fleet, InstanceRuntime, build_fleet, slice_gpu
from vllmstat.core.state import GpuSample, GpuSnapshot, Instance
from vllmstat.providers.vllm import ModelInfo, RawText

HOST = GpuSnapshot(
    available=True,
    source="multi",
    gpus=[
        GpuSample(index=0, name="A", vendor="intel", util_gpu=100.0),
        GpuSample(index=1, name="B", vendor="nvidia", util_gpu=50.0),
    ],
)


def test_slice_gpu_subset():
    sub = slice_gpu(HOST, (1,))
    assert sub.available and [g.index for g in sub.gpus] == [1]


def test_slice_gpu_empty_mapping():
    assert slice_gpu(HOST, ()).available is False


class FakeProvider:
    def __init__(self, text: str | None, ok: bool = True) -> None:
        self._text, self._ok = text, ok
        self.closed = False

    async def fetch_metrics(self) -> RawText:
        if self._text is None:
            raise RuntimeError("boom")
        return RawText(text=self._text, fetched_ok=self._ok, error=None if self._ok else "down")

    async def fetch_model_info(self) -> ModelInfo:
        return ModelInfo(model_names=["m"], max_model_len=None, root=None)

    async def aclose(self) -> None:
        self.closed = True


METRICS = '# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{model_name="m"} 3.0\n'


def test_runtime_poll_ok():
    inst = Instance(name="a", url="http://localhost:8000")
    rt = InstanceRuntime(inst, provider=FakeProvider(METRICS))
    snap = asyncio.run(rt.poll(1.0))
    assert snap.connected and snap.running == 3.0
    assert rt.history.series("running").values[-1] == 3.0


def test_fleet_poll_isolates_failures_and_slices_gpu():
    a = InstanceRuntime(
        Instance("a", "http://localhost:8000", gpus=(0,), locality="local"),
        provider=FakeProvider(METRICS),
    )
    b = InstanceRuntime(
        Instance("b", "http://gpu-box:8000", locality="remote"),
        provider=FakeProvider(None),  # raises
    )
    fleet = Fleet([], runtimes=[a, b])
    fs = asyncio.run(fleet.poll(HOST, 1.0))
    assert fs.items[0][1].connected is True
    assert [g.index for g in fs.items[0][1].gpu.gpus] == [0]  # local sliced
    assert fs.items[1][1].connected is False  # failure isolated
    assert fs.items[1][1].gpu.available is False  # remote → no gpu


def test_build_fleet_from_instances():
    fleet = build_fleet([Instance("a", "http://localhost:8000")])
    assert len(fleet.runtimes) == 1


def test_mock_fleet_returns_3_connected_items():
    from vllmstat.providers.mock import mock_fleet

    host = GpuSnapshot(
        available=True,
        source="mock",
        gpus=[
            GpuSample(index=0, name="GPU0", vendor="nvidia", util_gpu=50.0),
            GpuSample(index=1, name="GPU1", vendor="nvidia", util_gpu=60.0),
            GpuSample(index=2, name="GPU2", vendor="nvidia", util_gpu=70.0),
        ],
    )
    fleet = mock_fleet()
    assert len(fleet.runtimes) == 3
    fs = asyncio.run(fleet.poll(host, 1.0))
    connected = [item for _, item in fs.items if item.connected]
    assert len(connected) == 3
