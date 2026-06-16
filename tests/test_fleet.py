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


def test_slice_gpu_empty_mapping_shows_whole_host():
    # No explicit gpus mapping → show every host GPU (single-instance behaviour).
    sub = slice_gpu(HOST, ())
    assert sub.available and [g.index for g in sub.gpus] == [0, 1]


def test_slice_gpu_unavailable_host_stays_unavailable():
    assert slice_gpu(GpuSnapshot(available=False), (0,)).available is False


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


def test_runtime_poll_ok_when_model_info_fails():
    class Provider(FakeProvider):
        async def fetch_model_info(self) -> ModelInfo:
            return ModelInfo(model_names=[], max_model_len=None, root=None, error="models down")

    inst = Instance(name="a", url="http://localhost:8000")
    rt = InstanceRuntime(inst, provider=Provider(METRICS))
    snap = asyncio.run(rt.poll(1.0))
    assert snap.connected and snap.running == 3.0


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


def test_runtime_has_tee_buffer():
    from vllmstat.core.fleet import InstanceRuntime
    from vllmstat.core.state import Instance
    from vllmstat.core.tee import TeeBuffer

    rt = InstanceRuntime(Instance("a", "http://x"))
    assert isinstance(rt.tee, TeeBuffer) and len(rt.tee) == 0


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


def test_runtime_idle_power_average():
    rt = InstanceRuntime(Instance("a", "http://x"))
    assert rt.record_idle_power(0.0, 30.0) == 30.0
    assert rt.record_idle_power(5.0, 200.0) == 30.0  # active sample (running>0) ignored
    assert rt.record_idle_power(0.0, 34.0) == 32.0  # mean of idle samples (30+34)/2
    rt.reset_session()
    assert rt.record_idle_power(5.0, 200.0) is None  # reset cleared the accumulator


def test_runtime_efficiency_average_holds_when_idle():
    rt = InstanceRuntime(Instance("a", "http://x"))
    # active: 1400 tok/s @ 200 W -> 7.0 tok/W
    tokw, jpt = rt.record_efficiency(1400.0, 200.0)
    assert abs(tokw - 7.0) < 1e-9 and abs(jpt - 200.0 / 1400.0) < 1e-9
    # idle sample (gen_tps below the floor) is ignored -> averages held, not recomputed
    assert rt.record_efficiency(0.01, 200.0) == (tokw, jpt)
    # another active sample at 1000 tok/s @ 200 W -> 5.0 tok/W; session mean (7+5)/2 = 6.0
    tokw3, _ = rt.record_efficiency(1000.0, 200.0)
    assert abs(tokw3 - 6.0) < 1e-9
    rt.reset_session()
    assert rt.record_efficiency(0.01, 200.0) == (None, None)  # reset + idle -> no data yet
