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


# ---- combined (ALL) view -------------------------------------------------------------

_M2 = """\
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="{name}",engine="0"} {run}
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="{name}",engine="0"} 1.0
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="{name}",engine="0"} {gen}
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="{name}",engine="0"} {kv}
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{cache_dtype="fp8",num_gpu_blocks="{blocks}",block_size="16"} 1.0
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="{name}",le="0.1"} {b1}
vllm:time_to_first_token_seconds_bucket{model_name="{name}",le="1.0"} {b2}
vllm:time_to_first_token_seconds_bucket{model_name="{name}",le="+Inf"} {b2}
vllm:time_to_first_token_seconds_count{model_name="{name}"} {b2}
vllm:time_to_first_token_seconds_sum{model_name="{name}"} 1.0
"""


def _metrics(name, run, gen, kv, blocks, b1, b2):
    return (
        _M2.replace("{name}", name)
        .replace("{run}", str(run))
        .replace("{gen}", str(gen))
        .replace("{kv}", str(kv))
        .replace("{blocks}", str(blocks))
        .replace("{b1}", str(b1))
        .replace("{b2}", str(b2))
    )


class TwoStepProvider(FakeProvider):
    """Serves a different text on each fetch so rates and deltas are non-zero."""

    def __init__(self, texts: list[str | None]) -> None:
        super().__init__(texts[0])
        self._texts = list(texts)
        self._i = 0

    async def fetch_metrics(self) -> RawText:
        t = self._texts[min(self._i, len(self._texts) - 1)]
        self._i += 1
        if t is None:
            return RawText(text="", fetched_ok=False, error="down")
        return RawText(text=t, fetched_ok=True, error=None)


def test_merge_families_tags_instance_and_concatenates():
    from vllmstat.core.fleet import merge_families
    from vllmstat.core.parse import parse_metrics, sum_value

    fa = parse_metrics(_metrics("a", 3, 100, 0.5, 100, 1, 2))
    fb = parse_metrics(_metrics("b", 2, 50, 0.25, 300, 0, 4))
    m = merge_families([("a", fa), ("b", fb)])
    assert sum_value(m, "vllm:num_requests_running") == 5.0
    assert sum_value(m, "vllm:generation_tokens_total") == 150.0
    assert {lbl["instance"] for lbl, _ in m["vllm:num_requests_running"]} == {"a", "b"}


def test_combined_instance_gpu_union_and_locality():
    from vllmstat.core.fleet import COMBINED_NAME, combined_instance

    c = combined_instance(
        [
            Instance("a", "http://localhost:8000", gpus=(0,), locality="local"),
            Instance("b", "http://localhost:8001", gpus=(1, 2), locality="local"),
            Instance("r", "http://gpu-box:8000", locality="remote"),
        ]
    )
    assert c.name == COMBINED_NAME and c.url == "" and c.locality == "local"
    assert c.gpus == (0, 1, 2)
    # one local instance without a pin → whole host (empty mapping)
    c2 = combined_instance([Instance("a", "http://x", gpus=(0,)), Instance("b", "http://y")])
    assert c2.gpus == ()
    assert combined_instance([Instance("r", "http://z", locality="remote")]).locality == "remote"


def test_fleet_combined_sums_and_weights():
    a = InstanceRuntime(
        Instance("a", "http://localhost:8000", gpus=(0,), locality="local"),
        provider=TwoStepProvider(
            [_metrics("m-a", 3, 100, 0.5, 100, 1, 2), _metrics("m-a", 3, 200, 0.5, 100, 2, 4)]
        ),
    )
    b = InstanceRuntime(
        Instance("b", "http://localhost:8001", gpus=(1,), locality="local"),
        provider=TwoStepProvider(
            [_metrics("m-b", 2, 50, 0.25, 300, 0, 4), _metrics("m-b", 2, 150, 0.25, 300, 0, 8)]
        ),
    )
    fleet = Fleet([], runtimes=[a, b], combined=True)
    assert [rt.instance.name for rt in fleet.views][0] == "ALL"
    asyncio.run(fleet.poll(HOST, 0.0))
    fs = asyncio.run(fleet.poll(HOST, 1.0))
    assert len(fs.items) == 2 and fs.combined is not None
    inst, s = fs.combined
    assert s.connected and s.error is None
    assert s.running == 5.0 and s.waiting == 2.0
    assert s.model_names == ["m-a", "m-b"] and s.engine_count == 2
    # gen rate: (100+100 tokens)/1s smoothed by the EWMA alpha (0.3)
    assert abs(s.gen_tps - 0.3 * 200.0) < 1e-6
    # KV: capacity 100*16 + 300*16 blocks*tokens, used 0.5*1600 + 0.25*4800 → 2000/6400
    assert s.kv_capacity_tokens == 6400 and s.kv_used_tokens == 2000
    assert abs(s.kv_usage - 2000 / 6400) < 1e-9
    assert s.kv_dtype == "fp8"
    # per-server config is not meaningful fleet-wide
    assert s.max_model_len is None and s.gpu_memory_utilization is None
    # TTFT window: a saw 1 more <=0.1 and 2 total; b saw 4 more all in (0.1,1.0] → p50 > 0.1
    assert s.ttft.p50 is not None and s.ttft.p50 > 0.1
    # GPUs: union of both pins
    assert [g.index for g in s.gpu.gpus] == [0, 1]
    # history is kept for the combined runtime (drives the plots)
    assert fleet.combined is not None
    assert fleet.combined.history.series("running").values[-1] == 5.0
    assert fs.views[0][0].name == "ALL" and len(fs.views) == 3


def test_fleet_combined_reports_down_instances():
    a = InstanceRuntime(
        Instance("a", "http://localhost:8000"),
        provider=TwoStepProvider([_metrics("m-a", 3, 100, 0.5, 100, 1, 2)]),
    )
    b = InstanceRuntime(Instance("b", "http://localhost:8001"), provider=TwoStepProvider([None]))
    fleet = Fleet([], runtimes=[a, b], combined=True)
    fs = asyncio.run(fleet.poll(HOST, 0.0))
    assert fs.combined is not None
    s = fs.combined[1]
    assert s.connected is True and s.error == "down: b"
    assert s.running == 3.0 and s.model_names == ["m-a"]


def test_fleet_combined_all_down_is_disconnected():
    a = InstanceRuntime(Instance("a", "http://localhost:8000"), provider=FakeProvider(None))
    fleet = Fleet([], runtimes=[a], combined=True)
    fs = asyncio.run(fleet.poll(HOST, 0.0))
    assert fs.combined is not None and fs.combined[1].connected is False


def test_fleet_without_combined_has_no_entry():
    a = InstanceRuntime(Instance("a", "http://localhost:8000"), provider=FakeProvider(METRICS))
    fleet = Fleet([], runtimes=[a])
    fs = asyncio.run(fleet.poll(HOST, 0.0))
    assert fs.combined is None and fs.views == fs.items and fleet.views == fleet.runtimes
