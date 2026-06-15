import json
from pathlib import Path

import httpx

from vllmstat.providers.vllm import VllmProvider

FIXDIR = Path(__file__).parent / "fixtures"


def _client() -> httpx.AsyncClient:
    metrics = (FIXDIR / "metrics_qwen3.txt").read_text()
    models = json.loads((FIXDIR / "models.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text=metrics)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=models)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://t")


async def test_fetch_metrics_ok():
    p = VllmProvider(base_url="http://t", client=_client())
    raw = await p.fetch_metrics()
    assert raw.fetched_ok is True
    assert "vllm:num_requests_running" in raw.text


async def test_fetch_models_extracts_root_and_ctx():
    p = VllmProvider(base_url="http://t", client=_client())
    info = await p.fetch_model_info()
    assert info.model_names == ["qwen3-30b-tq"]
    assert info.max_model_len == 262144
    assert info.root and "Qwen3-30B" in info.root


async def test_fetch_metrics_error_sets_flag():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(boom), base_url="http://t")
    p = VllmProvider(base_url="http://t", client=client)
    raw = await p.fetch_metrics()
    assert raw.fetched_ok is False
    assert raw.error is not None


async def test_fetch_model_info_error_sets_flag():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(boom), base_url="http://t")
    p = VllmProvider(base_url="http://t", client=client)
    info = await p.fetch_model_info()
    assert info.model_names == []
    assert info.error is not None
