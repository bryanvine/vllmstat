from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class RawText:
    text: str
    fetched_ok: bool
    error: str | None = None


@dataclass
class ModelInfo:
    model_names: list[str]
    max_model_len: int | None
    root: str | None
    error: str | None = None


class VllmProvider:
    def __init__(
        self,
        *,
        base_url: str,
        metrics_path: str = "/metrics",
        api_key: str | None = None,
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.metrics_path = metrics_path
        self._timeout = timeout
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(base_url=self.base_url, headers=headers)

    async def fetch_metrics(self) -> RawText:
        try:
            r = await self._client.get(self.metrics_path, timeout=self._timeout)
            r.raise_for_status()
            return RawText(text=r.text, fetched_ok=True)
        except Exception as e:  # noqa: BLE001 - surface any transport/HTTP error to UI
            return RawText(text="", fetched_ok=False, error=str(e))

    async def fetch_model_info(self) -> ModelInfo:
        try:
            r = await self._client.get("/v1/models", timeout=self._timeout)
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception as e:  # noqa: BLE001
            return ModelInfo(model_names=[], max_model_len=None, root=None, error=str(e))
        names = [d.get("id") for d in data if d.get("id")]
        max_len = next((d.get("max_model_len") for d in data if d.get("max_model_len")), None)
        root = next((d.get("root") for d in data if d.get("root")), None)
        return ModelInfo(model_names=names, max_model_len=max_len, root=root)

    async def aclose(self) -> None:
        await self._client.aclose()
