# vllmstat v0.5.0 — Tee Phase 2 (proxy content tee) — Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD, checkbox steps.
> Spec: `docs/superpowers/specs/2026-06-15-tee-design.md` (Phase 2).

**Goal:** `vllmstat --proxy [HOST:]PORT` runs a reverse proxy in front of the (single) vLLM instance.
Clients point at it; it forwards everything to vLLM, relays responses **unchanged** (incl. streaming
SSE), and tees the **prompt + completion** into the existing TEE panel as `exchange` events. Optional
dep `aiohttp` (extra `vllmstat[proxy]`); upstream forwarding reuses the core `httpx` dep.

**Conventions:** Python ≥3.10, `from __future__ import annotations`, ruff+pyright clean, pytest
(`source .venv/bin/activate`; `aiohttp` is already installed in the dev venv). Commit per task.
`TeeEvent`/`TeeBuffer`/`render.tee` (with the `exchange` kind) already exist from v0.4.0.

---

## Task 1: pyproject extra + pure proxy helpers

**Files:** Modify `pyproject.toml`; Create `src/vllmstat/providers/proxy.py`; Test `tests/test_proxy.py`.

- [ ] `pyproject.toml`: add an optional extra and put `aiohttp` in dev:
```toml
[project.optional-dependencies]
proxy = ["aiohttp>=3.9"]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "ruff>=0.5", "pyright>=1.1", "aiohttp>=3.9"]
```

- [ ] Tests `tests/test_proxy.py`:
```python
from vllmstat.providers.proxy import (
    SSEAccumulator, aiohttp_available, endpoint_for, extract_prompt, parse_json_content,
    parse_proxy_addr,
)

def test_endpoint_for():
    assert endpoint_for("/v1/chat/completions") == "chat"
    assert endpoint_for("/v1/completions") == "completions"
    assert endpoint_for("/v1/embeddings") is None

def test_extract_prompt_chat_and_completions():
    assert extract_prompt("/v1/chat/completions",
                          {"messages": [{"role": "user", "content": "hi there"}]}) == "hi there"
    multi = {"messages": [{"role": "user", "content": [{"type": "text", "text": "look"},
                                                       {"type": "image_url", "image_url": {}}]}]}
    assert extract_prompt("/v1/chat/completions", multi) == "look"
    assert extract_prompt("/v1/completions", {"prompt": "once upon"}) == "once upon"

def test_sse_accumulator_chat_split_chunks():
    acc = SSEAccumulator("chat")
    acc.feed('data: {"choices":[{"delta":{"content":"Hel')
    acc.feed('lo"}}]}\n\ndata: {"choices":[{"delta":{"content":" there"}}]}\n\n')
    acc.feed("data: [DONE]\n\n")
    assert acc.text == "Hello there" and acc.done is True

def test_sse_accumulator_completions():
    acc = SSEAccumulator("completions")
    acc.feed('data: {"choices":[{"text":"abc"}]}\n\ndata: [DONE]\n\n')
    assert acc.text == "abc" and acc.done is True

def test_parse_json_content_chat_with_usage():
    text, pt, ct = parse_json_content(
        {"choices": [{"message": {"content": "hello there"}}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2}}, "chat")
    assert text == "hello there" and pt == 5 and ct == 2

def test_parse_proxy_addr():
    assert parse_proxy_addr("9000") == ("0.0.0.0", 9000)
    assert parse_proxy_addr("127.0.0.1:9000") == ("127.0.0.1", 9000)

def test_aiohttp_available_is_bool():
    assert isinstance(aiohttp_available(), bool)
```

- [ ] Implement `src/vllmstat/providers/proxy.py` (helpers only this task; `aiohttp` imported lazily
  later so importing this module never requires aiohttp):
```python
from __future__ import annotations

import json


def endpoint_for(path: str) -> str | None:
    if "chat/completions" in path:
        return "chat"
    if path.endswith("/completions"):
        return "completions"
    return None


def _content_text(content) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("type") == "text"]
        joined = " ".join(t for t in parts if t)
        return joined or None
    return None


def extract_prompt(path: str, body: dict) -> str | None:
    if endpoint_for(path) == "chat":
        msgs = body.get("messages") or []
        return _content_text(msgs[-1].get("content")) if msgs else None
    if endpoint_for(path) == "completions":
        p = body.get("prompt")
        if isinstance(p, list):
            return " ".join(str(x) for x in p)
        return p if isinstance(p, str) else None
    return None


def parse_json_content(body: dict, endpoint: str) -> tuple[str, int | None, int | None]:
    choices = body.get("choices") or []
    text = ""
    if choices:
        c = choices[0]
        text = ((c.get("message") or {}).get("content") if endpoint == "chat" else c.get("text")) or ""
    usage = body.get("usage") or {}
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens")


class SSEAccumulator:
    """Accumulates an OpenAI streaming (SSE) response, tolerant of chunk-split lines."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.text = ""
        self.done = False
        self._buf = ""

    def feed(self, chunk: str) -> None:
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._consume(line.strip())

    def _consume(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            self.done = True
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return
        choices = obj.get("choices") or []
        if not choices:
            return
        c = choices[0]
        if self.endpoint == "chat":
            self.text += (c.get("delta") or {}).get("content") or ""
        else:
            self.text += c.get("text") or ""


def parse_proxy_addr(s: str) -> tuple[str, int]:
    if ":" in s:
        host, _, port = s.rpartition(":")
        return (host or "0.0.0.0", int(port))
    return ("0.0.0.0", int(s))


def aiohttp_available() -> bool:
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False
```
- [ ] Run `pytest tests/test_proxy.py -q`; full checks clean; commit: `feat(proxy): pure helpers (prompt/SSE/json parsing, addr) + [proxy] extra`.

---

## Task 2: `TeeProxy` server

**Files:** Modify `src/vllmstat/providers/proxy.py` (add the class); Test `tests/test_proxy.py` (add
round-trip tests).

- [ ] Add round-trip tests (use a stub upstream aiohttp server; `aiohttp` is installed in dev):
```python
import asyncio
import socket

import httpx
from vllmstat.providers.proxy import TeeProxy


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _start_upstream(port: int):
    from aiohttp import web

    async def chat(request):
        body = await request.json()
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for tok in ["Hello", " there"]:
                await resp.write(f'data: {{"choices":[{{"delta":{{"content":"{tok}"}}}}]}}\n\n'.encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        return web.json_response(
            {"choices": [{"message": {"content": "Hello there"}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def test_proxy_round_trip_json_captures_exchange():
    async def go():
        up_port, px_port = _free_port(), _free_port()
        up = await _start_upstream(up_port)
        events = []
        proxy = TeeProxy(upstream_url=f"http://127.0.0.1:{up_port}",
                         host="127.0.0.1", port=px_port, on_event=events.append)
        await proxy.start()
        async with httpx.AsyncClient() as c:
            r = await c.post(f"http://127.0.0.1:{px_port}/v1/chat/completions",
                             json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                             timeout=10)
        await proxy.stop()
        await up.cleanup()
        return r, events

    r, events = asyncio.run(go())
    assert r.status_code == 200 and "Hello there" in r.text          # passthrough intact
    ex = [e for e in events if e.kind == "exchange"]
    assert ex and ex[0].prompt == "hi" and ex[0].response == "Hello there"
    assert ex[0].prompt_tokens == 5 and ex[0].done is True


def test_proxy_round_trip_streaming_accumulates():
    async def go():
        up_port, px_port = _free_port(), _free_port()
        up = await _start_upstream(up_port)
        events = []
        proxy = TeeProxy(upstream_url=f"http://127.0.0.1:{up_port}",
                         host="127.0.0.1", port=px_port, on_event=events.append)
        await proxy.start()
        chunks = []
        async with httpx.AsyncClient() as c:
            async with c.stream("POST", f"http://127.0.0.1:{px_port}/v1/chat/completions",
                                json={"model": "m", "stream": True,
                                      "messages": [{"role": "user", "content": "hi"}]},
                                timeout=10) as resp:
                async for ch in resp.aiter_text():
                    chunks.append(ch)
        await proxy.stop()
        await up.cleanup()
        return "".join(chunks), events

    text, events = asyncio.run(go())
    assert "[DONE]" in text                                          # SSE relayed to client
    ex = [e for e in events if e.kind == "exchange"]
    assert ex and ex[0].response == "Hello there" and ex[0].done is True
```

- [ ] Implement `TeeProxy` in `proxy.py` (aiohttp imported lazily inside methods; upstream via httpx;
  `accept-encoding: identity` so raw chunks are plaintext for both passthrough and parsing; the
  `exchange` event is pushed immediately and **mutated in place** as the response streams):
```python
import time
from collections.abc import Callable

import httpx

from vllmstat.core.tee import TeeEvent

_HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding"}


class TeeProxy:
    def __init__(self, *, upstream_url: str, host: str, port: int,
                 on_event: Callable[[TeeEvent], None], api_key: str | None = None) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.host = host
        self.port = port
        self._on_event = on_event
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=None)
        self._runner = None

    async def start(self) -> None:
        from aiohttp import web

        app = web.Application(client_max_size=0)  # 0 = unlimited request body
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        await self._client.aclose()

    async def _handle(self, request):
        from aiohttp import web

        body = await request.read()
        path = request.path
        endpoint = endpoint_for(path)
        streaming, prompt = False, None
        if endpoint and body:
            try:
                req = json.loads(body)
                prompt = extract_prompt(path, req)
                streaming = bool(req.get("stream"))
            except (ValueError, AttributeError):
                pass
        event = None
        if endpoint:
            event = TeeEvent(ts=time.time(), kind="exchange", endpoint=endpoint,
                             prompt=prompt, response="", streaming=streaming, done=False)
            self._on_event(event)

        fwd = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
        fwd["accept-encoding"] = "identity"
        if self._api_key and not any(k.lower() == "authorization" for k in fwd):
            fwd["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with self._client.stream(
                request.method, self.upstream_url + path,
                content=body or None, headers=fwd,
                params=request.query_string or None,
            ) as up:
                out_headers = {k: v for k, v in up.headers.items() if k.lower() not in _HOP}
                resp = web.StreamResponse(status=up.status_code, headers=out_headers)
                await resp.prepare(request)
                acc = SSEAccumulator(endpoint) if (endpoint and streaming) else None
                raw = bytearray()
                async for chunk in up.aiter_raw():
                    await resp.write(chunk)
                    if event is not None:
                        if acc is not None:
                            acc.feed(chunk.decode("utf-8", "replace"))
                            event.response, event.done = acc.text, acc.done
                        elif not streaming:
                            raw += chunk
                await resp.write_eof()
                if event is not None and not streaming:
                    try:
                        text, pt, ct = parse_json_content(json.loads(bytes(raw)), endpoint)
                        event.response, event.prompt_tokens, event.completion_tokens = text, pt, ct
                    except (ValueError, KeyError, TypeError):
                        pass
                    event.done = True
                return resp
        except Exception as e:  # noqa: BLE001 - never crash; surface upstream errors
            if event is not None:
                event.response, event.done = f"[proxy error: {e}]", True
            return web.Response(status=502, text=f"vllmstat proxy: {e}")
```
- [ ] Run `pytest tests/test_proxy.py -q` (round-trips included); full checks clean; commit:
  `feat(proxy): streaming reverse-proxy that tees prompts + completions`.

---

## Task 3: app wiring — `--proxy`, lifecycle, panel source

**Files:** Modify `src/vllmstat/config.py`, `src/vllmstat/app.py`; Test `tests/test_config.py`,
`tests/test_app.py`.

- [ ] `config.py`: add `proxy: str | None = None`; arg `p.add_argument("--proxy", dest="proxy",
  default=None)`; set in `from_sources`. Test `--proxy 9000` → `cfg.proxy == "9000"`.
- [ ] `app.py`:
  - imports: `from vllmstat.providers.proxy import TeeProxy, aiohttp_available, parse_proxy_addr`,
    and `from vllmstat.core.tee import TeeEvent`, `import time` (if not already).
  - `__init__`: after building `self.fleet`, set up the proxy against the first runtime:
    ```python
    self._proxy = None
    self._proxy_desc = ""
    if cfg.proxy:
        host, port = parse_proxy_addr(cfg.proxy)
        rt0 = self.fleet.runtimes[0]
        self._proxy = TeeProxy(upstream_url=rt0.instance.url, host=host, port=port,
                               on_event=rt0.tee.push, api_key=rt0.instance.api_key)
        self._proxy_desc = f"proxy :{port} → {rt0.instance.url}"
    ```
  - make `on_mount` **async** (`async def on_mount`) — keep its existing body (apply mode, timer,
    call_later(tick), start tailers), then start the proxy:
    ```python
    if self._proxy is not None:
        rt0 = self.fleet.runtimes[0]
        if not aiohttp_available():
            rt0.tee.push(TeeEvent(ts=time.time(), kind="note",
                                  text="proxy needs aiohttp — pip install 'vllmstat[proxy]'"))
            self._proxy = None
        else:
            try:
                await self._proxy.start()
            except Exception as e:  # noqa: BLE001
                rt0.tee.push(TeeEvent(ts=time.time(), kind="note", text=f"proxy failed: {e}"))
                self._proxy = None
    ```
  - make `on_unmount` **async**: keep `for t in self._tailers: t.terminate()`, then
    `if self._proxy is not None: await self._proxy.stop()`.
  - `_refresh_detail`: when computing the tee panel, fold the proxy in:
    ```python
    has_tee = bool(inst.logs) or self._proxy is not None or len(rt.tee) > 0
    self.p_tee.display = has_tee and self.tee_visible
    if self.p_tee.display:
        source = self._proxy_desc or inst.logs or "—"
        self.p_tee.update(render.tee(rt.tee.recent(40),
                                     width=self._panel_width(self.p_tee),
                                     source_desc=source, height=12))
    ```
    (Proxy is bound to the first runtime; in single mode that's the shown instance. Document that
    `--proxy` targets the single/first instance for v0.5.)
- [ ] Tests (`tests/test_app.py`):
```python
@pytest.mark.asyncio
async def test_proxy_starts_and_shows_tee_panel():
    cfg = Config(mock=True, interval=0.1, gpu=False)
    cfg.proxy = "127.0.0.1:0"          # ephemeral port, no client needed
    app = VllmStatApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert app._proxy is not None
        assert app.query_one("#tee").display is True
```
- [ ] Run `pytest tests/test_app.py tests/test_config.py -q` then full `pytest -q`; all prior tests
  green; checks clean; commit: `feat(proxy): --proxy flag, lifecycle, and TEE panel source`.

---

## Task 4 (controller): docs, version, screenshot, live test, release

- [ ] Bump to `0.5.0` (pyproject + `__init__`).
- [ ] README: expand the **Tee** section — proxy mode (`pip install 'vllmstat[proxy]'`,
  `vllmstat --proxy 9000 --url http://localhost:8000`, point open-webui/clients at it), full
  prompt/response tee, the privacy note (content renders in your local terminal only); add `--proxy`
  to the flags table; update the "what it does not show yet" caveat now that content IS shown via proxy.
- [ ] New screenshot of the TEE panel with `exchange` (▶ prompt / ◀ response) events.
- [ ] Live test: `vllmstat --proxy 9000 --url http://localhost:8000` in one shell; `curl` a chat
  request through `:9000` in another; confirm the prompt+response appear in the panel (or do it
  headlessly via `run_test` + an httpx call through the proxy, like the Phase-1 live check).
- [ ] Full `ruff && pyright && pytest` green; final review; merge to main; build; twine upload `0.5.0`.

## Self-review notes
- The proxy reuses `TeeEvent(kind="exchange")` + `render.tee` from v0.4.0 — same panel, richer source.
- `aiohttp` is optional: importing `providers/proxy` never imports aiohttp (lazy inside `start`/`_handle`);
  missing aiohttp degrades to a `note`, never a crash.
- `accept-encoding: identity` guarantees raw chunks are plaintext → correct passthrough AND parsing.
- Proxy targets the single/first instance (fleet-wide proxy is out of scope for v0.5).
