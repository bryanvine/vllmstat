import asyncio
import socket

import httpx

from vllmstat.providers.proxy import (
    SSEAccumulator,
    TeeProxy,
    aiohttp_available,
    endpoint_for,
    extract_prompt,
    parse_json_content,
    parse_proxy_addr,
)


def test_endpoint_for():
    assert endpoint_for("/v1/chat/completions") == "chat"
    assert endpoint_for("/v1/completions") == "completions"
    assert endpoint_for("/v1/embeddings") is None


def test_extract_prompt_chat_and_completions():
    assert (
        extract_prompt(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi there"}]}
        )
        == "hi there"
    )
    multi = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {}},
                ],
            }
        ]
    }
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
        {
            "choices": [{"message": {"content": "hello there"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
        "chat",
    )
    assert text == "hello there" and pt == 5 and ct == 2


def test_parse_proxy_addr():
    assert parse_proxy_addr("9000") == ("0.0.0.0", 9000)
    assert parse_proxy_addr("127.0.0.1:9000") == ("127.0.0.1", 9000)


def test_aiohttp_available_is_bool():
    assert isinstance(aiohttp_available(), bool)


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
                await resp.write(
                    f'data: {{"choices":[{{"delta":{{"content":"{tok}"}}}}]}}\n\n'.encode()
                )
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        return web.json_response(
            {
                "choices": [{"message": {"content": "Hello there"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }
        )

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
        proxy = TeeProxy(
            upstream_url=f"http://127.0.0.1:{up_port}",
            host="127.0.0.1",
            port=px_port,
            on_event=events.append,
        )
        await proxy.start()
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"http://127.0.0.1:{px_port}/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                timeout=10,
            )
        await proxy.stop()
        await up.cleanup()
        return r, events

    r, events = asyncio.run(go())
    assert r.status_code == 200 and "Hello there" in r.text
    ex = [e for e in events if e.kind == "exchange"]
    assert ex and ex[0].prompt == "hi" and ex[0].response == "Hello there"
    assert ex[0].prompt_tokens == 5 and ex[0].done is True


def test_proxy_round_trip_streaming_accumulates():
    async def go():
        up_port, px_port = _free_port(), _free_port()
        up = await _start_upstream(up_port)
        events = []
        proxy = TeeProxy(
            upstream_url=f"http://127.0.0.1:{up_port}",
            host="127.0.0.1",
            port=px_port,
            on_event=events.append,
        )
        await proxy.start()
        chunks = []
        async with httpx.AsyncClient() as c:
            async with c.stream(
                "POST",
                f"http://127.0.0.1:{px_port}/v1/chat/completions",
                json={
                    "model": "m",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                timeout=10,
            ) as resp:
                async for ch in resp.aiter_text():
                    chunks.append(ch)
        await proxy.stop()
        await up.cleanup()
        return "".join(chunks), events

    text, events = asyncio.run(go())
    assert "[DONE]" in text
    ex = [e for e in events if e.kind == "exchange"]
    assert ex and ex[0].response == "Hello there" and ex[0].done is True
