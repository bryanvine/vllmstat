import asyncio

from vllmstat.core.tee import TeeEvent
from vllmstat.providers.logsource import LogTailer, parse_access_line

LINE = '(APIServer pid=1) INFO:     172.18.0.1:47402 - "POST /v1/chat/completions HTTP/1.1" 200 OK'


def test_parse_access_line_real():
    e = parse_access_line(LINE, now=1.0)
    assert e and e.kind == "http" and e.method == "POST"
    assert e.path == "/v1/chat/completions" and e.status == 200 and e.client == "172.18.0.1"


def test_parse_filters_health_and_metrics():
    assert parse_access_line('INFO: 1.2.3.4:5 - "GET /health HTTP/1.1" 200 OK', now=1.0) is None
    assert parse_access_line('x - "GET /metrics HTTP/1.1" 200 OK', now=1.0) is None


def test_parse_4xx_and_nonaccess():
    e = parse_access_line('h 9.9.9.9:1 - "POST /v1/completions HTTP/1.1" 400 Bad Request', now=1.0)
    assert e and e.status == 400
    assert parse_access_line("Engine 000: Avg prompt throughput: 0.0 tokens/s", now=1.0) is None


def test_parse_ipv6_client():
    line = 'INFO: [::1]:54321 - "POST /v1/chat/completions HTTP/1.1" 200 OK'
    e = parse_access_line(line, now=1.0)
    assert e and e.client == "::1" and e.method == "POST" and e.status == 200


def test_logtailer_follows_file(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("")
    got = []
    tail = LogTailer(
        str(p),
        on_event=got.append,
        parse=lambda ln: TeeEvent(ts=0.0, kind="note", text=ln) if ln else None,
    )

    async def go():
        tail.start()
        await asyncio.sleep(0.1)
        with p.open("a") as f:
            f.write("hello\n")
            f.flush()
        await asyncio.sleep(0.5)
        await tail.stop()

    asyncio.run(go())
    assert any(e.text == "hello" for e in got)


def test_logtailer_survives_observer_exception(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("")
    got = []
    n_calls = 0

    def boom(ev):
        nonlocal n_calls
        n_calls += 1
        if n_calls == 1:
            raise RuntimeError("observer broke")
        got.append(ev)

    tail = LogTailer(
        str(p),
        on_event=boom,
        parse=lambda ln: TeeEvent(ts=0.0, kind="note", text=ln) if ln else None,
    )

    async def go():
        tail.start()
        await asyncio.sleep(0.1)
        with p.open("a") as f:
            f.write("first\nsecond\n")
            f.flush()
        await asyncio.sleep(0.5)
        await tail.stop()

    asyncio.run(go())
    assert n_calls >= 2
    assert any(e.text == "second" for e in got)
