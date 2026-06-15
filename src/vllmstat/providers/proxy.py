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
        parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        joined = " ".join(t for t in parts if t)
        return joined or None
    return None


def extract_prompt(path: str, body: dict) -> str | None:
    ep = endpoint_for(path)
    if ep == "chat":
        msgs = body.get("messages") or []
        return _content_text(msgs[-1].get("content")) if msgs else None
    if ep == "completions":
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
        if endpoint == "chat":
            text = ((c.get("message") or {}).get("content")) or ""
        else:
            text = c.get("text") or ""
    usage = body.get("usage") or {}
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens")


class SSEAccumulator:
    """Accumulates an OpenAI streaming (SSE) response; tolerant of chunk-split lines."""

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
