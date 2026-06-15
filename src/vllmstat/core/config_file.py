from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


def parse_config(text: str) -> tuple[list[dict], dict]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"invalid TOML: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("config root must be a table")
    raw = data.get("instance", [])
    if not isinstance(raw, list):
        raise ValueError("'instance' must be an array of tables ([[instance]])")
    globals_ = {k: v for k, v in data.items() if k != "instance"}
    return raw, globals_


def load_config(path: str) -> tuple[list[dict], dict]:
    return parse_config(Path(path).expanduser().read_text())


def find_config(
    explicit: str | None,
    env: dict[str, str],
    *,
    candidates: list[str] | None = None,
    exists=None,
) -> str | None:
    exists = exists or (lambda p: Path(p).expanduser().is_file())
    if explicit:
        return explicit
    if env.get("VLLMSTAT_CONFIG"):
        return env["VLLMSTAT_CONFIG"]
    for c in candidates or ["./vllmstat.toml", "~/.config/vllmstat/config.toml"]:
        if exists(c):
            return c
    return None
