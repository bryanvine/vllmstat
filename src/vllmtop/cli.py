from __future__ import annotations

import json
import os
import sys
import time

from vllmtop.config import Config
from vllmtop.core.metrics import MetricsEngine
from vllmtop.core.parse import parse_metrics
from vllmtop.providers.mock import MockProvider
from vllmtop.snapshot_json import snapshot_to_dict


def run_once_json(argv: list[str]) -> int:
    cfg = Config.from_sources(argv, dict(os.environ))
    eng = MetricsEngine(dims=None, max_model_len=None)
    if cfg.mock:
        mp = MockProvider()
        eng.derive(parse_metrics(mp.metrics_text()), now=0.0)
        snap = eng.derive(parse_metrics(mp.metrics_text()), now=1.0)
    else:
        import asyncio

        from vllmtop.providers.vllm import VllmProvider

        async def _go():
            p = VllmProvider(base_url=cfg.url, metrics_path=cfg.metrics_path, api_key=cfg.api_key)
            r0 = await p.fetch_metrics()
            time.sleep(min(cfg.interval, 1.0))
            r1 = await p.fetch_metrics()
            await p.aclose()
            return r0, r1

        r0, r1 = asyncio.run(_go())
        if not r1.fetched_ok:
            print(json.dumps({"error": r1.error}), file=sys.stderr)
            return 1
        eng.derive(parse_metrics(r0.text), now=0.0)
        snap = eng.derive(parse_metrics(r1.text), now=1.0)
    print(json.dumps(snapshot_to_dict(snap), default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = Config.from_sources(argv, dict(os.environ))
    if cfg.once and cfg.json:
        return run_once_json(argv)
    from vllmtop.app import run_app  # imported lazily so --once/--json needs no Textual

    return run_app(cfg)
