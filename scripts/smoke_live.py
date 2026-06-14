#!/usr/bin/env python
"""Smoke script: render vllmtop in mock mode and save an SVG screenshot.

Usage: python scripts/smoke_live.py
Output: docs/screenshot.svg
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure the package is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vllmtop.app import VllmTopApp
from vllmtop.config import Config

OUTPUT = Path(__file__).parent.parent / "docs" / "screenshot.svg"


async def main() -> None:
    cfg = Config(mock=True, gpu=True, interval=0.2)
    app = VllmTopApp(cfg)
    async with app.run_test(size=(120, 32)) as pilot:
        # Several ticks so sparklines fill and GPUs animate.
        await pilot.pause(1.0)
        svg = app.export_screenshot()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(svg, encoding="utf-8")
    size = len(svg.encode("utf-8"))
    print(f"Screenshot saved: {OUTPUT} ({size} bytes)")

    # Verify required strings are present (SVG uses &#160; for spaces).
    import html

    decoded = html.unescape(svg)
    required = ["vllmtop", "CACHE\xa0&\xa0KV\xa0MEMORY", "GPU\xa00"]
    labels = ["vllmtop", "CACHE & KV MEMORY", "GPU 0"]
    ok = True
    for needle, label in zip(required, labels, strict=True):
        found = needle in decoded
        print(f"  {label!r}: {'OK' if found else 'MISSING'}")
        if not found:
            ok = False

    if not ok:
        print("ERROR: required strings missing from screenshot", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
