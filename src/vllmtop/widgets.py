from __future__ import annotations

from typing import Any

from textual.widgets import Static


class Panel(Static):
    """A bordered panel whose content is set via .update(str).

    Textual 8.x dropped ``Static.renderable`` in favour of an internal Visual
    (exposed as ``.content``). We record the last value passed to ``update`` on
    ``self.renderable`` so callers/tests can read back the panel's text.
    """

    renderable: Any = ""

    def update(self, content: Any = "", *, layout: bool = True) -> None:
        self.renderable = content
        super().update(content, layout=layout)
