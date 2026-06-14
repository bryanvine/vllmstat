from __future__ import annotations

from collections import deque


class Series:
    def __init__(self, maxlen: int = 120) -> None:
        self.values: deque[float] = deque(maxlen=maxlen)

    def push(self, v: float) -> None:
        self.values.append(v)


class History:
    def __init__(self, maxlen: int = 120) -> None:
        self._maxlen = maxlen
        self._series: dict[str, Series] = {}

    def series(self, name: str) -> Series:
        if name not in self._series:
            self._series[name] = Series(self._maxlen)
        return self._series[name]

    def push(self, name: str, value: float) -> None:
        self.series(name).push(value)
