from __future__ import annotations


class Rate:
    """EWMA-smoothed per-second rate of a monotonic counter, robust to resets."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.value = 0.0
        self._prev_value: float | None = None
        self._prev_t: float | None = None

    def update(self, raw: float, t: float) -> float:
        if self._prev_value is None or self._prev_t is None:
            self._prev_value, self._prev_t = raw, t
            return self.value
        dt = t - self._prev_t
        if dt <= 0:
            return self.value
        if raw < self._prev_value:  # counter reset (server restart)
            self._prev_value, self._prev_t = raw, t
            return self.value
        inst = (raw - self._prev_value) / dt
        self.value = self.alpha * inst + (1 - self.alpha) * self.value
        self._prev_value, self._prev_t = raw, t
        return self.value
