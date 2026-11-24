"""Clock utilities for deterministic testing."""

from __future__ import annotations

import time
from collections.abc import Callable


def system_clock_ns() -> int:
    """Wall-clock nanoseconds."""
    return time.time_ns()


class ControllableClock:
    """Mutable clock for offline tests."""

    def __init__(self, start_ns: int = 1_700_000_000_000_000_000) -> None:
        self._now = start_ns

    def __call__(self) -> int:
        return self._now

    def advance(self, ns: int) -> int:
        self._now += ns
        return self._now

    def set(self, ns: int) -> None:
        self._now = ns


def elapsed_ms(start_ns: int, clock: Callable[[], int]) -> float:
    return max(0.0, (clock() - start_ns) / 1_000_000.0)
