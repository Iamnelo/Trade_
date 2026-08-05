"""Monotonic simulation clock.

Backtest, paper, and live all read the current time through this abstraction
so replay is deterministic and PIT-correctness can be enforced at the API
surface — no code path calls `datetime.now()` directly during a backtest.
"""

from __future__ import annotations

from datetime import datetime


class SimClock:
    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("SimClock start must be UTC-aware")
        self._now = start

    @property
    def now(self) -> datetime:
        return self._now

    def advance_to(self, ts: datetime) -> None:
        if ts.tzinfo is None:
            raise ValueError("advance target must be UTC-aware")
        if ts < self._now:
            raise ValueError(
                f"SimClock cannot move backward: {ts.isoformat()} < {self._now.isoformat()}"
            )
        self._now = ts
