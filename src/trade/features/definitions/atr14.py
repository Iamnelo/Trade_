"""Average True Range (Wilder, 14-period simple-average init)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from itertools import pairwise

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


class ATR14:
    def __init__(self, *, period: int = 14) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        self._period = period
        self.spec = FeatureSpec(
            name="atr",
            version=str(period),
            inputs=("high", "low", "close"),
            lookback_bars=period + 1,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if len(history) < self.spec.lookback_bars:
            return None
        window = history[-self.spec.lookback_bars :]
        trs: list[float] = []
        for prev, cur in pairwise(window):
            tr = max(
                cur.high - cur.low,
                abs(cur.high - prev.close),
                abs(cur.low - prev.close),
            )
            trs.append(tr)
        return sum(trs) / len(trs)
