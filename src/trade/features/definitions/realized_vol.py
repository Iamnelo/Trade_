"""Realized volatility feature: sample std-dev of log returns over N bars."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timedelta
from itertools import pairwise

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


class RealizedVolN:
    def __init__(self, *, window: int = 20) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._window = window
        self.spec = FeatureSpec(
            name="realized_vol",
            version=str(window),
            inputs=("close",),
            lookback_bars=window + 1,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if len(history) < self.spec.lookback_bars:
            return None
        window = history[-self.spec.lookback_bars :]
        rets = [
            math.log(b.close / a.close) for a, b in pairwise(window) if a.close > 0 and b.close > 0
        ]
        if len(rets) < 2:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var)
