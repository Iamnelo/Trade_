"""Rolling z-score of bar volume and turnover.

    volume_zscore@N   = (v_t - mean(v[t-N..t])) / std(v[t-N..t])
    turnover_zscore@N = same, on turnover

Anomalous liquidity is a well-known precursor to regime shifts and
directional moves. Volume vs turnover are complementary — turnover
scales with price, volume with unit count. Include both when you want
the model to distinguish "same coins traded at higher price" from "more
coins traded".

Both features share `_ZScoreOverColumn` so the pattern only lives once.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import timedelta

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


def _z_of(bars: Sequence[KlineRecord], column: Callable[[KlineRecord], float]) -> float | None:
    """Z-score of the last value against the full window (self included).

    Including the current sample in the mean/std keeps the statistic well-
    defined even when the history is momentarily constant: an outlier can
    still produce a large, correctly-signed z, whereas the standard
    "history excluding self" form would report 0.0 by fallback.
    """
    values = [column(b) for b in bars]
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = math.sqrt(var)
    if sd == 0.0:
        return 0.0
    return (values[-1] - mean) / sd


class VolumeZScoreN:
    def __init__(self, *, window: int = 48) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._window = window
        self.spec = FeatureSpec(
            name="volume_zscore",
            version=str(window),
            inputs=("volume",),
            lookback_bars=window + 1,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if len(history) < self.spec.lookback_bars:
            return None
        window = history[-self.spec.lookback_bars :]
        return _z_of(window, lambda b: b.volume)


class TurnoverZScoreN:
    def __init__(self, *, window: int = 48) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._window = window
        self.spec = FeatureSpec(
            name="turnover_zscore",
            version=str(window),
            inputs=("turnover",),
            lookback_bars=window + 1,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if len(history) < self.spec.lookback_bars:
            return None
        window = history[-self.spec.lookback_bars :]
        return _z_of(window, lambda b: b.turnover)
