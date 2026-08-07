"""Higher-order moments of rolling log returns: skewness and (excess) kurtosis.

Both live in the same module because they consume the identical rolling
log-return series over the window. Given N bars, we compute N-1 log
returns, then the third and fourth standardised moments — Fisher-Pearson
skewness and excess kurtosis (kurt - 3).

Variance already lives in `realized_vol@N`; these two complete the shape
description of the return distribution. Real markets show fat tails and
persistent asymmetries that a Gaussian assumption throws away, so a
model with variance-only features is systematically blind to skew and
tail-thickness shifts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timedelta
from itertools import pairwise

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


def _log_returns(bars: Sequence[KlineRecord]) -> list[float]:
    return [math.log(b.close / a.close) for a, b in pairwise(bars) if a.close > 0 and b.close > 0]


class ReturnSkewN:
    def __init__(self, *, window: int = 20) -> None:
        if window < 3:
            raise ValueError("window must be >= 3 for a defined skew")
        self._window = window
        self.spec = FeatureSpec(
            name="return_skew",
            version=str(window),
            inputs=("close",),
            lookback_bars=window + 1,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if len(history) < self.spec.lookback_bars:
            return None
        rets = _log_returns(history[-self.spec.lookback_bars :])
        n = len(rets)
        if n < 3:
            return None
        mean = sum(rets) / n
        m2 = sum((r - mean) ** 2 for r in rets) / n
        m3 = sum((r - mean) ** 3 for r in rets) / n
        if m2 == 0.0:
            return 0.0
        return float(m3 / (m2**1.5))


class ReturnKurtosisN:
    def __init__(self, *, window: int = 20) -> None:
        if window < 4:
            raise ValueError("window must be >= 4 for a defined excess kurtosis")
        self._window = window
        self.spec = FeatureSpec(
            name="return_kurtosis",
            version=str(window),
            inputs=("close",),
            lookback_bars=window + 1,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if len(history) < self.spec.lookback_bars:
            return None
        rets = _log_returns(history[-self.spec.lookback_bars :])
        n = len(rets)
        if n < 4:
            return None
        mean = sum(rets) / n
        m2 = sum((r - mean) ** 2 for r in rets) / n
        m4 = sum((r - mean) ** 4 for r in rets) / n
        if m2 == 0.0:
            return 0.0
        return m4 / (m2**2) - 3.0
