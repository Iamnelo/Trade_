"""Volatility regime feature.

    vol_regime@short_long = log(realized_vol_short / realized_vol_long)

with realized_vol_X = sample std-dev of close-to-close log returns over X
bars. Positive values mean short-window vol is running hotter than the
long-window baseline → high-vol regime; negative values mean quiescent.
Log ratio (rather than raw ratio) keeps the feature scale-symmetric:
2x higher and 2x lower vol produce equal-magnitude, opposite-sign
readings, which is what LightGBM's split-finding needs to treat them
symmetrically.

Version string is `short_long` (e.g. `20_120` for 20-bar short window
against 120-bar long window on hourly bars ≈ 20h vs 5 days).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timedelta
from itertools import pairwise

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


def _realized_vol(bars: Sequence[KlineRecord]) -> float | None:
    rets = [math.log(b.close / a.close) for a, b in pairwise(bars) if a.close > 0 and b.close > 0]
    n = len(rets)
    if n < 2:
        return None
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var)


class VolRegime:
    def __init__(self, *, short_window: int = 20, long_window: int = 120) -> None:
        if short_window < 2 or long_window < 2:
            raise ValueError("short_window and long_window must be >= 2")
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self._short = short_window
        self._long = long_window
        self.spec = FeatureSpec(
            name="vol_regime",
            version=f"{short_window}_{long_window}",
            inputs=("close",),
            lookback_bars=long_window + 1,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if len(history) < self.spec.lookback_bars:
            return None
        window = history[-self.spec.lookback_bars :]
        vol_long = _realized_vol(window)
        vol_short = _realized_vol(window[-(self._short + 1) :])
        if vol_long is None or vol_short is None or vol_long <= 0 or vol_short <= 0:
            return None
        return math.log(vol_short / vol_long)
