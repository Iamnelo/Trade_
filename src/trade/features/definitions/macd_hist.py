"""MACD histogram feature: MACD line minus signal EMA.

Computed strictly on the declared lookback window; EMAs are seeded from the
window's leading SMA. That makes the value a pure function of `lookback_bars`
bars — a hard requirement for feature-store PIT correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


def _ema(values: Sequence[float], span: int) -> float | None:
    if span < 1 or len(values) < span:
        return None
    alpha = 2.0 / (span + 1)
    ema = sum(values[:span]) / span  # SMA seed
    for v in values[span:]:
        ema = alpha * v + (1.0 - alpha) * ema
    return ema


class MACDHistogram:
    def __init__(self, *, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if not (1 <= fast < slow) or signal < 1:
            raise ValueError("require 1 <= fast < slow and signal >= 1")
        self._fast = fast
        self._slow = slow
        self._signal = signal
        # Enough bars for the signal EMA to be stable after the slow EMA warms up.
        lookback = slow * 3 + signal
        self.spec = FeatureSpec(
            name="macd_hist",
            version=f"{fast}_{slow}_{signal}",
            inputs=("close",),
            lookback_bars=lookback,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if len(history) < self.spec.lookback_bars:
            return None
        closes = [h.close for h in history[-self.spec.lookback_bars :]]
        # Build the MACD-line series bar by bar so we can EMA it into the signal.
        macd_series: list[float] = []
        for i in range(self._slow, len(closes) + 1):
            sub = closes[:i]
            fast_ema = _ema(sub, self._fast)
            slow_ema = _ema(sub, self._slow)
            if fast_ema is None or slow_ema is None:
                continue
            macd_series.append(fast_ema - slow_ema)
        if len(macd_series) < self._signal:
            return None
        signal_line = _ema(macd_series, self._signal)
        if signal_line is None:
            return None
        return macd_series[-1] - signal_line
