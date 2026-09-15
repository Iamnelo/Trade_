"""Bollinger-band mean-reversion benchmark.

This is a research benchmark, not a production recommendation.  It enters
against an extreme close and exits when price returns to the moving average.
"""

from __future__ import annotations

from math import sqrt

from trade.data.schemas import KlineRecord
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition


class BollingerMeanReversionStrategy:
    def __init__(
        self,
        *,
        symbol: str,
        interval: str = "60",
        window: int = 20,
        num_std: float = 2.0,
        notional_fraction: float = 0.5,
        allow_short: bool = True,
    ) -> None:
        if window < 2 or num_std <= 0:
            raise ValueError("require window >= 2 and num_std > 0")
        if not 0.0 < notional_fraction <= 1.0:
            raise ValueError("notional_fraction must be in (0.0, 1.0]")
        self._symbol = symbol
        self._interval = interval
        self._window = window
        self._num_std = num_std
        self._notional_fraction = notional_fraction
        self._allow_short = allow_short

    @property
    def name(self) -> str:
        return f"bollinger_mean_reversion({self._symbol},{self._window},{self._num_std:g})"

    def on_bar(
        self,
        bar: KlineRecord,
        source: MarketReplaySource,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        if bar.symbol != self._symbol:
            return []
        history = source.history(self._symbol, self._interval, lookback=self._window)
        if len(history) < self._window:
            return []
        closes = [item.close for item in history]
        mean = sum(closes) / len(closes)
        variance = sum((close - mean) ** 2 for close in closes) / len(closes)
        band = self._num_std * sqrt(variance)
        upper, lower = mean + band, mean - band
        if bar.close < lower:
            direction = 1.0
        elif bar.close > upper and self._allow_short:
            direction = -1.0
        else:
            direction = 0.0
        qty = direction * portfolio.equity * self._notional_fraction / bar.close
        return [TargetPosition(symbol=self._symbol, target_qty=qty)]
