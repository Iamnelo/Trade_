"""Donchian-channel breakout benchmark for trend-following comparisons."""

from __future__ import annotations

from trade.data.schemas import KlineRecord
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition


class DonchianBreakoutStrategy:
    def __init__(
        self,
        *,
        symbol: str,
        interval: str = "60",
        window: int = 20,
        notional_fraction: float = 0.5,
        allow_short: bool = True,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        if not 0.0 < notional_fraction <= 1.0:
            raise ValueError("notional_fraction must be in (0.0, 1.0]")
        self._symbol = symbol
        self._interval = interval
        self._window = window
        self._notional_fraction = notional_fraction
        self._allow_short = allow_short

    @property
    def name(self) -> str:
        return f"donchian_breakout({self._symbol},{self._window})"

    def on_bar(
        self,
        bar: KlineRecord,
        source: MarketReplaySource,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        if bar.symbol != self._symbol:
            return []
        history = source.history(self._symbol, self._interval, lookback=self._window + 1)
        if len(history) < self._window + 1:
            return []
        # A breakout must be measured against the *previous* channel.  The
        # current candle cannot define the boundary it is attempting to break.
        reference = history[:-1]
        high = max(item.high for item in reference)
        low = min(item.low for item in reference)
        if bar.close > high:
            direction = 1.0
        elif bar.close < low and self._allow_short:
            direction = -1.0
        else:
            direction = 0.0
        qty = direction * portfolio.equity * self._notional_fraction / bar.close
        return [TargetPosition(symbol=self._symbol, target_qty=qty)]
