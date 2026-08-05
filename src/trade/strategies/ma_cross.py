"""Moving-average crossover benchmark.

Long-only for V1 simplicity: goes flat when fast SMA is below slow SMA. A
long/short variant can be added later without touching the runner — it's a
strategy-level choice.
"""

from __future__ import annotations

from trade.data.schemas import KlineRecord
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition


class MACrossStrategy:
    def __init__(
        self,
        *,
        symbol: str,
        interval: str = "60",
        fast_window: int = 20,
        slow_window: int = 50,
        notional_fraction: float = 1.0,
    ) -> None:
        if fast_window < 1 or slow_window <= fast_window:
            raise ValueError("require 1 <= fast_window < slow_window")
        if not 0.0 < notional_fraction <= 1.0:
            raise ValueError("notional_fraction must be in (0.0, 1.0]")
        self._symbol = symbol
        self._interval = interval
        self._fast = fast_window
        self._slow = slow_window
        self._notional_fraction = notional_fraction

    @property
    def name(self) -> str:
        return f"ma_cross({self._symbol},{self._fast},{self._slow})"

    def on_bar(
        self,
        bar: KlineRecord,
        source: MarketReplaySource,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        if bar.symbol != self._symbol:
            return []
        history = source.history(self._symbol, self._interval, lookback=self._slow)
        if len(history) < self._slow:
            return []
        fast_ma = sum(h.close for h in history[-self._fast :]) / self._fast
        slow_ma = sum(h.close for h in history[-self._slow :]) / self._slow
        if fast_ma > slow_ma:
            target_qty = portfolio.equity * self._notional_fraction / bar.close
        else:
            target_qty = 0.0
        return [TargetPosition(symbol=self._symbol, target_qty=target_qty)]
