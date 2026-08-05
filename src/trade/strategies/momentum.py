"""12-1 momentum benchmark (crypto/hourly adaptation).

Classical 12-1 momentum ranks assets by their 12-month return excluding the
most recent month. For hourly crypto swing trading we default to the
equivalent hourly windows: `lookback_bars=288` (12 days) with a
`skip_bars=24` (1 day) blackout. Long when the momentum window return is
positive; flat otherwise.
"""

from __future__ import annotations

from trade.data.schemas import KlineRecord
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition


class Momentum12_1Strategy:  # noqa: N801 — intentional numeric convention name
    def __init__(
        self,
        *,
        symbol: str,
        interval: str = "60",
        lookback_bars: int = 288,
        skip_bars: int = 24,
        notional_fraction: float = 1.0,
    ) -> None:
        if lookback_bars < 2 or skip_bars < 0:
            raise ValueError("require lookback_bars >= 2 and skip_bars >= 0")
        if not 0.0 < notional_fraction <= 1.0:
            raise ValueError("notional_fraction must be in (0.0, 1.0]")
        self._symbol = symbol
        self._interval = interval
        self._lookback = lookback_bars
        self._skip = skip_bars
        self._notional_fraction = notional_fraction

    @property
    def name(self) -> str:
        return f"momentum_{self._lookback}_{self._skip}({self._symbol})"

    def on_bar(
        self,
        bar: KlineRecord,
        source: MarketReplaySource,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        if bar.symbol != self._symbol:
            return []
        need = self._lookback + self._skip + 1
        history = source.history(self._symbol, self._interval, lookback=need)
        if len(history) < need:
            return []
        # Return from (lookback + skip) bars ago to skip bars ago.
        old_close = history[-(self._lookback + self._skip)].close
        recent_close = history[-(self._skip + 1)].close
        if old_close <= 0:
            return []
        momentum = (recent_close - old_close) / old_close
        if momentum > 0.0:
            target_qty = portfolio.equity * self._notional_fraction / bar.close
        else:
            target_qty = 0.0
        return [TargetPosition(symbol=self._symbol, target_qty=target_qty)]
