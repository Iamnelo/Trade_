"""Buy-and-hold benchmark: first non-trivial strategy against the MRE.

Enters `notional_fraction` of starting equity into `symbol` at the first bar
seen for that symbol, then holds until the backtest ends. The target does
not rebalance with subsequent price moves — that's what makes it "hold"
rather than "vol-target". Any additional benchmarks (MA cross, momentum,
risk-parity, random-with-risk-overlay) will live alongside this file in
Phase 2b.
"""

from __future__ import annotations

from trade.data.schemas import KlineRecord
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition


class BuyAndHoldStrategy:
    def __init__(self, *, symbol: str, notional_fraction: float = 1.0) -> None:
        if not 0.0 < notional_fraction <= 1.0:
            raise ValueError("notional_fraction must be in (0.0, 1.0]")
        self._symbol = symbol
        self._notional_fraction = notional_fraction
        self._entered = False

    @property
    def name(self) -> str:
        return f"buy_hold({self._symbol},{self._notional_fraction:g})"

    def on_bar(
        self,
        bar: KlineRecord,
        source: MarketReplaySource,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        if self._entered or bar.symbol != self._symbol:
            return []
        # Use bar.close (the price the strategy is looking at) to size the
        # position. Fill will happen at the NEXT bar's open, so realized
        # entry may drift slightly from this reference.
        target_notional = portfolio.equity * self._notional_fraction
        target_qty = target_notional / bar.close
        self._entered = True
        return [TargetPosition(symbol=self._symbol, target_qty=target_qty)]
