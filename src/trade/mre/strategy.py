"""Strategy protocol used by the backtest runner.

The strategy receives, on each bar close: the closed bar, a read-only handle
on the market source (for PIT-safe history lookups), and a portfolio
snapshot (cash, positions, marks). It returns declarative `TargetPosition`
objects — the OMS diffs against current holdings, so the same target on
consecutive bars produces no duplicate orders.

`name` is used to label the run in the BacktestResult and MLflow logs.
"""

from __future__ import annotations

from typing import Protocol

from trade.data.schemas import KlineRecord
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition


class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    def on_bar(
        self,
        bar: KlineRecord,
        source: MarketReplaySource,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]: ...
