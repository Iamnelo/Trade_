"""Random-signal-with-risk-overlay control strategy.

Every `rebalance_period_bars`, flips a DETERMINISTIC (seeded) coin per bar
based on `sha256(seed || bar_event_time)`. Heads => long full notional;
tails => flat. Same bar times produce the same targets across runs.

Purpose: if the AI strategy does not beat this control on cost-adjusted
Sharpe over the full evaluation window, the risk overlay alone is doing the
work and the model has no alpha worth trading.
"""

from __future__ import annotations

import hashlib

from trade.data.backfill.common import interval_to_timedelta
from trade.data.schemas import KlineRecord
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition


class RandomSignalStrategy:
    def __init__(
        self,
        *,
        symbol: str,
        interval: str = "60",
        seed: int = 42,
        rebalance_period_bars: int = 24,
        notional_fraction: float = 1.0,
    ) -> None:
        if rebalance_period_bars < 1:
            raise ValueError("rebalance_period_bars must be >= 1")
        if not 0.0 < notional_fraction <= 1.0:
            raise ValueError("notional_fraction must be in (0.0, 1.0]")
        self._symbol = symbol
        self._interval = interval
        self._interval_seconds = int(interval_to_timedelta(interval).total_seconds())
        self._seed = seed
        self._rebalance = rebalance_period_bars
        self._notional_fraction = notional_fraction

    @property
    def name(self) -> str:
        return f"random_signal(seed={self._seed})"

    def _coin_is_heads(self, event_time_iso: str) -> bool:
        digest = hashlib.sha256(f"{self._seed}|{event_time_iso}".encode()).digest()
        return digest[0] >= 128  # 50/50

    def on_bar(
        self,
        bar: KlineRecord,
        source: MarketReplaySource,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        if bar.symbol != self._symbol:
            return []
        bar_idx = int(bar.event_time.timestamp() / self._interval_seconds)
        if bar_idx % self._rebalance != 0:
            return []
        heads = self._coin_is_heads(bar.event_time.isoformat())
        target_qty = portfolio.equity * self._notional_fraction / bar.close if heads else 0.0
        return [TargetPosition(symbol=self._symbol, target_qty=target_qty)]
