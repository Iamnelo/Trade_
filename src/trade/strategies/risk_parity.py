"""Inverse-volatility risk parity benchmark for multi-symbol portfolios.

Weights each symbol inversely proportional to its realized volatility over
`vol_lookback_bars`. Rebalances only on the primary symbol at multiples of
`rebalance_period_bars` (default: 24 hourly bars = daily) to keep turnover
bounded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from trade.data.backfill.common import interval_to_timedelta
from trade.data.schemas import KlineRecord
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


class RiskParityStrategy:
    def __init__(
        self,
        *,
        symbols: Sequence[str],
        interval: str = "60",
        vol_lookback_bars: int = 720,  # 30 days of hourly bars
        rebalance_period_bars: int = 24,  # 1 day of hourly bars
        total_notional_fraction: float = 1.0,
    ) -> None:
        if len(symbols) < 2:
            raise ValueError("risk parity requires >= 2 symbols")
        if vol_lookback_bars < 3 or rebalance_period_bars < 1:
            raise ValueError("require vol_lookback_bars >= 3 and rebalance_period_bars >= 1")
        if not 0.0 < total_notional_fraction <= 1.0:
            raise ValueError("total_notional_fraction must be in (0.0, 1.0]")
        self._symbols = tuple(symbols)
        self._primary = self._symbols[0]
        self._interval = interval
        self._interval_seconds = int(interval_to_timedelta(interval).total_seconds())
        self._vol_lookback = vol_lookback_bars
        self._rebalance = rebalance_period_bars
        self._notional_fraction = total_notional_fraction

    @property
    def name(self) -> str:
        return f"risk_parity({','.join(self._symbols)})"

    def on_bar(
        self,
        bar: KlineRecord,
        source: MarketReplaySource,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        # Rebalance only on primary-symbol bars at wall-clock period boundaries,
        # so replay/paper/live all rebalance at the same real times.
        if bar.symbol != self._primary:
            return []
        bar_idx = int(bar.event_time.timestamp() / self._interval_seconds)
        if bar_idx % self._rebalance != 0:
            return []

        inv_vols: dict[str, float] = {}
        prices: dict[str, float] = {}
        for sym in self._symbols:
            hist = source.history(sym, self._interval, lookback=self._vol_lookback + 1)
            if len(hist) < self._vol_lookback + 1:
                return []
            returns = [b.close / a.close - 1.0 for a, b in pairwise(hist) if a.close > 0]
            vol = _stdev(returns)
            if vol <= 0.0:
                return []
            inv_vols[sym] = 1.0 / vol
            prices[sym] = hist[-1].close

        total_inv = sum(inv_vols.values())
        if total_inv <= 0.0:
            return []
        total_notional = portfolio.equity * self._notional_fraction
        targets: list[TargetPosition] = []
        for sym in self._symbols:
            weight = inv_vols[sym] / total_inv
            notional = weight * total_notional
            qty = notional / prices[sym]
            targets.append(TargetPosition(symbol=sym, target_qty=qty))
        return targets
