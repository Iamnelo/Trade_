"""Oracle Benchmark (Perfect Foresight) — RESEARCH ONLY.

Computes the theoretical maximum PnL achievable over a price series given
perfect knowledge of every future bar. Used ONLY to report "capture ratio":
what fraction of the available opportunity a real strategy captured.

============================================================================
DO NOT IMPORT THIS MODULE FROM ANY STRATEGY, RISK MANAGER, FEATURE STORE
CODE PATH, OR LIVE RUNTIME. IT USES FUTURE DATA BY DESIGN.
============================================================================

The classic max-profit-with-multiple-transactions result:

- Long-only: buying at every future low and selling at every future high is
  equivalent to summing `max(0, close[i+1] - close[i])` over consecutive
  bars, then compounding by cumulative multiplicative gain. In returns
  space the max compounded return equals product((close[i+1]/close[i]) if
  close[i+1] > close[i] else 1).
- Long/short: the same but capturing the down moves via a short, so the
  compounded return uses `close[i+1]/close[i]` (or its reciprocal for
  shorts) at every bar.

Costs are ignored — the oracle is meant to be an upper bound, not a
realistic PnL.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from trade.data.schemas import KlineRecord


@dataclass(frozen=True, slots=True)
class OracleResult:
    initial_equity: float
    final_equity_long_only: float
    final_equity_long_short: float

    @property
    def total_return_long_only(self) -> float:
        return self.final_equity_long_only / self.initial_equity - 1.0

    @property
    def total_return_long_short(self) -> float:
        return self.final_equity_long_short / self.initial_equity - 1.0


def oracle_max_pnl(
    bars: Sequence[KlineRecord],
    *,
    initial_equity: float,
) -> OracleResult:
    """Return the perfect-foresight final equity for both long-only and long/short.

    Consecutive close-to-close moves are used. Zero and negative prices are
    guarded to avoid divide-by-zero.
    """
    if initial_equity <= 0.0:
        raise ValueError("initial_equity must be positive")
    if len(bars) < 2:
        return OracleResult(
            initial_equity=initial_equity,
            final_equity_long_only=initial_equity,
            final_equity_long_short=initial_equity,
        )
    sorted_bars = sorted(bars, key=lambda b: b.event_time)
    long_only_mult = 1.0
    long_short_mult = 1.0
    for a, b in pairwise(sorted_bars):
        if a.close <= 0 or b.close <= 0:
            continue
        ratio = b.close / a.close
        if ratio > 1.0:
            long_only_mult *= ratio
            long_short_mult *= ratio
        elif ratio < 1.0:
            long_short_mult *= 1.0 / ratio
        # ratio == 1.0: no change either way
    return OracleResult(
        initial_equity=initial_equity,
        final_equity_long_only=initial_equity * long_only_mult,
        final_equity_long_short=initial_equity * long_short_mult,
    )


def capture_ratio(
    *,
    strategy_final_equity: float,
    oracle_final_equity: float,
    initial_equity: float,
) -> float:
    """Fraction of the oracle's return captured by the strategy.

    Returned as a signed ratio where 1.0 means "captured everything the
    oracle did", 0.0 means "captured none", and negative means the
    strategy lost money on a series the oracle would have profited on.
    """
    if initial_equity <= 0.0:
        raise ValueError("initial_equity must be positive")
    oracle_return = oracle_final_equity / initial_equity - 1.0
    if oracle_return == 0.0:
        return 0.0
    strategy_return = strategy_final_equity / initial_equity - 1.0
    return strategy_return / oracle_return
