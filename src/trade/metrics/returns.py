"""Equity-curve utilities: returns, log returns, drawdown series.

Everything here is a pure function of an equity time series. Statistics that
depend on the sampling frequency (Sharpe, Sortino, Calmar) live in
`performance.py` and require an explicit `bars_per_year` so an hourly and a
daily backtest are compared on the same scale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from trade.mre.types import EquityPoint


def equity_values(equity_curve: Sequence[EquityPoint]) -> list[float]:
    return [p.equity for p in equity_curve]


def simple_returns(equity_curve: Sequence[EquityPoint]) -> list[float]:
    """Bar-over-bar simple returns: r_t = e_t / e_{t-1} - 1. Length = N-1."""
    values = equity_values(equity_curve)
    if len(values) < 2:
        return []
    out: list[float] = []
    for prev, cur in pairwise(values):
        if prev == 0:
            out.append(0.0)
        else:
            out.append(cur / prev - 1.0)
    return out


def log_returns(equity_curve: Sequence[EquityPoint]) -> list[float]:
    values = equity_values(equity_curve)
    if len(values) < 2:
        return []
    out: list[float] = []
    for prev, cur in pairwise(values):
        if prev <= 0 or cur <= 0:
            out.append(0.0)
        else:
            out.append(math.log(cur / prev))
    return out


def drawdown_series(equity_curve: Sequence[EquityPoint]) -> list[float]:
    """Fractional drawdown from running peak: dd_t = e_t / max(e_{<=t}) - 1.

    All values are in [-1, 0].
    """
    values = equity_values(equity_curve)
    if not values:
        return []
    peak = values[0]
    out: list[float] = []
    for v in values:
        peak = max(peak, v)
        out.append(v / peak - 1.0 if peak > 0 else 0.0)
    return out
