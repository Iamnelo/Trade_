"""Performance metrics for a backtest run.

All metrics take an `EquityPoint` series and (where relevant) a
`bars_per_year` scale factor so risk-adjusted numbers are comparable across
timeframes. `bars_per_year` for hourly bars = 24*365 = 8760.

Cost-adjusted Sharpe is Sharpe of returns net of a turnover-scaled cost
proxy — it's the primary ship/no-ship metric per the V1 spec.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from trade.metrics.returns import (
    drawdown_series,
    equity_values,
    log_returns,
    simple_returns,
)
from trade.mre.types import EquityPoint, Fill

HOURS_PER_YEAR = 8760
DAYS_PER_YEAR = 365


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def sharpe_ratio(
    equity_curve: Sequence[EquityPoint],
    *,
    bars_per_year: int,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio computed from bar-over-bar simple returns.

    risk_free_rate is the ANNUALIZED continuous rate; converted to a per-bar
    rate before subtraction.
    """
    returns = simple_returns(equity_curve)
    if not returns:
        return 0.0
    per_bar_rf = (1 + risk_free_rate) ** (1 / bars_per_year) - 1
    excess = [r - per_bar_rf for r in returns]
    sd = _stdev(excess)
    if sd == 0.0:
        return 0.0
    return _mean(excess) / sd * math.sqrt(bars_per_year)


def sortino_ratio(
    equity_curve: Sequence[EquityPoint],
    *,
    bars_per_year: int,
    target_return: float = 0.0,
) -> float:
    """Annualized Sortino ratio (downside-deviation Sharpe)."""
    returns = simple_returns(equity_curve)
    if not returns:
        return 0.0
    per_bar_target = (1 + target_return) ** (1 / bars_per_year) - 1
    excess = [r - per_bar_target for r in returns]
    downside = [min(0.0, e) for e in excess]
    dd_dev = math.sqrt(sum(d * d for d in downside) / len(downside))
    if dd_dev == 0.0:
        return 0.0
    return _mean(excess) / dd_dev * math.sqrt(bars_per_year)


def max_drawdown(equity_curve: Sequence[EquityPoint]) -> float:
    """Absolute max drawdown expressed as a fraction in [0, 1]."""
    dd = drawdown_series(equity_curve)
    if not dd:
        return 0.0
    return abs(min(dd))


def ulcer_index(equity_curve: Sequence[EquityPoint]) -> float:
    """RMS drawdown — like max_drawdown but penalizes duration too."""
    dd = drawdown_series(equity_curve)
    if not dd:
        return 0.0
    return math.sqrt(sum(d * d for d in dd) / len(dd))


def calmar_ratio(
    equity_curve: Sequence[EquityPoint],
    *,
    bars_per_year: int,
) -> float:
    """Annualized return divided by max drawdown.

    Overflow guard: when the sample is short relative to `bars_per_year`
    (e.g., a synthetic 4-bar unit test at hourly cadence), the annualization
    factor can blow past float64. In that case return a sign-preserved
    infinity — the metric is not meaningful at that scale anyway.
    """
    mdd = max_drawdown(equity_curve)
    if mdd == 0.0:
        return 0.0
    lr = log_returns(equity_curve)
    if not lr:
        return 0.0
    annualized_log = sum(lr) * (bars_per_year / len(lr))
    try:
        annualized_return = math.exp(annualized_log) - 1.0
    except OverflowError:
        return math.inf if annualized_log > 0 else -math.inf
    return annualized_return / mdd


def hit_rate(fills: Sequence[Fill]) -> float:
    """Fraction of matched round-trips that closed profitably.

    Uses FIFO to pair a closing fill against prior opening fills within the
    same symbol. Not equity-weighted; that's the *win rate*, not P&L.
    """
    if not fills:
        return 0.0
    # FIFO pairing per symbol; each open lot is (signed_qty, price).
    open_lots: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
    wins = 0
    trades = 0
    for f in fills:
        signed = f.quantity if f.side.value == "buy" else -f.quantity
        lots = open_lots[f.symbol]
        remaining = signed
        while lots and remaining != 0 and (lots[0][0] > 0) != (remaining > 0):
            # Opposite-direction: closing an open lot.
            open_qty, open_px = lots[0]
            if abs(remaining) >= abs(open_qty):
                pnl = (f.price - open_px) * open_qty
                trades += 1
                wins += 1 if pnl > 0 else 0
                remaining += open_qty  # signed, moves toward 0
                lots.popleft()
            else:
                closed = -remaining
                pnl = (f.price - open_px) * closed
                trades += 1
                wins += 1 if pnl > 0 else 0
                lots[0] = (open_qty + remaining, open_px)  # remaining is opposite sign of open_qty
                remaining = 0.0
        if remaining != 0:
            lots.append((remaining, f.price))
    return wins / trades if trades else 0.0


def turnover(
    fills: Sequence[Fill],
    *,
    initial_equity: float,
) -> float:
    """Sum of |fill notional| divided by initial equity. Higher = more trading."""
    if initial_equity <= 0 or not fills:
        return 0.0
    return sum(f.quantity * f.price for f in fills) / initial_equity


def cvar(
    equity_curve: Sequence[EquityPoint],
    *,
    alpha: float = 0.05,
) -> float:
    """Expected shortfall of bar-over-bar returns at the lower `alpha` tail.

    Returned as a negative number when the tail is loss-heavy.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    returns = simple_returns(equity_curve)
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    cutoff = max(1, math.floor(alpha * len(sorted_r)))
    tail = sorted_r[:cutoff]
    return _mean(tail)


def cost_adjusted_sharpe(
    equity_curve: Sequence[EquityPoint],
    fills: Sequence[Fill],
    *,
    bars_per_year: int,
    cost_bps_per_side: float,
    initial_equity: float,
) -> float:
    """Sharpe after subtracting a turnover-based cost proxy from each return.

    Costs are already embedded in the fills that shaped `equity_curve`
    (fee + slippage). This metric applies an ADDITIONAL conservative penalty
    equal to `cost_bps_per_side` per side per unit of turnover, spread
    uniformly across the equity path. Use it as a robust ship/no-ship gate.
    """
    returns = simple_returns(equity_curve)
    if not returns or initial_equity <= 0:
        return 0.0
    total_turnover_notional = sum(f.quantity * f.price for f in fills)
    total_cost = total_turnover_notional * cost_bps_per_side / 10_000.0
    per_bar_cost_return = -total_cost / (len(returns) * initial_equity)
    adjusted = [r + per_bar_cost_return for r in returns]
    sd = _stdev(adjusted)
    if sd == 0.0:
        return 0.0
    return _mean(adjusted) / sd * math.sqrt(bars_per_year)


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    initial_equity: float
    final_equity: float
    total_return_pct: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    ulcer_index_pct: float
    hit_rate: float
    turnover: float
    cvar_5pct: float
    cost_adjusted_sharpe: float
    n_bars: int
    n_fills: int
    strategy_name: str = ""
    halted_reasons_seen: tuple[str, ...] = field(default_factory=tuple)


def summarize(
    *,
    equity_curve: Sequence[EquityPoint],
    fills: Sequence[Fill],
    initial_equity: float,
    bars_per_year: int,
    cost_bps_per_side: float = 5.5,
    strategy_name: str = "",
    halted_reasons_seen: tuple[str, ...] = (),
) -> PerformanceReport:
    values = equity_values(equity_curve)
    final = values[-1] if values else initial_equity
    total_ret = (final / initial_equity - 1.0) if initial_equity > 0 else 0.0
    return PerformanceReport(
        initial_equity=initial_equity,
        final_equity=final,
        total_return_pct=total_ret * 100.0,
        sharpe=sharpe_ratio(equity_curve, bars_per_year=bars_per_year),
        sortino=sortino_ratio(equity_curve, bars_per_year=bars_per_year),
        calmar=calmar_ratio(equity_curve, bars_per_year=bars_per_year),
        max_drawdown_pct=max_drawdown(equity_curve) * 100.0,
        ulcer_index_pct=ulcer_index(equity_curve) * 100.0,
        hit_rate=hit_rate(fills),
        turnover=turnover(fills, initial_equity=initial_equity),
        cvar_5pct=cvar(equity_curve, alpha=0.05),
        cost_adjusted_sharpe=cost_adjusted_sharpe(
            equity_curve,
            fills,
            bars_per_year=bars_per_year,
            cost_bps_per_side=cost_bps_per_side,
            initial_equity=initial_equity,
        ),
        n_bars=len(equity_curve),
        n_fills=len(fills),
        strategy_name=strategy_name,
        halted_reasons_seen=tuple(halted_reasons_seen),
    )
