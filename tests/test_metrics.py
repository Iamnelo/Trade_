"""Tests for the metrics library."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from trade.metrics.performance import (
    HOURS_PER_YEAR,
    calmar_ratio,
    cost_adjusted_sharpe,
    cvar,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    turnover,
    ulcer_index,
)
from trade.metrics.returns import drawdown_series, log_returns, simple_returns
from trade.mre.types import EquityPoint, Fill, Side


def _curve(values: list[float]) -> list[EquityPoint]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        EquityPoint(timestamp=base + timedelta(hours=i), equity=v) for i, v in enumerate(values)
    ]


def test_simple_returns_basic() -> None:
    r = simple_returns(_curve([100.0, 110.0, 121.0]))
    assert r == pytest.approx([0.1, 0.1])


def test_log_returns_basic() -> None:
    r = log_returns(_curve([100.0, 110.0]))
    assert r[0] == pytest.approx(math.log(1.1))


def test_returns_empty_or_singleton() -> None:
    assert simple_returns(_curve([])) == []
    assert simple_returns(_curve([100.0])) == []


def test_drawdown_series_monotone_up_all_zero() -> None:
    dd = drawdown_series(_curve([100.0, 110.0, 120.0]))
    assert dd == [0.0, 0.0, 0.0]


def test_drawdown_series_recovery_pattern() -> None:
    dd = drawdown_series(_curve([100.0, 120.0, 90.0, 108.0]))
    assert dd[0] == 0.0
    assert dd[1] == 0.0
    assert dd[2] == pytest.approx(-0.25)  # 90 vs peak 120
    assert dd[3] == pytest.approx(-0.10)  # 108 vs peak 120


def test_max_drawdown_matches_manual() -> None:
    mdd = max_drawdown(_curve([100.0, 120.0, 60.0, 90.0]))
    assert mdd == pytest.approx(0.5)


def test_sharpe_positive_for_upward_curve() -> None:
    curve = _curve([100.0 * (1.001**i) for i in range(100)])
    s = sharpe_ratio(curve, bars_per_year=HOURS_PER_YEAR)
    assert s > 0
    # Constant per-bar return => infinite Sharpe in theory, huge in practice.
    assert s > 100


def test_sharpe_zero_for_flat_curve() -> None:
    assert sharpe_ratio(_curve([100.0] * 10), bars_per_year=HOURS_PER_YEAR) == 0.0


def test_sortino_ignores_upside_vol() -> None:
    # Alternating +5% / +0% / +5% / +0% ... = all non-negative -> infinite Sortino,
    # so we get 0.0 (guard against zero-division).
    curve = _curve([100.0, 105.0, 105.0, 110.25, 110.25])
    assert sortino_ratio(curve, bars_per_year=HOURS_PER_YEAR) == 0.0


def test_calmar_positive_with_max_dd() -> None:
    # Use a per-sample scaling so the 4-bar toy sample doesn't blow up
    # exp(annualized_log).
    curve = _curve([100.0, 120.0, 80.0, 130.0])
    c = calmar_ratio(curve, bars_per_year=len(curve))
    assert c != 0.0


def test_calmar_zero_when_no_drawdown() -> None:
    curve = _curve([100.0, 110.0, 120.0])
    assert calmar_ratio(curve, bars_per_year=HOURS_PER_YEAR) == 0.0


def test_ulcer_index_nonnegative_and_zero_for_flat() -> None:
    assert ulcer_index(_curve([100.0] * 5)) == 0.0
    assert ulcer_index(_curve([100.0, 80.0, 60.0])) > 0.0


def test_cvar_captures_left_tail() -> None:
    # Curve with clear tail losses.
    curve = _curve([100.0, 105.0, 100.0, 80.0, 82.0, 85.0, 90.0, 95.0, 100.0, 105.0])
    c = cvar(curve, alpha=0.10)
    assert c < 0  # tail is loss-heavy


def test_cvar_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        cvar(_curve([100.0, 110.0]), alpha=0.0)
    with pytest.raises(ValueError):
        cvar(_curve([100.0, 110.0]), alpha=1.0)


def _fill(side: Side, qty: float, price: float, symbol: str = "BTCUSDT") -> Fill:
    return Fill(
        client_order_id=f"{symbol}-{side.value}-{price}",
        symbol=symbol,
        side=side,
        quantity=qty,
        price=price,
        fee=0.0,
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_hit_rate_all_winners() -> None:
    fills = [
        _fill(Side.BUY, 1.0, 100.0),
        _fill(Side.SELL, 1.0, 110.0),
        _fill(Side.BUY, 1.0, 100.0),
        _fill(Side.SELL, 1.0, 105.0),
    ]
    assert hit_rate(fills) == pytest.approx(1.0)


def test_hit_rate_mixed() -> None:
    fills = [
        _fill(Side.BUY, 1.0, 100.0),
        _fill(Side.SELL, 1.0, 110.0),  # win
        _fill(Side.BUY, 1.0, 100.0),
        _fill(Side.SELL, 1.0, 90.0),  # loss
    ]
    assert hit_rate(fills) == pytest.approx(0.5)


def test_hit_rate_no_fills() -> None:
    assert hit_rate([]) == 0.0


def test_turnover_basic() -> None:
    fills = [_fill(Side.BUY, 1.0, 100.0), _fill(Side.SELL, 1.0, 100.0)]
    assert turnover(fills, initial_equity=1000.0) == pytest.approx(0.2)


def test_turnover_zero_initial() -> None:
    assert turnover([_fill(Side.BUY, 1.0, 100.0)], initial_equity=0.0) == 0.0


def test_cost_adjusted_sharpe_always_less_or_equal_to_sharpe_when_costs_positive() -> None:
    curve = _curve([100.0 + i * 0.1 for i in range(200)])
    fills = [_fill(Side.BUY, 1.0, 100.0), _fill(Side.SELL, 1.0, 110.0)]
    s = sharpe_ratio(curve, bars_per_year=HOURS_PER_YEAR)
    cas = cost_adjusted_sharpe(
        curve,
        fills,
        bars_per_year=HOURS_PER_YEAR,
        cost_bps_per_side=10.0,
        initial_equity=1000.0,
    )
    assert cas <= s


def test_summarize_returns_a_report() -> None:
    curve = _curve([100.0, 101.0, 102.0, 103.0])
    report = summarize(
        equity_curve=curve,
        fills=[],
        initial_equity=100.0,
        bars_per_year=HOURS_PER_YEAR,
        strategy_name="test",
    )
    assert report.strategy_name == "test"
    assert report.n_bars == 4
    assert report.n_fills == 0
    assert report.final_equity == 103.0
    assert report.total_return_pct == pytest.approx(3.0)
