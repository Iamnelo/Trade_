"""Tests for round-trip trade metrics: expectancy, profit factor, hit stats."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.metrics.trade_metrics import compute_trade_metrics
from trade.mre.types import Fill, Side


def _fill(*, side: Side, qty: float, price: float, fee: float = 0.0, i: int = 0) -> Fill:
    return Fill(
        client_order_id=f"coid_{i}",
        symbol="BTCUSDT",
        side=side,
        quantity=qty,
        price=price,
        fee=fee,
        event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
    )


def test_no_fills_yields_empty_metrics() -> None:
    m = compute_trade_metrics([])
    assert m.n_trades == 0
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0


def test_single_round_trip_long_win() -> None:
    fills = [
        _fill(side=Side.BUY, qty=1.0, price=100.0, i=0),
        _fill(side=Side.SELL, qty=1.0, price=110.0, i=1),
    ]
    m = compute_trade_metrics(fills)
    assert m.n_trades == 1
    assert m.n_wins == 1
    assert m.total_pnl == pytest.approx(10.0)
    assert m.expectancy_per_trade == pytest.approx(10.0)
    assert m.win_rate == 1.0
    assert m.profit_factor == float("inf")  # no losing trades


def test_single_round_trip_long_loss() -> None:
    fills = [
        _fill(side=Side.BUY, qty=1.0, price=100.0, i=0),
        _fill(side=Side.SELL, qty=1.0, price=90.0, i=1),
    ]
    m = compute_trade_metrics(fills)
    assert m.n_trades == 1
    assert m.n_losses == 1
    assert m.total_pnl == pytest.approx(-10.0)
    assert m.avg_loss_pnl == pytest.approx(-10.0)


def test_short_round_trip_profits_when_price_falls() -> None:
    fills = [
        _fill(side=Side.SELL, qty=1.0, price=100.0, i=0),
        _fill(side=Side.BUY, qty=1.0, price=90.0, i=1),
    ]
    m = compute_trade_metrics(fills)
    assert m.n_trades == 1
    assert m.total_pnl == pytest.approx(10.0)


def test_fees_subtract_from_round_trip_pnl() -> None:
    fills = [
        _fill(side=Side.BUY, qty=1.0, price=100.0, fee=1.0, i=0),
        _fill(side=Side.SELL, qty=1.0, price=110.0, fee=1.0, i=1),
    ]
    m = compute_trade_metrics(fills)
    assert m.total_pnl == pytest.approx(8.0)  # 10 gross - 2 fees


def test_position_flip_closes_and_opens_new_trade() -> None:
    # Long 1@100, then flip short 2@110 (closes long at 110, opens short 1@110),
    # then buy 1@105 to close the short.
    fills = [
        _fill(side=Side.BUY, qty=1.0, price=100.0, i=0),
        _fill(side=Side.SELL, qty=2.0, price=110.0, i=1),
        _fill(side=Side.BUY, qty=1.0, price=105.0, i=2),
    ]
    m = compute_trade_metrics(fills)
    assert m.n_trades == 2
    # Trade 1: long 1@100 -> 1@110 = +10
    # Trade 2: short 1@110 -> covered @105 = +5
    assert m.total_pnl == pytest.approx(15.0)


def test_open_position_at_end_is_ignored() -> None:
    fills = [
        _fill(side=Side.BUY, qty=1.0, price=100.0, i=0),
    ]
    m = compute_trade_metrics(fills)
    assert m.n_trades == 0
    assert m.total_pnl == 0.0


def test_profit_factor_computed_correctly() -> None:
    fills = [
        # Winner: +$20
        _fill(side=Side.BUY, qty=1.0, price=100.0, i=0),
        _fill(side=Side.SELL, qty=1.0, price=120.0, i=1),
        # Loser: -$10
        _fill(side=Side.BUY, qty=1.0, price=120.0, i=2),
        _fill(side=Side.SELL, qty=1.0, price=110.0, i=3),
        # Winner: +$5
        _fill(side=Side.BUY, qty=1.0, price=110.0, i=4),
        _fill(side=Side.SELL, qty=1.0, price=115.0, i=5),
    ]
    m = compute_trade_metrics(fills)
    assert m.n_trades == 3
    assert m.n_wins == 2
    assert m.n_losses == 1
    assert m.total_pnl == pytest.approx(15.0)
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.expectancy_per_trade == pytest.approx(5.0)
    assert m.avg_win_pnl == pytest.approx(12.5)  # (20 + 5) / 2
    assert m.avg_loss_pnl == pytest.approx(-10.0)
    assert m.profit_factor == pytest.approx(25.0 / 10.0)  # gross_wins / gross_losses
    assert m.largest_win_pnl == pytest.approx(20.0)
    assert m.largest_loss_pnl == pytest.approx(-10.0)


def test_scaling_into_position_computes_weighted_avg_entry() -> None:
    # Scale in: buy 1@100 then buy 1@120 → avg entry 110.
    # Close at 130 -> +20 per unit x 2 units = +40.
    fills = [
        _fill(side=Side.BUY, qty=1.0, price=100.0, i=0),
        _fill(side=Side.BUY, qty=1.0, price=120.0, i=1),
        _fill(side=Side.SELL, qty=2.0, price=130.0, i=2),
    ]
    m = compute_trade_metrics(fills)
    assert m.n_trades == 1
    assert m.total_pnl == pytest.approx(40.0)


def test_partial_close_leaves_remainder_open() -> None:
    # Long 2@100, sell 1@110 → one trade closed for +10; the second unit
    # remains open at the end and does not count.
    fills = [
        _fill(side=Side.BUY, qty=2.0, price=100.0, i=0),
        _fill(side=Side.SELL, qty=1.0, price=110.0, i=1),
    ]
    m = compute_trade_metrics(fills)
    assert m.n_trades == 1
    assert m.total_pnl == pytest.approx(10.0)
