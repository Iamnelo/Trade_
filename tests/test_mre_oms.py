"""Tests for OMS bookkeeping."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade.mre.oms import OrderManager
from trade.mre.types import Fill, Side, TargetPosition


def _fill(side: Side, qty: float, price: float, symbol: str = "BTCUSDT", fee: float = 0.0) -> Fill:
    return Fill(
        client_order_id=f"{symbol}-{side.value}",
        symbol=symbol,
        side=side,
        quantity=qty,
        price=price,
        fee=fee,
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_initial_state() -> None:
    oms = OrderManager(initial_equity=1000.0)
    assert oms.cash == 1000.0
    assert oms.positions() == {}
    assert oms.equity({}) == 1000.0
    assert oms.position_qty("BTCUSDT") == 0.0


def test_rejects_nonpositive_initial_equity() -> None:
    with pytest.raises(ValueError, match="positive"):
        OrderManager(initial_equity=0.0)


def test_buy_reduces_cash_and_grows_position() -> None:
    oms = OrderManager(initial_equity=1000.0)
    oms.apply_fill(_fill(Side.BUY, qty=1.0, price=100.0, fee=0.5))
    assert oms.position_qty("BTCUSDT") == 1.0
    assert oms.cash == pytest.approx(1000.0 - 100.0 - 0.5)
    # Equity marked at 110 = cash + 1 * 110 = 899.5 + 110 = 1009.5.
    assert oms.equity({"BTCUSDT": 110.0}) == pytest.approx(899.5 + 110.0)


def test_short_sell_increases_cash_and_shorts_position() -> None:
    oms = OrderManager(initial_equity=1000.0)
    oms.apply_fill(_fill(Side.SELL, qty=1.0, price=100.0, fee=0.5))
    assert oms.position_qty("BTCUSDT") == -1.0
    assert oms.cash == pytest.approx(1000.0 + 100.0 - 0.5)
    # Equity at mark 90 = cash + (-1) * 90 = 1099.5 - 90 = 1009.5 (short profit).
    assert oms.equity({"BTCUSDT": 90.0}) == pytest.approx(1099.5 - 90.0)


def test_buy_then_sell_full_flat_and_realizes_pnl_in_cash() -> None:
    oms = OrderManager(initial_equity=1000.0)
    oms.apply_fill(_fill(Side.BUY, qty=1.0, price=100.0))
    oms.apply_fill(_fill(Side.SELL, qty=1.0, price=110.0))
    assert oms.position_qty("BTCUSDT") == 0.0
    assert oms.cash == pytest.approx(1010.0)
    # No open position; equity = cash regardless of marks.
    assert oms.equity({"BTCUSDT": 500.0}) == pytest.approx(1010.0)


def test_compute_delta_orders_generates_buy_for_positive_target() -> None:
    oms = OrderManager(initial_equity=1000.0)
    orders = oms.compute_delta_orders(
        [TargetPosition(symbol="BTCUSDT", target_qty=0.5)],
        submit_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
    )
    assert len(orders) == 1
    assert orders[0].side is Side.BUY
    assert orders[0].quantity == 0.5


def test_compute_delta_orders_generates_sell_when_reducing() -> None:
    oms = OrderManager(initial_equity=1000.0)
    oms.apply_fill(_fill(Side.BUY, qty=1.0, price=100.0))
    orders = oms.compute_delta_orders(
        [TargetPosition(symbol="BTCUSDT", target_qty=0.3)],
        submit_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
    )
    assert len(orders) == 1
    assert orders[0].side is Side.SELL
    assert orders[0].quantity == pytest.approx(0.7)


def test_repeating_target_produces_no_new_orders() -> None:
    oms = OrderManager(initial_equity=1000.0)
    orders = oms.compute_delta_orders(
        [TargetPosition(symbol="BTCUSDT", target_qty=1.0)],
        submit_time=datetime(2024, 1, 1, tzinfo=UTC),
    )
    # Simulate a fill at the target.
    oms.apply_fill(_fill(Side.BUY, qty=1.0, price=100.0))
    # Re-issuing the same target now = no-op.
    again = oms.compute_delta_orders(
        [TargetPosition(symbol="BTCUSDT", target_qty=1.0)],
        submit_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
    )
    assert len(orders) == 1
    assert again == []


def test_record_equity_appends_to_history() -> None:
    oms = OrderManager(initial_equity=1000.0)
    oms.record_equity(datetime(2024, 1, 1, tzinfo=UTC), {})
    oms.record_equity(datetime(2024, 1, 1, 1, tzinfo=UTC), {})
    assert len(oms.equity_history()) == 2
    assert oms.equity_history()[0].equity == 1000.0
