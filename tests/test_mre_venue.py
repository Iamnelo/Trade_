"""Tests for the simulated execution venue."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade.mre.types import Order, Side
from trade.mre.venue import SimulatedVenue


def _order(side: Side, qty: float = 1.0, symbol: str = "BTCUSDT") -> Order:
    return Order(
        client_order_id=f"{symbol}-{side.value}",
        symbol=symbol,
        side=side,
        quantity=qty,
        submit_time=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_rejects_negative_fee_or_slippage() -> None:
    with pytest.raises(ValueError):
        SimulatedVenue(fee_bps=-0.1, slippage_bps=1.0)
    with pytest.raises(ValueError):
        SimulatedVenue(fee_bps=1.0, slippage_bps=-0.1)


def test_process_open_fills_at_price_plus_slippage_for_buy() -> None:
    venue = SimulatedVenue(fee_bps=10.0, slippage_bps=5.0)
    venue.submit([_order(Side.BUY, qty=2.0)])
    fills = venue.process_open(
        symbol="BTCUSDT",
        bar_open_price=100.0,
        bar_open_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
    )
    assert len(fills) == 1
    f = fills[0]
    # Slippage pushes buy price UP by 5 bps.
    assert f.price == pytest.approx(100.0 * (1 + 5 / 10_000))
    # Fee = qty * fill_price * 10 bps.
    assert f.fee == pytest.approx(2.0 * f.price * 10 / 10_000)


def test_process_open_fills_at_price_minus_slippage_for_sell() -> None:
    venue = SimulatedVenue(fee_bps=10.0, slippage_bps=5.0)
    venue.submit([_order(Side.SELL, qty=1.0)])
    fills = venue.process_open(
        symbol="BTCUSDT",
        bar_open_price=100.0,
        bar_open_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
    )
    assert fills[0].price == pytest.approx(100.0 * (1 - 5 / 10_000))


def test_only_symbol_matching_orders_fill() -> None:
    venue = SimulatedVenue(fee_bps=10.0, slippage_bps=5.0)
    venue.submit([_order(Side.BUY, symbol="BTCUSDT"), _order(Side.BUY, symbol="ETHUSDT")])
    fills = venue.process_open(
        symbol="BTCUSDT",
        bar_open_price=100.0,
        bar_open_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
    )
    assert len(fills) == 1
    assert fills[0].symbol == "BTCUSDT"
    remaining = venue.pending
    assert len(remaining) == 1
    assert remaining[0].symbol == "ETHUSDT"


def test_zero_or_negative_open_price_rejected() -> None:
    venue = SimulatedVenue(fee_bps=10.0, slippage_bps=5.0)
    with pytest.raises(ValueError, match="positive"):
        venue.process_open(
            symbol="BTCUSDT", bar_open_price=0.0, bar_open_time=datetime(2024, 1, 1, tzinfo=UTC)
        )


def test_cancel_all_clears_pending() -> None:
    venue = SimulatedVenue(fee_bps=10.0, slippage_bps=5.0)
    venue.submit([_order(Side.BUY), _order(Side.SELL)])
    assert venue.cancel_all() == 2
    assert venue.pending == ()
