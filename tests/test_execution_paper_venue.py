"""Tests for PaperExecutionVenue."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade.execution.paper_venue import PaperExecutionVenue
from trade.mre.types import Order, Side


def _order(side: Side, qty: float = 1.0, symbol: str = "BTCUSDT", coid: str = "coid-1") -> Order:
    return Order(
        client_order_id=coid,
        symbol=symbol,
        side=side,
        quantity=qty,
        submit_time=datetime(2024, 1, 1, tzinfo=UTC),
    )


async def test_buy_creates_long_position_at_mark_plus_slippage() -> None:
    marks = {"BTCUSDT": 100.0}
    venue = PaperExecutionVenue(mark_fn=marks.get, fee_bps=10.0, slippage_bps=5.0)
    ack = await venue.submit_order(_order(Side.BUY, qty=0.5))
    assert ack.client_order_id == "coid-1"
    assert ack.venue_order_id.startswith("paper-")

    positions = await venue.get_positions()
    assert len(positions) == 1
    assert positions[0].quantity == pytest.approx(0.5)
    # Slippage pushes buy up 5 bps.
    assert positions[0].entry_price == pytest.approx(100.0 * (1 + 5 / 10_000))


async def test_sell_creates_short_position() -> None:
    marks = {"BTCUSDT": 100.0}
    venue = PaperExecutionVenue(mark_fn=marks.get, fee_bps=0.0, slippage_bps=5.0)
    await venue.submit_order(_order(Side.SELL, qty=0.3))
    positions = await venue.get_positions()
    assert positions[0].quantity == pytest.approx(-0.3)
    # Slippage pushes sell down 5 bps.
    assert positions[0].entry_price == pytest.approx(100.0 * (1 - 5 / 10_000))


async def test_duplicate_client_order_id_is_idempotent() -> None:
    marks = {"BTCUSDT": 100.0}
    venue = PaperExecutionVenue(mark_fn=marks.get, fee_bps=0.0, slippage_bps=0.0)
    await venue.submit_order(_order(Side.BUY, qty=1.0, coid="same"))
    await venue.submit_order(_order(Side.BUY, qty=1.0, coid="same"))
    positions = await venue.get_positions()
    # Duplicate id => single fill effect.
    assert positions[0].quantity == pytest.approx(1.0)


async def test_get_positions_symbol_filter() -> None:
    marks = {"BTCUSDT": 100.0, "ETHUSDT": 200.0}
    venue = PaperExecutionVenue(mark_fn=marks.get, fee_bps=0.0, slippage_bps=0.0)
    await venue.submit_order(_order(Side.BUY, qty=1.0, symbol="BTCUSDT", coid="a"))
    await venue.submit_order(_order(Side.BUY, qty=1.0, symbol="ETHUSDT", coid="b"))
    btc_only = await venue.get_positions(symbol="BTCUSDT")
    assert [p.symbol for p in btc_only] == ["BTCUSDT"]


async def test_positions_track_unrealized_pnl_from_mark() -> None:
    marks = {"BTCUSDT": 100.0}
    venue = PaperExecutionVenue(mark_fn=marks.get, fee_bps=0.0, slippage_bps=0.0)
    await venue.submit_order(_order(Side.BUY, qty=1.0))
    marks["BTCUSDT"] = 110.0  # mark moves up
    positions = await venue.get_positions()
    # entry was 100 (no slippage), qty=1, mark now 110 => +10 unrealized.
    assert positions[0].unrealized_pnl == pytest.approx(10.0)


async def test_cancel_calls_are_no_ops() -> None:
    venue = PaperExecutionVenue(mark_fn=lambda _s: 100.0)
    await venue.cancel_order(symbol="BTCUSDT", client_order_id="x")
    assert await venue.cancel_all() == 0
    assert await venue.cancel_all(symbol="BTCUSDT") == 0


def test_rejects_negative_fee_or_slippage() -> None:
    with pytest.raises(ValueError):
        PaperExecutionVenue(mark_fn=lambda _s: 100.0, fee_bps=-1.0)
    with pytest.raises(ValueError):
        PaperExecutionVenue(mark_fn=lambda _s: 100.0, slippage_bps=-1.0)


async def test_rejects_nonpositive_mark() -> None:
    venue = PaperExecutionVenue(mark_fn=lambda _s: 0.0, fee_bps=0.0, slippage_bps=0.0)
    with pytest.raises(ValueError, match="non-positive mark"):
        await venue.submit_order(_order(Side.BUY, qty=1.0))
