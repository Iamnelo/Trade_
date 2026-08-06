"""Tests for BybitExecutionVenue (signed private API)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from trade.exchanges.bybit_signed import BybitSigner
from trade.exchanges.bybit_venue import BybitExecutionVenue, _fmt_qty
from trade.execution.types import ExchangeError, VenuePosition
from trade.mre.types import Order, Side

_BASE = "https://api.example.com"


def _signer() -> BybitSigner:
    return BybitSigner(api_key="ak", api_secret="sk", recv_window_ms=5000, now_ms=lambda: 12345)


def _order(side: Side = Side.BUY, qty: float = 0.01) -> Order:
    return Order(
        client_order_id="my-coid-1",
        symbol="BTCUSDT",
        side=side,
        quantity=qty,
        submit_time=datetime(2024, 1, 1, tzinfo=UTC),
    )


async def test_submit_order_serialises_body_and_returns_ack() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        route = router.post("/v5/order/create").respond(
            json={
                "retCode": 0,
                "retMsg": "OK",
                "result": {"orderId": "venue-uuid", "orderLinkId": "my-coid-1"},
            }
        )
        async with BybitExecutionVenue(base_url=_BASE, signer=_signer()) as venue:
            ack = await venue.submit_order(_order(Side.BUY, qty=0.5))

    body = json.loads(route.calls.last.request.content)
    assert body == {
        "category": "linear",
        "orderLinkId": "my-coid-1",
        "orderType": "Market",
        "qty": "0.5",
        "side": "Buy",
        "symbol": "BTCUSDT",
    }
    headers = route.calls.last.request.headers
    assert headers["X-BAPI-API-KEY"] == "ak"
    assert "X-BAPI-SIGN" in headers
    assert ack.client_order_id == "my-coid-1"
    assert ack.venue_order_id == "venue-uuid"


async def test_submit_order_ret_code_error_raises_and_is_not_retried() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        route = router.post("/v5/order/create").respond(
            json={"retCode": 10001, "retMsg": "bad param", "result": {}}
        )
        async with BybitExecutionVenue(base_url=_BASE, signer=_signer()) as venue:
            with pytest.raises(ExchangeError):
                await venue.submit_order(_order())
    # Business errors are not retryable.
    assert route.call_count == 1


async def test_submit_order_retries_on_transient_5xx_then_succeeds() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.post("/v5/order/create").side_effect = [
            httpx.Response(503),
            httpx.Response(
                200,
                json={"retCode": 0, "result": {"orderId": "v", "orderLinkId": "my-coid-1"}},
            ),
        ]
        async with BybitExecutionVenue(base_url=_BASE, signer=_signer()) as venue:
            ack = await venue.submit_order(_order())
    assert ack.venue_order_id == "v"


async def test_cancel_order_hits_expected_endpoint() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        route = router.post("/v5/order/cancel").respond(json={"retCode": 0, "result": {}})
        async with BybitExecutionVenue(base_url=_BASE, signer=_signer()) as venue:
            await venue.cancel_order(symbol="BTCUSDT", client_order_id="my-coid-1")
    body = json.loads(route.calls.last.request.content)
    assert body == {"category": "linear", "orderLinkId": "my-coid-1", "symbol": "BTCUSDT"}


async def test_cancel_all_returns_count() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.post("/v5/order/cancel-all").respond(
            json={"retCode": 0, "result": {"list": [{"orderId": "a"}, {"orderId": "b"}]}}
        )
        async with BybitExecutionVenue(base_url=_BASE, signer=_signer()) as venue:
            n = await venue.cancel_all(symbol="BTCUSDT")
    assert n == 2


async def test_get_positions_parses_signed_size() -> None:
    payload = {
        "retCode": 0,
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "size": "0.5",
                    "avgPrice": "50000",
                    "unrealisedPnl": "12.5",
                },
                {
                    "symbol": "ETHUSDT",
                    "side": "Sell",
                    "size": "1.0",
                    "avgPrice": "3000",
                    "unrealisedPnl": "-5.0",
                },
                {
                    "symbol": "SOLUSDT",
                    "side": "None",
                    "size": "0",
                    "avgPrice": "0",
                    "unrealisedPnl": "0",
                },
            ]
        },
    }
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        route = router.get("/v5/position/list").respond(json=payload)
        async with BybitExecutionVenue(base_url=_BASE, signer=_signer()) as venue:
            positions = await venue.get_positions()

    q = route.calls.last.request.url.query
    assert b"category=linear" in q
    assert list(positions) == [
        VenuePosition(symbol="BTCUSDT", quantity=0.5, entry_price=50000.0, unrealized_pnl=12.5),
        VenuePosition(symbol="ETHUSDT", quantity=-1.0, entry_price=3000.0, unrealized_pnl=-5.0),
    ]


async def test_get_positions_symbol_filter_added_to_query() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        route = router.get("/v5/position/list").respond(json={"retCode": 0, "result": {"list": []}})
        async with BybitExecutionVenue(base_url=_BASE, signer=_signer()) as venue:
            await venue.get_positions(symbol="BTCUSDT")
    q = route.calls.last.request.url.query
    assert b"symbol=BTCUSDT" in q


def test_qty_formatting_drops_trailing_zeros() -> None:
    assert _fmt_qty(0.5) == "0.5"
    assert _fmt_qty(1.0) == "1"
    assert _fmt_qty(0.0001) == "0.0001"
    assert _fmt_qty(1234.56789) == "1234.56789"
