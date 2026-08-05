"""Tests for the Binance USDT-M perpetual futures public REST client."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import respx

from trade.exchanges.binance_public import BinancePublicClient

_BASE = "https://fapi.example.com"


async def test_fetch_klines_parses_and_maps_interval() -> None:
    payload = [
        [
            1700000000000,
            "35000.0",
            "35100.0",
            "34900.0",
            "35050.0",
            "12.5",
            1700003599999,
            "437500.0",
            123,
            "5.0",
            "175000.0",
            "0",
        ]
    ]
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        route = router.get("/fapi/v1/klines").respond(json=payload)
        client = BinancePublicClient(base_url=_BASE)
        try:
            klines = await client.fetch_klines("BTCUSDT", "60", limit=1)
        finally:
            await client.aclose()

    request = route.calls.last.request
    assert b"interval=1h" in request.url.query
    assert b"symbol=BTCUSDT" in request.url.query
    assert len(klines) == 1
    assert klines[0].open_time == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
    assert klines[0].close == 35050.0
    assert klines[0].turnover == 437500.0


async def test_fetch_klines_passes_time_bounds() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        route = router.get("/fapi/v1/klines").respond(json=[])
        client = BinancePublicClient(base_url=_BASE)
        try:
            await client.fetch_klines(
                "BTCUSDT",
                "60",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 2, tzinfo=UTC),
                limit=100,
            )
        finally:
            await client.aclose()

    request = route.calls.last.request
    assert b"startTime=1704067200000" in request.url.query
    assert b"endTime=1704153600000" in request.url.query


async def test_invalid_interval_rejected() -> None:
    client = BinancePublicClient(base_url=_BASE)
    try:
        with pytest.raises(ValueError, match="Invalid interval"):
            await client.fetch_klines("BTCUSDT", "13")
    finally:
        await client.aclose()


async def test_limit_out_of_range_rejected() -> None:
    client = BinancePublicClient(base_url=_BASE)
    try:
        with pytest.raises(ValueError, match="limit"):
            await client.fetch_klines("BTCUSDT", "60", limit=1501)
    finally:
        await client.aclose()


async def test_non_list_payload_raises() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.get("/fapi/v1/klines").respond(json={"code": -1, "msg": "err"})
        client = BinancePublicClient(base_url=_BASE)
        try:
            with pytest.raises(RuntimeError, match="unexpected payload"):
                await client.fetch_klines("BTCUSDT", "60")
        finally:
            await client.aclose()
