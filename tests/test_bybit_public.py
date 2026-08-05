"""Tests for the Bybit v5 public REST client.

Uses respx to mock httpx; no real network calls in the suite.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import respx

from trade.exchanges.bybit_public import BybitPublicClient

_BASE = "https://api.example.com"


async def test_fetch_klines_parses_response() -> None:
    payload = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "category": "linear",
            "symbol": "BTCUSDT",
            "list": [
                [
                    "1700000000000",
                    "35000.0",
                    "35100.0",
                    "34900.0",
                    "35050.0",
                    "12.5",
                    "437500.0",
                ],
                [
                    "1699996400000",
                    "34990.0",
                    "35010.0",
                    "34970.0",
                    "35000.0",
                    "8.0",
                    "279960.0",
                ],
            ],
        },
    }
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.get("/v5/market/kline").respond(json=payload)
        client = BybitPublicClient(base_url=_BASE)
        try:
            klines = await client.fetch_klines("BTCUSDT", "60", limit=2)
        finally:
            await client.aclose()

    assert len(klines) == 2
    assert klines[0].symbol == "BTCUSDT"
    assert klines[0].interval == "60"
    assert klines[0].open == 35000.0
    assert klines[0].high == 35100.0
    assert klines[0].close == 35050.0
    assert klines[0].open_time == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)


async def test_fetch_klines_passes_time_bounds() -> None:
    empty = {"retCode": 0, "retMsg": "OK", "result": {"list": []}}
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        route = router.get("/v5/market/kline").respond(json=empty)
        client = BybitPublicClient(base_url=_BASE)
        try:
            await client.fetch_klines(
                "BTCUSDT",
                "60",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 2, tzinfo=UTC),
                limit=10,
            )
        finally:
            await client.aclose()

    assert route.called
    request = route.calls.last.request
    assert b"start=1704067200000" in request.url.query
    assert b"end=1704153600000" in request.url.query
    assert b"limit=10" in request.url.query
    assert b"symbol=BTCUSDT" in request.url.query


async def test_invalid_interval_rejected() -> None:
    client = BybitPublicClient(base_url=_BASE)
    try:
        with pytest.raises(ValueError, match="Invalid Bybit interval"):
            await client.fetch_klines("BTCUSDT", "13")
    finally:
        await client.aclose()


async def test_limit_out_of_range_rejected() -> None:
    client = BybitPublicClient(base_url=_BASE)
    try:
        with pytest.raises(ValueError, match="limit"):
            await client.fetch_klines("BTCUSDT", "60", limit=0)
        with pytest.raises(ValueError, match="limit"):
            await client.fetch_klines("BTCUSDT", "60", limit=1001)
    finally:
        await client.aclose()


async def test_error_retcode_raises() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.get("/v5/market/kline").respond(
            json={"retCode": 10001, "retMsg": "bad param", "result": {}}
        )
        client = BybitPublicClient(base_url=_BASE)
        try:
            with pytest.raises(RuntimeError, match="Bybit API error"):
                await client.fetch_klines("BTCUSDT", "60")
        finally:
            await client.aclose()


async def test_http_error_propagates() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.get("/v5/market/kline").respond(status_code=500)
        client = BybitPublicClient(base_url=_BASE)
        try:
            with pytest.raises(Exception):  # noqa: B017
                await client.fetch_klines("BTCUSDT", "60")
        finally:
            await client.aclose()


async def test_context_manager_closes_owned_client() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.get("/v5/market/kline").respond(
            json={"retCode": 0, "retMsg": "OK", "result": {"list": []}}
        )
        async with BybitPublicClient(base_url=_BASE) as client:
            klines = await client.fetch_klines("BTCUSDT", "60")
    assert klines == []
