"""Tests for the Bybit funding-history public endpoint."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import respx

from trade.exchanges.bybit_public import BybitPublicClient

_BASE = "https://api.example.com"


async def test_fetch_funding_history_parses_response() -> None:
    payload = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "category": "linear",
            "symbol": "BTCUSDT",
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "fundingRate": "0.0001",
                    "fundingRateTimestamp": "1700000000000",
                },
                {
                    "symbol": "BTCUSDT",
                    "fundingRate": "-0.00005",
                    "fundingRateTimestamp": "1699971200000",
                },
            ],
        },
    }
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.get("/v5/market/funding/history").respond(json=payload)
        async with BybitPublicClient(base_url=_BASE) as client:
            ticks = await client.fetch_funding_history("BTCUSDT", limit=200)

    assert len(ticks) == 2
    assert ticks[0].symbol == "BTCUSDT"
    assert ticks[0].funding_rate == pytest.approx(0.0001)
    assert ticks[0].event_time == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
    assert ticks[1].funding_rate == pytest.approx(-0.00005)


async def test_fetch_funding_history_passes_time_bounds() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        route = router.get("/v5/market/funding/history").respond(
            json={"retCode": 0, "retMsg": "OK", "result": {"list": []}}
        )
        async with BybitPublicClient(base_url=_BASE) as client:
            await client.fetch_funding_history(
                "BTCUSDT",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 2, tzinfo=UTC),
                limit=200,
            )
    assert route.called
    q = route.calls.last.request.url.query
    assert b"startTime=1704067200000" in q
    assert b"endTime=1704153600000" in q
    assert b"symbol=BTCUSDT" in q


async def test_fetch_funding_history_rejects_bad_limit() -> None:
    async with BybitPublicClient(base_url=_BASE) as client:
        with pytest.raises(ValueError, match="limit"):
            await client.fetch_funding_history("BTCUSDT", limit=0)
        with pytest.raises(ValueError, match="limit"):
            await client.fetch_funding_history("BTCUSDT", limit=201)


async def test_fetch_funding_history_ret_code_error() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.get("/v5/market/funding/history").respond(
            json={"retCode": 10001, "retMsg": "err", "result": {}}
        )
        async with BybitPublicClient(base_url=_BASE) as client:
            with pytest.raises(RuntimeError, match="funding error"):
                await client.fetch_funding_history("BTCUSDT")
