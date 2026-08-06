"""Tests for the Binance funding-history public endpoint."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import respx

from trade.exchanges.binance_public import BinancePublicClient

_BASE = "https://fapi.example.com"


async def test_fetch_funding_history_parses_response() -> None:
    payload = [
        {"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingTime": 1700000000000},
        {"symbol": "BTCUSDT", "fundingRate": "-0.00005", "fundingTime": 1700028800000},
    ]
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.get("/fapi/v1/fundingRate").respond(json=payload)
        async with BinancePublicClient(base_url=_BASE) as client:
            ticks = await client.fetch_funding_history("BTCUSDT", limit=1000)

    assert len(ticks) == 2
    assert ticks[0].event_time == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
    assert ticks[0].funding_rate == pytest.approx(0.0001)


async def test_fetch_funding_history_passes_time_bounds() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        route = router.get("/fapi/v1/fundingRate").respond(json=[])
        async with BinancePublicClient(base_url=_BASE) as client:
            await client.fetch_funding_history(
                "BTCUSDT",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 2, tzinfo=UTC),
                limit=1000,
            )
    q = route.calls.last.request.url.query
    assert b"startTime=1704067200000" in q
    assert b"endTime=1704153600000" in q
    assert b"symbol=BTCUSDT" in q


async def test_fetch_funding_history_rejects_bad_limit() -> None:
    async with BinancePublicClient(base_url=_BASE) as client:
        with pytest.raises(ValueError, match="limit"):
            await client.fetch_funding_history("BTCUSDT", limit=0)
        with pytest.raises(ValueError, match="limit"):
            await client.fetch_funding_history("BTCUSDT", limit=1001)


async def test_fetch_funding_history_non_list_payload_raises() -> None:
    with respx.mock(base_url=_BASE, assert_all_called=True) as router:
        router.get("/fapi/v1/fundingRate").respond(json={"code": -1, "msg": "err"})
        async with BinancePublicClient(base_url=_BASE) as client:
            with pytest.raises(RuntimeError, match="unexpected payload"):
                await client.fetch_funding_history("BTCUSDT")
