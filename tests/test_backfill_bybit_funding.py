"""Tests for the Bybit funding-rate backfill loop."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx

from trade.data.backfill.bybit_funding import backfill_bybit_funding
from trade.exchanges.bybit_public import BybitPublicClient


def _funding_page(rows: list[tuple[int, str]]) -> dict[str, Any]:
    # Bybit returns newest-first.
    payload_rows = [
        {"symbol": "BTCUSDT", "fundingRate": rate, "fundingRateTimestamp": str(ts_ms)}
        for ts_ms, rate in reversed(rows)
    ]
    return {"retCode": 0, "retMsg": "OK", "result": {"list": payload_rows}}


async def test_bybit_funding_backfill_pages_forward() -> None:
    # Two settlements at 00:00 and 08:00 UTC on 2024-01-01.
    ts_a = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    ts_b = int(datetime(2024, 1, 1, 8, tzinfo=UTC).timestamp() * 1000)
    with respx.mock(base_url="https://bybit.example.com") as router:
        router.get("/v5/market/funding/history").side_effect = [
            httpx.Response(200, json=_funding_page([(ts_a, "0.0001"), (ts_b, "0.0002")])),
            httpx.Response(200, json={"retCode": 0, "retMsg": "OK", "result": {"list": []}}),
        ]
        async with BybitPublicClient(base_url="https://bybit.example.com") as client:
            records = []
            async for batch in backfill_bybit_funding(
                client,
                symbol="BTCUSDT",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 2, tzinfo=UTC),
                batch_size=200,
            ):
                records.extend(batch)

    assert [r.event_time for r in records] == [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 8, tzinfo=UTC),
    ]
    assert records[0].source == "bybit"
    assert records[0].funding_rate == pytest.approx(0.0001)


async def test_bybit_funding_backfill_drops_settlements_past_end() -> None:
    ts_a = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    ts_b = int(datetime(2024, 1, 1, 8, tzinfo=UTC).timestamp() * 1000)
    ts_c = int(datetime(2024, 1, 2, tzinfo=UTC).timestamp() * 1000)  # at end (exclusive)
    with respx.mock(base_url="https://bybit.example.com") as router:
        router.get("/v5/market/funding/history").side_effect = [
            httpx.Response(
                200,
                json=_funding_page([(ts_a, "0.0001"), (ts_b, "0.0002"), (ts_c, "0.0003")]),
            ),
            httpx.Response(200, json={"retCode": 0, "retMsg": "OK", "result": {"list": []}}),
        ]
        async with BybitPublicClient(base_url="https://bybit.example.com") as client:
            records = []
            async for batch in backfill_bybit_funding(
                client,
                symbol="BTCUSDT",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 2, tzinfo=UTC),
            ):
                records.extend(batch)

    # ts_c is at 2024-01-02 00:00 UTC which is the exclusive end — dropped.
    assert len(records) == 2


async def test_bybit_funding_backfill_rejects_bad_bounds() -> None:
    async with BybitPublicClient(base_url="https://bybit.example.com") as client:
        gen = backfill_bybit_funding(
            client,
            symbol="BTCUSDT",
            start=datetime(2024, 1, 1),  # naive
            end=datetime(2024, 1, 2, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="UTC-aware"):
            async for _ in gen:
                pass

    async with BybitPublicClient(base_url="https://bybit.example.com") as client:
        gen = backfill_bybit_funding(
            client,
            symbol="BTCUSDT",
            start=datetime(2024, 1, 2, tzinfo=UTC),
            end=datetime(2024, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="strictly before"):
            async for _ in gen:
                pass
