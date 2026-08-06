"""Tests for the Binance funding-rate backfill loop."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from trade.data.backfill.binance_funding import backfill_binance_funding
from trade.exchanges.binance_public import BinancePublicClient


def _funding_page(rows: list[tuple[int, str]]) -> list[dict[str, object]]:
    return [
        {"symbol": "BTCUSDT", "fundingRate": rate, "fundingTime": ts_ms} for ts_ms, rate in rows
    ]


async def test_binance_funding_backfill_pages_forward() -> None:
    ts_a = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    ts_b = int(datetime(2024, 1, 1, 8, tzinfo=UTC).timestamp() * 1000)
    with respx.mock(base_url="https://binance.example.com") as router:
        router.get("/fapi/v1/fundingRate").side_effect = [
            httpx.Response(200, json=_funding_page([(ts_a, "0.0001"), (ts_b, "0.0002")])),
            httpx.Response(200, json=[]),
        ]
        async with BinancePublicClient(base_url="https://binance.example.com") as client:
            records = []
            async for batch in backfill_binance_funding(
                client,
                symbol="BTCUSDT",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 2, tzinfo=UTC),
            ):
                records.extend(batch)

    assert [r.event_time for r in records] == [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 8, tzinfo=UTC),
    ]
    assert records[0].source == "binance"
    assert records[0].funding_rate == pytest.approx(0.0001)


async def test_binance_funding_backfill_rejects_bad_bounds() -> None:
    async with BinancePublicClient(base_url="https://binance.example.com") as client:
        gen = backfill_binance_funding(
            client,
            symbol="BTCUSDT",
            start=datetime(2024, 1, 1),  # naive
            end=datetime(2024, 1, 2, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="UTC-aware"):
            async for _ in gen:
                pass
