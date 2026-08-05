"""End-to-end tests for the backfill paging helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import respx

from trade.data.backfill.binance import backfill_binance_klines
from trade.data.backfill.bybit import backfill_bybit_klines
from trade.data.backfill.common import interval_to_timedelta
from trade.exchanges.binance_public import BinancePublicClient
from trade.exchanges.bybit_public import BybitPublicClient


def test_interval_to_timedelta_known() -> None:
    assert interval_to_timedelta("60") == timedelta(hours=1)
    assert interval_to_timedelta("D") == timedelta(days=1)


def test_interval_to_timedelta_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown interval"):
        interval_to_timedelta("nope")


def _bybit_page(start_ms: int, count: int, interval_ms: int = 3_600_000) -> dict[str, object]:
    # Bybit returns newest-first; the paging helper sorts internally.
    rows = [
        [
            str(start_ms + i * interval_ms),
            "100.0",
            "101.0",
            "99.0",
            "100.5",
            "10.0",
            "1005.0",
        ]
        for i in range(count)
    ]
    return {"retCode": 0, "retMsg": "OK", "result": {"list": list(reversed(rows))}}


async def test_bybit_backfill_pages_forward_until_end() -> None:
    # First page: 2 bars at 2024-01-01 00:00 and 01:00.
    # Second page: empty — signals end.
    page_1 = _bybit_page(start_ms=1704067200000, count=2)
    page_2 = {"retCode": 0, "retMsg": "OK", "result": {"list": []}}

    with respx.mock(base_url="https://bybit.example.com") as router:
        route = router.get("/v5/market/kline")
        route.side_effect = [
            _resp(page_1),
            _resp(page_2),
        ]

        client = BybitPublicClient(base_url="https://bybit.example.com")
        try:
            batches = []
            async for batch in backfill_bybit_klines(
                client,
                symbol="BTCUSDT",
                interval="60",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 1, 10, tzinfo=UTC),
                batch_size=2,
            ):
                batches.append(batch)
        finally:
            await client.aclose()

    flat = [r for batch in batches for r in batch]
    assert len(flat) == 2
    assert flat[0].event_time == datetime(2024, 1, 1, tzinfo=UTC)
    assert flat[1].event_time == datetime(2024, 1, 1, 1, tzinfo=UTC)
    assert all(r.source == "bybit" for r in flat)


async def test_bybit_backfill_drops_bars_past_end() -> None:
    # Venue returns 3 bars but end is 02:00, so only the first two count.
    page = _bybit_page(start_ms=1704067200000, count=3)  # 00:00, 01:00, 02:00
    with respx.mock(base_url="https://bybit.example.com") as router:
        router.get("/v5/market/kline").side_effect = [
            _resp(page),
            _resp({"retCode": 0, "retMsg": "OK", "result": {"list": []}}),
        ]

        client = BybitPublicClient(base_url="https://bybit.example.com")
        try:
            records = []
            async for batch in backfill_bybit_klines(
                client,
                symbol="BTCUSDT",
                interval="60",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 1, 2, tzinfo=UTC),
                batch_size=3,
            ):
                records.extend(batch)
        finally:
            await client.aclose()

    assert [r.event_time.hour for r in records] == [0, 1]


async def test_bybit_backfill_rejects_naive_bounds() -> None:
    client = BybitPublicClient(base_url="https://bybit.example.com")
    try:
        gen = backfill_bybit_klines(
            client,
            symbol="BTCUSDT",
            interval="60",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="UTC-aware"):
            async for _ in gen:
                pass
    finally:
        await client.aclose()


async def test_bybit_backfill_rejects_start_after_end() -> None:
    client = BybitPublicClient(base_url="https://bybit.example.com")
    try:
        gen = backfill_bybit_klines(
            client,
            symbol="BTCUSDT",
            interval="60",
            start=datetime(2024, 1, 2, tzinfo=UTC),
            end=datetime(2024, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="strictly before"):
            async for _ in gen:
                pass
    finally:
        await client.aclose()


async def test_binance_backfill_records_source() -> None:
    payload = [
        [
            1704067200000,  # 2024-01-01 00:00 UTC
            "100.0",
            "101.0",
            "99.0",
            "100.5",
            "10.0",
            1704070799999,
            "1005.0",
            5,
            "5.0",
            "500.0",
            "0",
        ]
    ]
    with respx.mock(base_url="https://binance.example.com") as router:
        router.get("/fapi/v1/klines").side_effect = [
            _resp(payload),
            _resp([]),
        ]

        client = BinancePublicClient(base_url="https://binance.example.com")
        try:
            records = []
            async for batch in backfill_binance_klines(
                client,
                symbol="BTCUSDT",
                interval="60",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 1, 5, tzinfo=UTC),
                batch_size=100,
            ):
                records.extend(batch)
        finally:
            await client.aclose()

    assert len(records) == 1
    assert records[0].source == "binance"
    assert records[0].event_time == datetime(2024, 1, 1, tzinfo=UTC)


# -----------------------
# helpers
# -----------------------

import httpx  # noqa: E402


def _resp(payload: object) -> httpx.Response:
    return httpx.Response(200, json=payload)
