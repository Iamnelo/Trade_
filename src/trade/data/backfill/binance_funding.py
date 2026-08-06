"""Historical funding-rate backfill from Binance USDT-M perpetuals.

Cross-source complement to `bybit_funding` for DQ validation. Binance's
`/fapi/v1/fundingRate` returns ASC by time, up to 1000 rows per call — a
significantly larger page than Bybit's 200.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta

from trade.data.schemas import FundingRecord
from trade.exchanges.binance_public import BinancePublicClient
from trade.utils.clock import utcnow


async def backfill_binance_funding(
    client: BinancePublicClient,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    category: str = "linear",
    batch_size: int = 1000,
) -> AsyncIterator[list[FundingRecord]]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be UTC-aware datetimes")
    if start >= end:
        raise ValueError("start must be strictly before end")

    cursor = start
    while cursor < end:
        ticks = await client.fetch_funding_history(
            symbol,
            start=cursor,
            end=end,
            limit=batch_size,
        )
        if not ticks:
            return
        ticks.sort(key=lambda t: t.event_time)
        ticks = [t for t in ticks if cursor <= t.event_time < end]
        if not ticks:
            return

        ingest_time = utcnow()
        yield [
            FundingRecord(
                source="binance",
                category=category,
                symbol=t.symbol,
                event_time=t.event_time,
                ingest_time=ingest_time,
                funding_rate=t.funding_rate,
            )
            for t in ticks
        ]

        next_cursor = max(t.event_time for t in ticks) + timedelta(milliseconds=1)
        if next_cursor <= cursor:
            return
        cursor = next_cursor
