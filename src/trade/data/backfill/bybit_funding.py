"""Historical funding-rate backfill from Bybit.

Forward-pages from `start` to `end` in batches of `batch_size` funding
settlements, yielding `FundingRecord` batches. Bybit publishes funding
every 8 hours (00:00 / 08:00 / 16:00 UTC) so a full year is roughly
3 * 365 ≈ 1095 rows per symbol — trivial by kline-scale standards.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta

from trade.data.schemas import FundingRecord
from trade.exchanges.bybit_public import BybitPublicClient
from trade.utils.clock import utcnow


async def backfill_bybit_funding(
    client: BybitPublicClient,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    category: str = "linear",
    batch_size: int = 200,
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
        # Drop any settlements outside the requested window.
        ticks = [t for t in ticks if cursor <= t.event_time < end]
        if not ticks:
            return

        ingest_time = utcnow()
        yield [
            FundingRecord(
                source="bybit",
                category=category,
                symbol=t.symbol,
                event_time=t.event_time,
                ingest_time=ingest_time,
                funding_rate=t.funding_rate,
            )
            for t in ticks
        ]

        # Advance past the newest returned settlement.
        next_cursor = max(t.event_time for t in ticks) + timedelta(milliseconds=1)
        if next_cursor <= cursor:
            return
        cursor = next_cursor
