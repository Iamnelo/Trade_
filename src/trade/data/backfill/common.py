"""Shared paging logic for historical kline backfill.

Forward-pages from `start` to `end` in batches of `batch_size` bars, yielding
`KlineRecord` batches with `ingest_time` stamped at receive time. The caller
supplies a fetch function that returns raw `Kline` objects (source-agnostic).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta

from trade.data.schemas import KlineRecord
from trade.exchanges.base import Kline
from trade.utils.clock import utcnow

FetchFn = Callable[..., Awaitable[list[Kline]]]

_INTERVAL_MINUTES: dict[str, int] = {
    "1": 1,
    "3": 3,
    "5": 5,
    "15": 15,
    "30": 30,
    "60": 60,
    "120": 120,
    "240": 240,
    "360": 360,
    "720": 720,
    "D": 24 * 60,
    "W": 7 * 24 * 60,
    "M": 30 * 24 * 60,
}


def interval_to_timedelta(interval: str) -> timedelta:
    minutes = _INTERVAL_MINUTES.get(interval)
    if minutes is None:
        raise ValueError(f"Unknown interval: {interval!r}")
    return timedelta(minutes=minutes)


async def page_klines(
    fetch: FetchFn,
    *,
    source: str,
    category: str,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    batch_size: int,
) -> AsyncIterator[list[KlineRecord]]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be UTC-aware datetimes")
    if start >= end:
        raise ValueError("start must be strictly before end")

    step = interval_to_timedelta(interval)
    cursor = start
    while cursor < end:
        klines = await fetch(symbol, interval, start=cursor, end=end, limit=batch_size)
        if not klines:
            return
        klines.sort(key=lambda k: k.open_time)
        # Drop bars past `end` (the venue may include the current in-progress bar).
        klines = [k for k in klines if k.open_time < end]
        if not klines:
            return

        ingest_time = utcnow()
        yield [
            KlineRecord(
                source=source,
                category=category,
                symbol=symbol,
                interval=interval,
                event_time=k.open_time,
                ingest_time=ingest_time,
                open=k.open,
                high=k.high,
                low=k.low,
                close=k.close,
                volume=k.volume,
                turnover=k.turnover,
            )
            for k in klines
        ]

        max_time = klines[-1].open_time
        next_cursor = max_time + step
        # Safety: if the cursor did not advance, stop rather than loop forever.
        if next_cursor <= cursor:
            return
        cursor = next_cursor
