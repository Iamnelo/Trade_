"""Backfill entry point for Bybit."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

from trade.data.backfill.common import page_klines
from trade.data.schemas import KlineRecord
from trade.exchanges.bybit_public import BybitPublicClient


async def backfill_bybit_klines(
    client: BybitPublicClient,
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    category: str = "linear",
    batch_size: int = 1000,
) -> AsyncIterator[list[KlineRecord]]:
    async for batch in page_klines(
        client.fetch_klines,
        source="bybit",
        category=category,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        batch_size=batch_size,
    ):
        yield batch
