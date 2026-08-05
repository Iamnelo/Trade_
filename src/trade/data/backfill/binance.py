"""Backfill entry point for Binance."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

from trade.data.backfill.common import page_klines
from trade.data.schemas import KlineRecord
from trade.exchanges.binance_public import BinancePublicClient


async def backfill_binance_klines(
    client: BinancePublicClient,
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    category: str = "linear",
    batch_size: int = 1500,
) -> AsyncIterator[list[KlineRecord]]:
    async for batch in page_klines(
        client.fetch_klines,
        source="binance",
        category=category,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        batch_size=batch_size,
    ):
        yield batch
