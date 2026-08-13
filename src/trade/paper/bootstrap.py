"""Historical bootstrap for the live paper-trading loop.

The live engine only accumulates history from newly-arrived WebSocket candles,
so on daily bars it would sit in WARMUP for ~87 (BTC) / ~121 (ETH) calendar
days before its first real decision. This module fetches the last N CLOSED
candles from Bybit's PUBLIC market-data REST endpoint (no API keys, no order
or testnet endpoints) so the engine can be seeded at startup.

It reuses the existing backfill stack (`BybitPublicClient` + `backfill_bybit_klines`)
rather than introducing a parallel data loader, and returns plain `KlineRecord`s
for `PaperTradingEngine.seed_history`. Only closed candles are returned: `end`
is pinned to the current interval boundary, so the in-progress candle is
excluded (the live WS delivers it when it closes).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from trade.data.backfill.bybit import backfill_bybit_klines
from trade.data.backfill.common import interval_to_timedelta
from trade.data.schemas import KlineRecord
from trade.exchanges.bybit_public import BybitPublicClient
from trade.utils.clock import utcnow


def _current_interval_boundary(now: datetime, interval: str) -> datetime:
    """Floor `now` to the interval boundary (epoch-anchored).

    For daily bars this is 00:00 UTC today; the in-progress candle opens at that
    instant, so using it as an exclusive `end` drops the unclosed candle.
    """
    step_s = int(interval_to_timedelta(interval).total_seconds())
    floored = (int(now.timestamp()) // step_s) * step_s
    return datetime.fromtimestamp(floored, tz=UTC)


async def fetch_seed_bars(
    *,
    symbols: Sequence[str],
    interval: str,
    n_bars: int,
    base_url: str,
    category: str = "linear",
    now: datetime | None = None,
    client: BybitPublicClient | None = None,
) -> list[KlineRecord]:
    """Fetch the last `n_bars` CLOSED candles per symbol (public REST).

    `client` may be injected for tests; when omitted a `BybitPublicClient` is
    created and closed here.
    """
    if n_bars <= 0:
        raise ValueError("n_bars must be positive")
    step = interval_to_timedelta(interval)
    end = _current_interval_boundary(now or utcnow(), interval)
    start = end - n_bars * step

    owned = client is None
    active = client or BybitPublicClient(base_url=base_url, category=category)
    out: list[KlineRecord] = []
    try:
        for symbol in symbols:
            async for batch in backfill_bybit_klines(
                active,
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
                category=category,
            ):
                out.extend(batch)
    finally:
        if owned:
            await active.aclose()
    return out
