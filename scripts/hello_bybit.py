"""Manual smoke script: pull the last 5 hourly bars for BTCUSDT and ETHUSDT.

Read-only, unauthenticated. Safe against the production Bybit endpoint. Use
this as a first check that networking, config, and the client all work end to
end. Not part of the automated test suite.

Run: ``python scripts/hello_bybit.py``
"""

from __future__ import annotations

import asyncio

from trade.config import get_settings
from trade.exchanges.bybit_public import BybitPublicClient
from trade.logging_setup import configure_logging, get_logger


async def _run() -> int:
    configure_logging()
    log = get_logger("hello_bybit")
    settings = get_settings()

    async with BybitPublicClient(
        base_url=settings.bybit_base_url,
        category=settings.default_category,
    ) as client:
        for symbol in settings.default_symbols:
            klines = await client.fetch_klines(symbol, "60", limit=5)
            log.info("fetched", symbol=symbol, count=len(klines))
            for k in klines:
                log.info(
                    "kline",
                    symbol=k.symbol,
                    open_time=k.open_time.isoformat(),
                    open=k.open,
                    high=k.high,
                    low=k.low,
                    close=k.close,
                    volume=k.volume,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
