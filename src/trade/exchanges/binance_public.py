"""Binance USDT-M perpetual futures public REST client.

Used as a cross-validation source for the Bybit historical backfill. NOT used
for execution — Bybit is the sole execution venue in V1.
"""

from __future__ import annotations

from datetime import datetime
from types import TracebackType
from typing import Any, Self

import httpx

from trade.exchanges.base import FundingTick, Kline, MarketDataSource
from trade.utils.clock import from_epoch_ms, to_epoch_ms

# The CLI accepts Bybit-style intervals ("60") and this table maps them across.
_BYBIT_TO_BINANCE_INTERVAL: dict[str, str] = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "360": "6h",
    "720": "12h",
    "D": "1d",
    "W": "1w",
    "M": "1M",
}
_MIN_LIMIT = 1
_MAX_LIMIT = 1500  # Binance futures kline max
_MAX_FUNDING_LIMIT = 1000  # Binance futures funding-rate max


class BinancePublicClient(MarketDataSource):
    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[Kline]:
        binance_interval = _BYBIT_TO_BINANCE_INTERVAL.get(interval)
        if binance_interval is None:
            raise ValueError(f"Invalid interval for Binance client: {interval!r}")
        if not _MIN_LIMIT <= limit <= _MAX_LIMIT:
            raise ValueError(f"limit must be within [{_MIN_LIMIT}, {_MAX_LIMIT}]")

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": binance_interval,
            "limit": limit,
        }
        if start is not None:
            params["startTime"] = to_epoch_ms(start)
        if end is not None:
            params["endTime"] = to_epoch_ms(end)

        response = await self._client.get(f"{self._base_url}/fapi/v1/klines", params=params)
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise RuntimeError(f"Binance returned unexpected payload: {rows!r}")

        # Binance kline row schema:
        # [openTime, open, high, low, close, volume, closeTime, quoteAssetVolume,
        #  numberOfTrades, takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume, ignore]
        return [
            Kline(
                symbol=symbol,
                interval=interval,
                open_time=from_epoch_ms(int(row[0])),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                turnover=float(row[7]),
            )
            for row in rows
        ]

    async def fetch_funding_history(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[FundingTick]:
        """Binance USDT-M perp /fapi/v1/fundingRate — max 1000 rows per call, ASC by time."""
        if not 1 <= limit <= _MAX_FUNDING_LIMIT:
            raise ValueError(f"limit must be within [1, {_MAX_FUNDING_LIMIT}]")

        params: dict[str, Any] = {"symbol": symbol, "limit": limit}
        if start is not None:
            params["startTime"] = to_epoch_ms(start)
        if end is not None:
            params["endTime"] = to_epoch_ms(end)

        response = await self._client.get(f"{self._base_url}/fapi/v1/fundingRate", params=params)
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise RuntimeError(f"Binance funding returned unexpected payload: {rows!r}")

        return [
            FundingTick(
                symbol=str(row.get("symbol", symbol)),
                event_time=from_epoch_ms(int(row["fundingTime"])),
                funding_rate=float(row["fundingRate"]),
            )
            for row in rows
        ]
