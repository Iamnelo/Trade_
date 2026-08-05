"""Bybit v5 public market-data client.

Unauthenticated endpoints only; no API keys required. Used for historical
backfill and ad-hoc queries. Streaming market data (WebSocket) belongs in a
separate module and lands in Phase 1.
"""

from __future__ import annotations

from datetime import datetime
from types import TracebackType
from typing import Any, Self

import httpx

from trade.exchanges.base import Kline, MarketDataSource
from trade.utils.clock import from_epoch_ms, to_epoch_ms

_VALID_INTERVALS = frozenset(
    {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "M", "W"}
)
_MIN_LIMIT = 1
_MAX_LIMIT = 1000


class BybitPublicClient(MarketDataSource):
    def __init__(
        self,
        base_url: str,
        *,
        category: str = "linear",
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._category = category
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
        if interval not in _VALID_INTERVALS:
            raise ValueError(f"Invalid Bybit interval: {interval!r}")
        if not _MIN_LIMIT <= limit <= _MAX_LIMIT:
            raise ValueError(f"limit must be within [{_MIN_LIMIT}, {_MAX_LIMIT}]")

        params: dict[str, Any] = {
            "category": self._category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start is not None:
            params["start"] = to_epoch_ms(start)
        if end is not None:
            params["end"] = to_epoch_ms(end)

        response = await self._client.get(f"{self._base_url}/v5/market/kline", params=params)
        response.raise_for_status()
        payload = response.json()

        ret_code = payload.get("retCode")
        if ret_code != 0:
            raise RuntimeError(
                f"Bybit API error retCode={ret_code} retMsg={payload.get('retMsg')!r}"
            )

        rows: list[list[str]] = payload.get("result", {}).get("list", [])
        return [_parse_kline(row, symbol=symbol, interval=interval) for row in rows]


def _parse_kline(row: list[str], *, symbol: str, interval: str) -> Kline:
    return Kline(
        symbol=symbol,
        interval=interval,
        open_time=from_epoch_ms(int(row[0])),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        turnover=float(row[6]),
    )
