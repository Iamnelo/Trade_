"""Abstract exchange interfaces.

The signal engine, risk manager, and OMS must depend only on these
interfaces. That way a single code path drives backtest, paper, and live —
only the concrete adapter changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Kline:
    symbol: str
    interval: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


class MarketDataSource(ABC):
    @abstractmethod
    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[Kline]: ...
