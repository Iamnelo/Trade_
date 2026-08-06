"""Data schemas that flow through backfill, storage, and DQ.

Every record carries `event_time` (source-reported wall clock, UTC-aware) and
`ingest_time` (moment this platform received it, UTC-aware). Never rely on
`event_time` alone for freshness or gap checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# Canonical parquet column order for kline records. Kept alongside the
# parquet writer's schema so drift becomes a hard error at write time.
KLINE_COLUMNS: tuple[str, ...] = (
    "source",
    "category",
    "symbol",
    "interval",
    "event_time",
    "ingest_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
)


@dataclass(frozen=True, slots=True)
class KlineRecord:
    source: str
    category: str
    symbol: str
    interval: str
    event_time: datetime
    ingest_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


# Canonical column order for funding parquet. Funding rows are event-driven
# (not bar-aligned): one row per funding settlement.
FUNDING_COLUMNS: tuple[str, ...] = (
    "source",
    "category",
    "symbol",
    "event_time",
    "ingest_time",
    "funding_rate",
)


@dataclass(frozen=True, slots=True)
class FundingRecord:
    source: str
    category: str
    symbol: str
    event_time: datetime  # settlement time (typically 00:00 / 08:00 / 16:00 UTC)
    ingest_time: datetime
    funding_rate: float


def partition_key(
    *,
    dataset: str,
    source: str,
    category: str,
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> str:
    """Canonical object-storage key for a monthly kline partition."""
    return f"raw/{dataset}/{source}/{category}/{symbol}/{interval}/{year:04d}/{month:02d}.parquet"
