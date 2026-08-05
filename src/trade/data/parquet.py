"""Parquet write/read for kline records.

Uses polars for a strict-typed round trip. The schema is fixed at module
level so drift becomes a hard error at write time rather than silent
corruption downstream.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

import polars as pl

from trade.data.schemas import KLINE_COLUMNS, KlineRecord

_KLINE_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.Utf8(),
    "category": pl.Utf8(),
    "symbol": pl.Utf8(),
    "interval": pl.Utf8(),
    "event_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "ingest_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "turnover": pl.Float64(),
}


def klines_to_dataframe(records: Iterable[KlineRecord]) -> pl.DataFrame:
    materialised = list(records)
    if not materialised:
        return pl.DataFrame(schema=_KLINE_SCHEMA)

    data: dict[str, list[object]] = {
        "source": [r.source for r in materialised],
        "category": [r.category for r in materialised],
        "symbol": [r.symbol for r in materialised],
        "interval": [r.interval for r in materialised],
        "event_time": [r.event_time for r in materialised],
        "ingest_time": [r.ingest_time for r in materialised],
        "open": [r.open for r in materialised],
        "high": [r.high for r in materialised],
        "low": [r.low for r in materialised],
        "close": [r.close for r in materialised],
        "volume": [r.volume for r in materialised],
        "turnover": [r.turnover for r in materialised],
    }
    df = pl.DataFrame(data, schema=_KLINE_SCHEMA)
    # Dedupe on the natural key so replayed batches never write duplicate rows;
    # sort last so the on-disk layout is deterministic regardless of unique's
    # internal ordering.
    return df.unique(subset=["source", "symbol", "interval", "event_time"], keep="last").sort(
        "event_time"
    )


def klines_to_parquet_bytes(records: Iterable[KlineRecord]) -> bytes:
    df = klines_to_dataframe(records)
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd")
    return buf.getvalue()


def parquet_bytes_to_dataframe(data: bytes) -> pl.DataFrame:
    df = pl.read_parquet(io.BytesIO(data))
    missing = set(KLINE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {sorted(missing)}")
    return df
