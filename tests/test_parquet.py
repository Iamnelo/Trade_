"""Round-trip tests for the kline parquet writer."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trade.data.parquet import (
    klines_to_dataframe,
    klines_to_parquet_bytes,
    parquet_bytes_to_dataframe,
)
from trade.data.schemas import KLINE_COLUMNS, KlineRecord


def _make_record(offset_minutes: int, close: float = 100.0) -> KlineRecord:
    return KlineRecord(
        source="bybit",
        category="linear",
        symbol="BTCUSDT",
        interval="60",
        event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=offset_minutes),
        ingest_time=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=10.0,
        turnover=close * 10.0,
    )


def test_empty_input_yields_empty_frame_with_schema() -> None:
    df = klines_to_dataframe([])
    assert df.height == 0
    assert tuple(df.columns) == KLINE_COLUMNS


def test_round_trip_preserves_values() -> None:
    records = [_make_record(0), _make_record(60), _make_record(120)]
    data = klines_to_parquet_bytes(records)
    df = parquet_bytes_to_dataframe(data)

    assert df.height == 3
    assert set(df.columns) >= set(KLINE_COLUMNS)
    closes = df["close"].to_list()
    assert closes == [100.0, 100.0, 100.0]

    # UTC-tagged datetimes survive the round trip.
    dtype = df.schema["event_time"]
    assert isinstance(dtype, pl.Datetime)
    assert dtype.time_zone == "UTC"


def test_duplicate_rows_deduped_by_natural_key() -> None:
    a = _make_record(0, close=100.0)
    b = _make_record(0, close=101.0)  # same key, newer value
    df = klines_to_dataframe([a, b])
    assert df.height == 1
    assert df["close"].to_list() == [101.0]


def test_sort_order_ascending() -> None:
    records = [_make_record(120), _make_record(0), _make_record(60)]
    df = klines_to_dataframe(records)
    event_times = df["event_time"].to_list()
    assert event_times == sorted(event_times)


def test_missing_columns_rejected_on_read() -> None:
    df = pl.DataFrame({"source": ["bybit"], "close": [1.0]})
    buf = io.BytesIO()
    df.write_parquet(buf)
    with pytest.raises(ValueError, match="missing required columns"):
        parquet_bytes_to_dataframe(buf.getvalue())
