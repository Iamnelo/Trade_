"""Tests for data schemas and partition-key layout."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade.data.schemas import KLINE_COLUMNS, KlineRecord, partition_key


def test_partition_key_layout() -> None:
    key = partition_key(
        dataset="bybit_kline",
        source="bybit",
        category="linear",
        symbol="BTCUSDT",
        interval="60",
        year=2023,
        month=11,
    )
    assert key == "raw/bybit_kline/bybit/linear/BTCUSDT/60/2023/11.parquet"


def test_partition_key_zero_pads_month() -> None:
    key = partition_key(
        dataset="bybit_kline",
        source="bybit",
        category="linear",
        symbol="ETHUSDT",
        interval="D",
        year=2024,
        month=3,
    )
    assert "/2024/03.parquet" in key


def test_kline_columns_stable() -> None:
    # This ordering is persisted in parquet; adding/removing must be intentional.
    assert KLINE_COLUMNS == (
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


def test_kline_record_is_frozen() -> None:
    r = KlineRecord(
        source="bybit",
        category="linear",
        symbol="BTCUSDT",
        interval="60",
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        ingest_time=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
        turnover=1.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        r.close = 2.0  # type: ignore[misc]
