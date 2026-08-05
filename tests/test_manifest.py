"""Tests for the dataset manifest: hashing, coverage, JSON round-trip."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from trade.data.manifest import DatasetManifest, sha256_hex


def test_sha256_hex_matches_hashlib() -> None:
    payload = b"hello world"
    assert sha256_hex(payload) == hashlib.sha256(payload).hexdigest()


def test_add_partition_hashes_and_sizes() -> None:
    m = DatasetManifest(dataset="bybit_kline_BTCUSDT_60")
    m.add_partition(
        key="raw/bybit_kline/bybit/linear/BTCUSDT/60/2024/01.parquet",
        data=b"x" * 42,
        rows=720,
        event_time_min=datetime(2024, 1, 1, tzinfo=UTC),
        event_time_max=datetime(2024, 1, 31, 23, tzinfo=UTC),
    )
    p = m.partitions[0]
    assert p.rows == 720
    assert p.bytes == 42
    assert p.sha256 == sha256_hex(b"x" * 42)
    assert m.total_rows == 720
    assert m.coverage_start == datetime(2024, 1, 1, tzinfo=UTC)
    assert m.coverage_end == datetime(2024, 1, 31, 23, tzinfo=UTC)


def test_coverage_none_when_empty() -> None:
    m = DatasetManifest(dataset="empty")
    assert m.coverage_start is None
    assert m.coverage_end is None
    assert m.total_rows == 0


def test_json_round_trip_is_stable() -> None:
    m = DatasetManifest(
        dataset="bybit_kline_BTCUSDT_60",
        created_at=datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
    )
    m.add_partition(
        key="raw/bybit_kline/bybit/linear/BTCUSDT/60/2024/01.parquet",
        data=b"data1",
        rows=100,
        event_time_min=datetime(2024, 1, 1, tzinfo=UTC),
        event_time_max=datetime(2024, 1, 31, tzinfo=UTC),
    )
    m.add_partition(
        key="raw/bybit_kline/bybit/linear/BTCUSDT/60/2024/02.parquet",
        data=b"data2",
        rows=200,
        event_time_min=datetime(2024, 2, 1, tzinfo=UTC),
        event_time_max=datetime(2024, 2, 29, tzinfo=UTC),
    )
    s = m.to_json()
    restored = DatasetManifest.from_json(s)

    assert restored.dataset == m.dataset
    assert restored.created_at == m.created_at
    assert len(restored.partitions) == 2
    assert restored.partitions[0].sha256 == m.partitions[0].sha256
    assert restored.total_rows == 300
    # Emitting the same manifest twice yields byte-identical JSON.
    assert restored.to_json() == s
