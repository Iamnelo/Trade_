"""Tests for FeatureSetManifest hashing and JSON round-trip."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from trade.features.manifest import (
    FeatureSetManifest,
    compute_feature_spec_sha256,
    sha256_hex,
)


def test_sha256_hex_matches_hashlib() -> None:
    assert sha256_hex(b"payload") == hashlib.sha256(b"payload").hexdigest()


def test_add_partition_hashes_and_reports_coverage() -> None:
    m = FeatureSetManifest(
        feature_id="rsi_close@14",
        entity_id="BTCUSDT",
        feature_spec_sha256="abc",
    )
    m.add_partition(
        key="features/rsi_close@14/BTCUSDT/2024/01.parquet",
        data=b"x" * 128,
        rows=100,
        event_time_min=datetime(2024, 1, 1, tzinfo=UTC),
        event_time_max=datetime(2024, 1, 31, tzinfo=UTC),
    )
    assert m.partitions[0].sha256 == sha256_hex(b"x" * 128)
    assert m.total_rows == 100
    assert m.coverage_start == datetime(2024, 1, 1, tzinfo=UTC)
    assert m.coverage_end == datetime(2024, 1, 31, tzinfo=UTC)


def test_derived_from_serializes_sorted() -> None:
    m = FeatureSetManifest(
        feature_id="rsi_close@14",
        entity_id="BTCUSDT",
        feature_spec_sha256="abc",
        derived_from=["dataset_z", "dataset_a"],
    )
    payload = m.to_json()
    # to_json sorts derived_from
    assert payload.index("dataset_a") < payload.index("dataset_z")


def test_json_round_trip() -> None:
    m = FeatureSetManifest(
        feature_id="rsi_close@14",
        entity_id="BTCUSDT",
        feature_spec_sha256="abc",
        code_git_sha="deadbeef",
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
    )
    m.add_partition(
        key="k",
        data=b"data",
        rows=5,
        event_time_min=datetime(2024, 1, 1, tzinfo=UTC),
        event_time_max=datetime(2024, 1, 2, tzinfo=UTC),
    )
    restored = FeatureSetManifest.from_json(m.to_json())
    assert restored.feature_id == m.feature_id
    assert restored.code_git_sha == "deadbeef"
    assert restored.partitions[0].sha256 == m.partitions[0].sha256


def test_feature_spec_sha256_is_deterministic_and_input_order_invariant() -> None:
    a = compute_feature_spec_sha256(
        name="rsi",
        version="14",
        inputs=("close", "open"),
        lookback_bars=15,
        availability_delay_seconds=0.0,
        entity="symbol",
        interval="60",
    )
    b = compute_feature_spec_sha256(
        name="rsi",
        version="14",
        inputs=("open", "close"),  # reversed
        lookback_bars=15,
        availability_delay_seconds=0.0,
        entity="symbol",
        interval="60",
    )
    assert a == b
