"""Tests for the DuckDB-backed FeatureStore — same contract, different engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.duckdb_store import DuckDBFeatureStore
from trade.features.types import FeatureSpec, LabelRow


class _MockFeature:
    def __init__(self, *, lookback: int = 3, availability_delay: timedelta = timedelta(0)) -> None:
        self.spec = FeatureSpec(
            name="mock",
            version="1",
            inputs=("close",),
            lookback_bars=lookback,
            availability_delay=availability_delay,
        )

    def compute(self, history):
        if len(history) < self.spec.lookback_bars:
            return None
        return sum(h.close for h in history[-self.spec.lookback_bars :]) / self.spec.lookback_bars


def _bar(hour: int, close: float = 100.0) -> KlineRecord:
    return KlineRecord(
        source="bybit",
        category="linear",
        symbol="BTCUSDT",
        interval="60",
        event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour),
        ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour, seconds=1),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
        turnover=close,
    )


def test_duckdb_pit_join_matches_in_memory_semantics() -> None:
    store = DuckDBFeatureStore()
    feat = _MockFeature(lookback=3, availability_delay=timedelta(0))
    bars = [_bar(i, close=100.0 + i) for i in range(10)]
    store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)

    labels = [
        LabelRow(entity_id="BTCUSDT", event_time=datetime(2024, 1, 1, 5, tzinfo=UTC), label=1.0),
        LabelRow(entity_id="BTCUSDT", event_time=datetime(2024, 1, 1, 9, tzinfo=UTC), label=-1.0),
    ]
    frame = store.point_in_time_join(labels=labels, feature_ids=["mock@1"])
    # Bar 5 latest -> mean(103,104,105) = 104. Bar 9 latest -> mean(107,108,109) = 108.
    assert frame.features["mock@1"] == (pytest.approx(104.0), pytest.approx(108.0))


def test_duckdb_pit_join_none_when_no_feature_before_label() -> None:
    store = DuckDBFeatureStore()
    feat = _MockFeature(lookback=3)
    bars = [_bar(i, close=100.0 + i) for i in range(5)]
    store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)

    labels = [
        LabelRow(
            entity_id="BTCUSDT",
            event_time=datetime(2023, 12, 31, tzinfo=UTC),
            label=1.0,
        ),
    ]
    frame = store.point_in_time_join(labels=labels, feature_ids=["mock@1"])
    assert frame.features["mock@1"] == (None,)


def test_duckdb_pit_join_empty_labels() -> None:
    store = DuckDBFeatureStore()
    frame = store.point_in_time_join(labels=[], feature_ids=["mock@1"])
    assert frame.n_rows == 0
    assert frame.features["mock@1"] == ()


def test_duckdb_pit_join_multi_entity() -> None:
    store = DuckDBFeatureStore()
    feat = _MockFeature(lookback=3)
    store.materialize(
        feature=feat, entity_id="BTCUSDT", bars=[_bar(i, close=100.0) for i in range(5)]
    )
    store.materialize(
        feature=feat, entity_id="ETHUSDT", bars=[_bar(i, close=200.0) for i in range(5)]
    )

    labels = [
        LabelRow(entity_id="ETHUSDT", event_time=datetime(2024, 1, 1, 5, tzinfo=UTC), label=1.0),
        LabelRow(entity_id="BTCUSDT", event_time=datetime(2024, 1, 1, 5, tzinfo=UTC), label=1.0),
    ]
    frame = store.point_in_time_join(labels=labels, feature_ids=["mock@1"])
    # Order preserved from the labels list.
    assert frame.features["mock@1"] == (pytest.approx(200.0), pytest.approx(100.0))
