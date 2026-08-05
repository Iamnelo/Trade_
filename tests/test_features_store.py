"""Tests for InMemoryFeatureStore — PIT-correctness lives or dies here."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.store import InMemoryFeatureStore, materialize_feature
from trade.features.types import (
    FeatureSpec,
    LabelRow,
    MaterializedFeature,
)


class _MockFeature:
    def __init__(
        self,
        *,
        lookback: int = 3,
        availability_delay: timedelta = timedelta(0),
    ) -> None:
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
        # Return the mean close of the last N bars.
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


def test_materialize_skips_bars_with_insufficient_history() -> None:
    feat = _MockFeature(lookback=3)
    bars = [_bar(i, close=100.0 + i) for i in range(5)]
    rows = materialize_feature(feature=feat, entity_id="BTCUSDT", bars=bars)
    # Bars 0, 1 lack history; bars 2, 3, 4 produce values.
    assert len(rows) == 3
    assert [r.event_time.hour for r in rows] == [2, 3, 4]


def test_pit_join_returns_latest_available_feature() -> None:
    store = InMemoryFeatureStore()
    feat = _MockFeature(lookback=3, availability_delay=timedelta(0))
    bars = [_bar(i, close=100.0 + i) for i in range(6)]
    store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)

    # Label at hour 5 => available features are those with availability_time <= hour 5.
    labels = [
        LabelRow(
            entity_id="BTCUSDT",
            event_time=datetime(2024, 1, 1, 5, tzinfo=UTC),
            label=1.0,
        ),
    ]
    frame = store.point_in_time_join(labels=labels, feature_ids=["mock@1"])
    assert frame.n_rows == 1
    # The latest available feature is bar 5 (mean of 103, 104, 105 = 104).
    assert frame.features["mock@1"][0] == pytest.approx(104.0)


def test_pit_join_yields_none_when_no_feature_before_label() -> None:
    store = InMemoryFeatureStore()
    feat = _MockFeature(lookback=3)
    bars = [_bar(i) for i in range(6)]
    store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)

    labels = [
        LabelRow(
            entity_id="BTCUSDT",
            event_time=datetime(2023, 12, 31, tzinfo=UTC),  # before any feature
            label=1.0,
        ),
    ]
    frame = store.point_in_time_join(labels=labels, feature_ids=["mock@1"])
    assert frame.features["mock@1"] == (None,)


def test_pit_join_never_uses_features_from_the_future() -> None:
    """The core PIT contract: no feature with availability_time > label.event_time may appear."""
    store = InMemoryFeatureStore()
    feat = _MockFeature(lookback=3, availability_delay=timedelta(0))
    bars = [_bar(i, close=100.0 + i) for i in range(10)]
    store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)

    label = LabelRow(
        entity_id="BTCUSDT",
        event_time=datetime(2024, 1, 1, 4, tzinfo=UTC),
        label=1.0,
    )
    frame = store.point_in_time_join(labels=[label], feature_ids=["mock@1"])
    # Latest available feature at hour 4 is bar 4 (mean of 102, 103, 104 = 103.0).
    assert frame.features["mock@1"][0] == pytest.approx(103.0)


def test_pit_join_respects_availability_delay() -> None:
    """A 60-minute delay means the bar-4 feature is only usable at hour 5."""
    store = InMemoryFeatureStore()
    feat = _MockFeature(lookback=3, availability_delay=timedelta(hours=1))
    bars = [_bar(i, close=100.0 + i) for i in range(10)]
    store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)

    # At event_time=4:59 the bar-4 feature is not yet available; latest usable is bar 3.
    label = LabelRow(
        entity_id="BTCUSDT",
        event_time=datetime(2024, 1, 1, 4, 59, tzinfo=UTC),
        label=1.0,
    )
    frame = store.point_in_time_join(labels=[label], feature_ids=["mock@1"])
    # Bar 3 feature = mean(101, 102, 103) = 102.
    assert frame.features["mock@1"][0] == pytest.approx(102.0)


def test_pit_join_isolates_across_entities() -> None:
    store = InMemoryFeatureStore()
    feat = _MockFeature(lookback=3)
    store.materialize(
        feature=feat, entity_id="BTCUSDT", bars=[_bar(i, close=100.0) for i in range(5)]
    )
    store.materialize(
        feature=feat, entity_id="ETHUSDT", bars=[_bar(i, close=200.0) for i in range(5)]
    )

    labels = [
        LabelRow(entity_id="BTCUSDT", event_time=datetime(2024, 1, 1, 5, tzinfo=UTC), label=1.0),
        LabelRow(entity_id="ETHUSDT", event_time=datetime(2024, 1, 1, 5, tzinfo=UTC), label=1.0),
    ]
    frame = store.point_in_time_join(labels=labels, feature_ids=["mock@1"])
    assert frame.features["mock@1"] == (pytest.approx(100.0), pytest.approx(200.0))


def test_store_has_no_latest_features_method() -> None:
    """HARD REQUIREMENT: no bypass API. Any getter that skips PIT is a spec violation."""
    store = InMemoryFeatureStore()
    banned = {"get_latest_features", "load_latest", "snapshot_now", "features_now"}
    exposed = set(dir(store))
    assert banned.isdisjoint(exposed), (
        f"FeatureStore exposes banned latest-feature methods: {banned & exposed}"
    )


def test_put_materialized_accepts_precomputed_rows() -> None:
    """External materialisers (e.g., a distributed job) can push in rows directly."""
    store = InMemoryFeatureStore()
    spec = _MockFeature().spec
    row = MaterializedFeature(
        feature_id=spec.full_id,
        entity_id="BTCUSDT",
        event_time=datetime(2024, 1, 1, 5, tzinfo=UTC),
        availability_time=datetime(2024, 1, 1, 5, tzinfo=UTC),
        value=42.0,
    )
    store.put_materialized(materialized=[row], feature_spec=spec)
    labels = [
        LabelRow(entity_id="BTCUSDT", event_time=datetime(2024, 1, 1, 6, tzinfo=UTC), label=1.0),
    ]
    frame = store.point_in_time_join(labels=labels, feature_ids=[spec.full_id])
    assert frame.features[spec.full_id][0] == 42.0
