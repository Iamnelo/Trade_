"""Tests for the feature-store value types."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trade.features.types import (
    FeatureSpec,
    LabelRow,
    MaterializedFeature,
    TrainingFrame,
)


def test_feature_spec_full_id_composition() -> None:
    spec = FeatureSpec(
        name="rsi_close",
        version="14",
        inputs=("close",),
        lookback_bars=15,
        availability_delay=timedelta(0),
    )
    assert spec.full_id == "rsi_close@14"


def test_materialized_feature_stores_availability_time() -> None:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    row = MaterializedFeature(
        feature_id="rsi_close@14",
        entity_id="BTCUSDT",
        event_time=now,
        availability_time=now + timedelta(seconds=5),
        value=42.0,
    )
    assert row.availability_time - row.event_time == timedelta(seconds=5)


def test_training_frame_derived_properties() -> None:
    frame = TrainingFrame(
        entity_ids=("BTCUSDT", "BTCUSDT"),
        event_times=(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, 1, tzinfo=UTC)),
        labels=(1.0, -1.0),
        features={"rsi@14": (50.0, None), "sma@20": (100.0, 101.0)},
    )
    assert frame.n_rows == 2
    assert frame.feature_ids == ("rsi@14", "sma@20")


def test_label_row_frozen() -> None:
    row = LabelRow(entity_id="BTCUSDT", event_time=datetime(2024, 1, 1, tzinfo=UTC), label=1.0)
    # dataclass(frozen=True): reassignment raises
    try:
        row.label = 2.0  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("LabelRow must be frozen")
