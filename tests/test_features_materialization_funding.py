"""Tests for funding-rate materialisation into the FeatureStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import FundingRecord
from trade.features.materialization.funding import (
    FUNDING_RATE_SPEC,
    funding_rate_mean_spec,
    materialize_funding_rate,
    materialize_funding_rate_mean,
)
from trade.features.store import InMemoryFeatureStore
from trade.features.types import LabelRow


def _funding(hour: int, rate: float, symbol: str = "BTCUSDT") -> FundingRecord:
    ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour)
    return FundingRecord(
        source="bybit",
        category="linear",
        symbol=symbol,
        event_time=ts,
        ingest_time=ts + timedelta(seconds=1),
        funding_rate=rate,
    )


def test_funding_spec_matches_id() -> None:
    assert FUNDING_RATE_SPEC.full_id == "funding_rate@1"
    assert funding_rate_mean_spec(3).full_id == "funding_rate_mean@3"


def test_materialize_funding_rate_one_row_per_settlement() -> None:
    settlements = [_funding(0, 0.0001), _funding(8, -0.0002), _funding(16, 0.00005)]
    rows = materialize_funding_rate(settlements)
    assert len(rows) == 3
    # Sorted by event_time.
    assert [r.event_time.hour for r in rows] == [0, 8, 16]
    assert rows[0].feature_id == "funding_rate@1"
    assert rows[0].availability_time == rows[0].event_time  # zero delay


def test_materialize_funding_rate_mean_over_window() -> None:
    settlements = [_funding(i * 8, 0.0001 * (i + 1)) for i in range(5)]
    rows = materialize_funding_rate_mean(settlements, window=3)
    # First value is at index 2 (3 settlements needed).
    assert len(rows) == 3
    assert rows[0].value == pytest.approx((0.0001 + 0.0002 + 0.0003) / 3)


def test_pit_join_respects_funding_availability() -> None:
    store = InMemoryFeatureStore()
    settlements = [_funding(i * 8, 0.0001 * (i + 1)) for i in range(3)]
    store.put_materialized(
        materialized=materialize_funding_rate(settlements),
        feature_spec=FUNDING_RATE_SPEC,
    )

    # A label 4 hours after the 08:00 settlement should see the 08:00 rate,
    # not the 16:00 one.
    labels = [
        LabelRow(
            entity_id="BTCUSDT",
            event_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
            label=1.0,
        )
    ]
    frame = store.point_in_time_join(labels=labels, feature_ids=["funding_rate@1"])
    assert frame.features["funding_rate@1"][0] == pytest.approx(0.0002)


def test_funding_rate_mean_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        funding_rate_mean_spec(1)
