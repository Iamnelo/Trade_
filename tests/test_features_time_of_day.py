"""Contract + textbook tests for cyclic hour-of-day encoding."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import assert_feature_contract
from trade.features.definitions.time_of_day import HourOfDayCos, HourOfDaySin


def _bars_at(hours: list[int]) -> list[KlineRecord]:
    return [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol="BTCUSDT",
            interval="60",
            event_time=datetime(2024, 1, 1, h, tzinfo=UTC),
            ingest_time=datetime(2024, 1, 1, h, tzinfo=UTC) + timedelta(seconds=1),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1.0,
            turnover=100.0,
        )
        for h in hours
    ]


def _hourly_stream(n: int) -> list[KlineRecord]:
    return [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol="BTCUSDT",
            interval="60",
            event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
            open=100.0 + i,
            high=100.0 + i,
            low=100.0 + i,
            close=100.0 + i,
            volume=1.0,
            turnover=100.0 + i,
        )
        for i in range(n)
    ]


def test_hour_sin_contract() -> None:
    assert_feature_contract(HourOfDaySin(), _hourly_stream(50))


def test_hour_cos_contract() -> None:
    assert_feature_contract(HourOfDayCos(), _hourly_stream(50))


def test_hour_sin_matches_formula_at_various_hours() -> None:
    feat = HourOfDaySin()
    for hour in [0, 3, 6, 12, 18, 23]:
        got = feat.compute(_bars_at([hour]))
        assert got == pytest.approx(math.sin(2 * math.pi * hour / 24.0))


def test_hour_sin_is_zero_at_midnight_and_noon() -> None:
    feat = HourOfDaySin()
    assert feat.compute(_bars_at([0])) == pytest.approx(0.0)
    assert feat.compute(_bars_at([12])) == pytest.approx(0.0, abs=1e-9)


def test_hour_cos_is_one_at_midnight_and_minus_one_at_noon() -> None:
    feat = HourOfDayCos()
    assert feat.compute(_bars_at([0])) == pytest.approx(1.0)
    assert feat.compute(_bars_at([12])) == pytest.approx(-1.0)


def test_only_last_bar_matters() -> None:
    # Feature reads only history[-1]; earlier hours must not shift the value.
    feat = HourOfDaySin()
    v_at_9 = feat.compute(_bars_at([9]))
    v_at_9_with_prefix = feat.compute(_bars_at([0, 3, 7, 9]))
    assert v_at_9 == v_at_9_with_prefix
