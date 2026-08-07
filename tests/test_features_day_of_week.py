"""Contract + textbook tests for cyclic day-of-week encoding."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import assert_feature_contract
from trade.features.definitions.day_of_week import DayOfWeekCos, DayOfWeekSin


def _bar_on(date: datetime) -> KlineRecord:
    return KlineRecord(
        source="bybit",
        category="linear",
        symbol="BTCUSDT",
        interval="60",
        event_time=date,
        ingest_time=date + timedelta(seconds=1),
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=1.0,
        turnover=100.0,
    )


def _daily_stream(n: int) -> list[KlineRecord]:
    start = datetime(2024, 1, 1, tzinfo=UTC)  # Monday
    return [_bar_on(start + timedelta(days=i)) for i in range(n)]


def test_dow_sin_contract() -> None:
    assert_feature_contract(DayOfWeekSin(), _daily_stream(20))


def test_dow_cos_contract() -> None:
    assert_feature_contract(DayOfWeekCos(), _daily_stream(20))


def test_monday_gives_dow_sin_zero_cos_one() -> None:
    # 2024-01-01 is a Monday (weekday() == 0)
    monday = datetime(2024, 1, 1, tzinfo=UTC)
    assert DayOfWeekSin().compute([_bar_on(monday)]) == pytest.approx(0.0)
    assert DayOfWeekCos().compute([_bar_on(monday)]) == pytest.approx(1.0)


def test_wednesday_matches_formula() -> None:
    # 2024-01-03 is a Wednesday (weekday() == 2)
    wednesday = datetime(2024, 1, 3, tzinfo=UTC)
    sin_val = DayOfWeekSin().compute([_bar_on(wednesday)])
    cos_val = DayOfWeekCos().compute([_bar_on(wednesday)])
    assert sin_val == pytest.approx(math.sin(2 * math.pi * 2 / 7.0))
    assert cos_val == pytest.approx(math.cos(2 * math.pi * 2 / 7.0))
