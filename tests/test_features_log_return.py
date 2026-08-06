"""Contract + textbook tests for LogReturnN."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import assert_feature_contract
from trade.features.definitions.log_return import LogReturnN


def _bars(closes: list[float]) -> list[KlineRecord]:
    return [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol="BTCUSDT",
            interval="60",
            event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
            open=c,
            high=c,
            low=c,
            close=c,
            volume=1.0,
            turnover=c,
        )
        for i, c in enumerate(closes)
    ]


def test_log_return_contract() -> None:
    closes = [100.0 + i * 0.5 for i in range(30)]
    assert_feature_contract(LogReturnN(window=5), _bars(closes))


def test_log_return_matches_manual_formula() -> None:
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    feat = LogReturnN(window=5)
    value = feat.compute(_bars(closes))
    assert value == pytest.approx(math.log(105.0 / 100.0))


def test_log_return_zero_when_flat() -> None:
    assert LogReturnN(window=3).compute(_bars([100.0] * 10)) == pytest.approx(0.0)


def test_log_return_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        LogReturnN(window=0)
