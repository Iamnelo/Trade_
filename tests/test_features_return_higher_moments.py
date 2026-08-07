"""Contract + textbook tests for ReturnSkewN and ReturnKurtosisN."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import assert_feature_contract
from trade.features.definitions.return_higher_moments import (
    ReturnKurtosisN,
    ReturnSkewN,
)


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


def _series_from_returns(returns: list[float], seed_close: float = 100.0) -> list[float]:
    closes = [seed_close]
    for r in returns:
        closes.append(closes[-1] * math.exp(r))
    return closes


def test_return_skew_contract() -> None:
    closes = [100.0 * (1 + 0.001 * i) for i in range(60)]
    assert_feature_contract(ReturnSkewN(window=20), _bars(closes))


def test_return_kurtosis_contract() -> None:
    closes = [100.0 * (1 + 0.001 * i) for i in range(60)]
    assert_feature_contract(ReturnKurtosisN(window=20), _bars(closes))


def test_symmetric_returns_have_zero_skew() -> None:
    rets = [0.01, -0.01, 0.02, -0.02, 0.005, -0.005]
    closes = _series_from_returns(rets)
    got = ReturnSkewN(window=len(rets)).compute(_bars(closes))
    assert got == pytest.approx(0.0, abs=1e-9)


def test_positive_skew_when_one_big_up_move() -> None:
    rets = [-0.001, -0.001, -0.001, -0.001, -0.001, 0.05]
    closes = _series_from_returns(rets)
    got = ReturnSkewN(window=len(rets)).compute(_bars(closes))
    assert got is not None and got > 0.0


def test_kurtosis_positive_when_returns_have_outlier() -> None:
    rets = [0.001, -0.001, 0.001, -0.001, 0.001, -0.001, 0.05]
    closes = _series_from_returns(rets)
    got = ReturnKurtosisN(window=len(rets)).compute(_bars(closes))
    assert got is not None and got > 0.0


def test_kurtosis_near_zero_for_gaussian_like() -> None:
    # Draw a deterministic-looking small symmetric spread of returns —
    # this is a smoke check, not a Gaussianity assertion.
    rets = [0.01, -0.01, 0.005, -0.005, 0.01, -0.01, 0.005, -0.005] * 5
    closes = _series_from_returns(rets)
    got = ReturnKurtosisN(window=len(rets)).compute(_bars(closes))
    # 2-value alternating series is bimodal → negative excess kurtosis.
    assert got is not None and got < 0.0


def test_skew_returns_none_before_window() -> None:
    feat = ReturnSkewN(window=20)
    assert feat.compute(_bars([100.0, 101.0])) is None


def test_skew_rejects_tiny_window() -> None:
    with pytest.raises(ValueError):
        ReturnSkewN(window=2)


def test_kurtosis_rejects_tiny_window() -> None:
    with pytest.raises(ValueError):
        ReturnKurtosisN(window=3)
