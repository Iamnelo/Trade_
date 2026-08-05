"""Tests for the feature contract-check framework itself."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import (
    assert_feature_contract,
    assert_feature_deterministic,
    assert_feature_handles_insufficient_history,
    assert_feature_respects_lookback,
)
from trade.features.types import FeatureSpec


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


class _WellBehavedFeature:
    """Mean of last 3 closes — deterministic, respects lookback, returns None early."""

    spec = FeatureSpec(
        name="mean_close",
        version="3",
        inputs=("close",),
        lookback_bars=3,
        availability_delay=timedelta(0),
    )

    def compute(self, history):
        if len(history) < self.spec.lookback_bars:
            return None
        window = history[-self.spec.lookback_bars :]
        return sum(h.close for h in window) / self.spec.lookback_bars


class _NondeterministicFeature:
    spec = _WellBehavedFeature.spec

    def compute(self, history):  # returns different value each call
        if len(history) < 3:
            return None
        return random.random()


class _LookbackViolator:
    """Reads MORE than declared lookback — the classic leak."""

    spec = FeatureSpec(
        name="leaky",
        version="1",
        inputs=("close",),
        lookback_bars=3,
        availability_delay=timedelta(0),
    )

    def compute(self, history):
        if len(history) < 3:
            return None
        # Cheats by using ALL history, not the declared tail.
        return sum(h.close for h in history) / len(history)


class _IgnoresInsufficientHistory:
    spec = FeatureSpec(
        name="broken",
        version="1",
        inputs=("close",),
        lookback_bars=5,
        availability_delay=timedelta(0),
    )

    def compute(self, history):
        # Doesn't check length — returns nonsense on short history.
        return 0.0 if not history else history[-1].close


def test_deterministic_passes_for_well_behaved() -> None:
    assert_feature_deterministic(_WellBehavedFeature(), [_bar(i, i) for i in range(5)])


def test_deterministic_fails_for_nondeterministic() -> None:
    with pytest.raises(AssertionError, match="non-deterministic"):
        assert_feature_deterministic(_NondeterministicFeature(), [_bar(i, i) for i in range(5)])


def test_respects_lookback_passes_for_well_behaved() -> None:
    assert_feature_respects_lookback(_WellBehavedFeature(), [_bar(i, i) for i in range(10)])


def test_respects_lookback_catches_leaky_feature() -> None:
    with pytest.raises(AssertionError, match="lookback"):
        assert_feature_respects_lookback(_LookbackViolator(), [_bar(i, i) for i in range(10)])


def test_insufficient_history_check_catches_broken_feature() -> None:
    with pytest.raises(AssertionError, match="expected None"):
        assert_feature_handles_insufficient_history(
            _IgnoresInsufficientHistory(), [_bar(i) for i in range(10)]
        )


def test_full_contract_composes() -> None:
    # Well-behaved feature passes the full contract.
    assert_feature_contract(_WellBehavedFeature(), [_bar(i, i) for i in range(10)])
