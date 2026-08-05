"""Contract test for the RSI-14 reference feature.

Also verifies textbook RSI values on known inputs so we catch regressions
in the formula itself, not just the contract layer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import assert_feature_contract
from trade.features.definitions.rsi import RSI14


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


def test_rsi14_contract() -> None:
    # 30 bars of noisy monotone uptrend so contract tests have enough history.
    closes = [100.0 + i * 0.3 + (0.5 if i % 3 == 0 else -0.2) for i in range(30)]
    assert_feature_contract(RSI14(), _bars(closes))


def test_rsi14_all_up_returns_100() -> None:
    # Purely rising closes -> no losses -> RSI = 100 by definition.
    closes = [100.0 + i for i in range(15)]
    assert RSI14().compute(_bars(closes)) == 100.0


def test_rsi14_returns_none_with_insufficient_history() -> None:
    closes = [100.0 + i for i in range(10)]  # < 15 bars
    assert RSI14().compute(_bars(closes)) is None


def test_rsi14_symmetric_updown_near_50() -> None:
    # Alternating +1 / -1 pattern -> avg_gain == avg_loss -> RSI = 50.
    closes = [100.0]
    for _ in range(14):
        closes.append(closes[-1] + 1.0)
        closes.append(closes[-1] - 1.0)
    value = RSI14().compute(_bars(closes))
    assert value is not None
    assert value == pytest.approx(50.0, abs=1.0)
