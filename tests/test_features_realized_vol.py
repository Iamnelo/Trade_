"""Contract + textbook tests for RealizedVolN."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import assert_feature_contract
from trade.features.definitions.realized_vol import RealizedVolN


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


def test_realized_vol_contract() -> None:
    closes = [100.0 * (1.001**i) + (0.5 if i % 2 == 0 else -0.3) for i in range(40)]
    assert_feature_contract(RealizedVolN(window=10), _bars(closes))


def test_realized_vol_zero_for_flat_series() -> None:
    assert RealizedVolN(window=5).compute(_bars([100.0] * 20)) == pytest.approx(0.0)


def test_realized_vol_positive_for_noisy_series() -> None:
    closes = [100.0, 105.0, 95.0, 110.0, 90.0, 100.0]
    v = RealizedVolN(window=5).compute(_bars(closes))
    assert v is not None and v > 0.0


def test_realized_vol_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        RealizedVolN(window=1)
