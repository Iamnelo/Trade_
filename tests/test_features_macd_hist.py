"""Contract + smoke tests for MACDHistogram."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import assert_feature_contract
from trade.features.definitions.macd_hist import MACDHistogram


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


def test_macd_hist_contract_default_params() -> None:
    # Default lookback is 3*26 + 9 = 87; provide ~120 bars for a comfortable margin.
    closes = [100.0 + 0.3 * i for i in range(120)]
    assert_feature_contract(MACDHistogram(), _bars(closes))


def test_macd_hist_flat_series_returns_zero() -> None:
    feat = MACDHistogram(fast=3, slow=6, signal=3)
    # Constant closes -> both EMAs equal -> MACD line 0 -> signal 0 -> hist 0.
    v = feat.compute(_bars([100.0] * 60))
    assert v == pytest.approx(0.0, abs=1e-12)


def test_macd_hist_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        MACDHistogram(fast=26, slow=12)  # inverted
    with pytest.raises(ValueError):
        MACDHistogram(fast=12, slow=26, signal=0)
