"""Contract + textbook tests for VolRegime."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import assert_feature_contract
from trade.features.definitions.vol_regime import VolRegime


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


def _stitch(quiet_part: list[float], loud_part: list[float]) -> list[float]:
    return quiet_part + loud_part


def test_vol_regime_contract() -> None:
    quiet = [100.0 + 0.001 * i for i in range(80)]
    loud = [100.1 + (i % 5) * 2.0 - 5.0 for i in range(40)]
    assert_feature_contract(
        VolRegime(short_window=20, long_window=100),
        _bars(_stitch(quiet, loud)),
    )


def test_vol_regime_positive_when_recent_vol_higher() -> None:
    # 100-bar quiet baseline + 20-bar loud tail → short-window vol >> long-window vol
    # so log(short/long) should be positive.
    quiet = [100.0 + 0.0005 * i for i in range(100)]
    loud = [100.5 * (1 + 0.02 * ((-1) ** i)) for i in range(20)]
    got = VolRegime(short_window=20, long_window=100).compute(_bars(_stitch(quiet, loud)))
    assert got is not None and got > 0.0


def test_vol_regime_negative_when_recent_quieter_than_long() -> None:
    # Loud baseline + quiet recent tail → short-window vol < long-window vol
    loud = [100.0 * (1 + 0.02 * ((-1) ** i)) for i in range(100)]
    quiet = [100.0 + 0.0001 * i for i in range(20)]
    got = VolRegime(short_window=20, long_window=100).compute(_bars(_stitch(loud, quiet)))
    assert got is not None and got < 0.0


def test_vol_regime_rejects_bad_windows() -> None:
    with pytest.raises(ValueError):
        VolRegime(short_window=1, long_window=100)
    with pytest.raises(ValueError):
        VolRegime(short_window=50, long_window=50)
    with pytest.raises(ValueError):
        VolRegime(short_window=100, long_window=20)


def test_vol_regime_none_before_long_window_ready() -> None:
    assert VolRegime(short_window=20, long_window=100).compute(_bars([100.0] * 50)) is None
