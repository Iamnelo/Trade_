"""Contract + textbook tests for VolumeZScoreN and TurnoverZScoreN."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import assert_feature_contract
from trade.features.definitions.volume_zscore import TurnoverZScoreN, VolumeZScoreN


def _bars(volumes: list[float], turnovers: list[float] | None = None) -> list[KlineRecord]:
    if turnovers is None:
        turnovers = [v * 100.0 for v in volumes]
    return [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol="BTCUSDT",
            interval="60",
            event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=v,
            turnover=t,
        )
        for i, (v, t) in enumerate(zip(volumes, turnovers, strict=True))
    ]


def test_volume_zscore_contract() -> None:
    volumes = [10.0 + i for i in range(50)]
    assert_feature_contract(VolumeZScoreN(window=20), _bars(volumes))


def test_turnover_zscore_contract() -> None:
    volumes = [10.0 + i for i in range(50)]
    assert_feature_contract(TurnoverZScoreN(window=20), _bars(volumes))


def test_zscore_zero_when_current_matches_history_mean() -> None:
    # 20 constant history bars then a matching final bar.
    volumes = [50.0] * 20 + [50.0]
    got = VolumeZScoreN(window=20).compute(_bars(volumes))
    assert got == pytest.approx(0.0)


def test_zscore_positive_on_volume_spike() -> None:
    volumes = [10.0] * 20 + [100.0]
    got = VolumeZScoreN(window=20).compute(_bars(volumes))
    assert got is not None and got > 0.0


def test_zscore_negative_on_volume_dip() -> None:
    volumes = [50.0 + (i % 3) * 5 for i in range(20)] + [1.0]
    got = VolumeZScoreN(window=20).compute(_bars(volumes))
    assert got is not None and got < 0.0


def test_zscore_matches_manual_formula() -> None:
    # Z-score of the last value uses the FULL window (self included) so it
    # stays well-defined when the history is momentarily constant.
    volumes = [10.0, 12.0, 14.0, 16.0, 18.0, 30.0]
    n = len(volumes)
    mean = sum(volumes) / n
    var = sum((v - mean) ** 2 for v in volumes) / n
    sd = math.sqrt(var)
    expected = (volumes[-1] - mean) / sd
    got = VolumeZScoreN(window=n - 1).compute(_bars(volumes))
    assert got == pytest.approx(expected)


def test_turnover_zscore_uses_turnover_not_volume() -> None:
    # Volumes constant, turnovers spiking — turnover z-score must react.
    n = 20
    volumes = [10.0] * (n + 1)
    turnovers = [100.0] * n + [500.0]
    got = TurnoverZScoreN(window=n).compute(_bars(volumes, turnovers))
    assert got is not None and got > 0.0
