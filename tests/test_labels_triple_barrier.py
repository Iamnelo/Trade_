"""Tests for triple-barrier label generation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.labels.triple_barrier import triple_barrier_labels, vol_scaled_barriers


def _bar(hour: int, o: float, hi: float, lo: float, c: float) -> KlineRecord:
    return KlineRecord(
        source="bybit",
        category="linear",
        symbol="BTCUSDT",
        interval="60",
        event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour),
        ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour, seconds=1),
        open=o,
        high=hi,
        low=lo,
        close=c,
        volume=1.0,
        turnover=c,
    )


def test_upper_barrier_hit_within_horizon() -> None:
    bars = [
        _bar(0, 100, 100.5, 99.5, 100),
        _bar(1, 100, 100.5, 99.5, 100),
        _bar(2, 100, 106.0, 100.0, 105),  # +5% high -> upper hit if up_pct <= 0.05
    ]
    labels = triple_barrier_labels(bars, horizon_bars=3, up_pct=0.03, down_pct=0.03)
    assert labels[0].label == 1.0


def test_lower_barrier_hit_within_horizon() -> None:
    bars = [
        _bar(0, 100, 100.5, 99.5, 100),
        _bar(1, 100, 100.5, 96.0, 100),  # -4% low -> lower hit if down_pct <= 0.04
    ]
    labels = triple_barrier_labels(bars, horizon_bars=3, up_pct=0.10, down_pct=0.03)
    assert labels[0].label == -1.0


def test_timeout_when_neither_barrier_hit() -> None:
    bars = [_bar(i, 100, 100.1, 99.9, 100) for i in range(5)]
    labels = triple_barrier_labels(bars, horizon_bars=3, up_pct=0.05, down_pct=0.05)
    # First bar's horizon covers bars 1..3, none exceed 5% band.
    assert labels[0].label == 0.0


def test_both_hit_same_bar_labels_ambiguous_as_zero() -> None:
    bars = [
        _bar(0, 100, 100.5, 99.5, 100),
        _bar(1, 100, 106.0, 94.0, 100),  # BOTH triggered in one bar
    ]
    labels = triple_barrier_labels(bars, horizon_bars=1, up_pct=0.03, down_pct=0.03)
    assert labels[0].label == 0.0


def test_last_bar_has_no_future_and_labels_timeout() -> None:
    bars = [_bar(i, 100, 100.5, 99.5, 100) for i in range(3)]
    labels = triple_barrier_labels(bars, horizon_bars=5, up_pct=0.03, down_pct=0.03)
    assert labels[-1].label == 0.0
    # And there is one label per bar.
    assert len(labels) == 3


def test_labels_carry_entity_and_event_time() -> None:
    bars = [_bar(0, 100, 101, 99, 100)]
    labels = triple_barrier_labels(bars, horizon_bars=1, up_pct=0.03, down_pct=0.03)
    assert labels[0].entity_id == "BTCUSDT"
    assert labels[0].event_time == datetime(2024, 1, 1, tzinfo=UTC)


def test_triple_barrier_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        triple_barrier_labels([], horizon_bars=0, up_pct=0.01, down_pct=0.01)
    with pytest.raises(ValueError):
        triple_barrier_labels([], horizon_bars=1, up_pct=0.0, down_pct=0.01)


def test_vol_scaled_barriers_symmetric() -> None:
    up, dn = vol_scaled_barriers(atr=2.0, entry_price=100.0, k=1.5)
    assert up == dn == pytest.approx(0.03)


def test_vol_scaled_barriers_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        vol_scaled_barriers(atr=1.0, entry_price=0.0)
    with pytest.raises(ValueError):
        vol_scaled_barriers(atr=-1.0, entry_price=100.0)
