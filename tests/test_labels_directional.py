"""Tests for triple_barrier_labels_directional (drops flat rows)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trade.data.schemas import KlineRecord
from trade.labels.triple_barrier import (
    triple_barrier_labels,
    triple_barrier_labels_directional,
)


def _bars(
    closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None
) -> list[KlineRecord]:
    if highs is None:
        highs = closes
    if lows is None:
        lows = closes
    return [
        KlineRecord(
            source="synthetic",
            category="linear",
            symbol="BTCUSDT",
            interval="60",
            event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
            open=c,
            high=h,
            low=lo,
            close=c,
            volume=1.0,
            turnover=c,
        )
        for i, (c, h, lo) in enumerate(zip(closes, highs, lows, strict=True))
    ]


def test_directional_drops_all_flat_rows() -> None:
    # Bars barely move → all barriers time out → all labels are 0.
    bars = _bars([100.0 + 0.001 * i for i in range(30)])
    kept = triple_barrier_labels_directional(bars, horizon_bars=5, up_pct=0.05, down_pct=0.05)
    assert kept == []


def test_directional_keeps_only_up_or_down_bars() -> None:
    # Alternating big-up, big-down closes → every bar hits a barrier fast.
    closes = [100.0, 106.0, 96.0, 105.0, 95.0, 105.0, 95.0]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    kept = triple_barrier_labels_directional(
        _bars(closes, highs, lows), horizon_bars=1, up_pct=0.03, down_pct=0.03
    )
    for lb in kept:
        assert lb.label in {1.0, -1.0}


def test_directional_is_subset_of_original() -> None:
    closes = [100.0 + (i % 5) * 0.5 for i in range(40)]
    highs = [c + 0.6 for c in closes]
    lows = [c - 0.6 for c in closes]
    horizon = 3
    all_labels = triple_barrier_labels(
        _bars(closes, highs, lows), horizon_bars=horizon, up_pct=0.005, down_pct=0.005
    )
    kept = triple_barrier_labels_directional(
        _bars(closes, highs, lows), horizon_bars=horizon, up_pct=0.005, down_pct=0.005
    )
    kept_times = {lb.event_time for lb in kept}
    all_directional_times = {lb.event_time for lb in all_labels if lb.label != 0.0}
    assert kept_times == all_directional_times
    assert all(lb.label in {1.0, -1.0} for lb in kept)
