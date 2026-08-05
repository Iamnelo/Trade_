"""Tests for the walk-forward splitter and runner."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.mre.types import BacktestConfig
from trade.strategies.buy_hold import BuyAndHoldStrategy
from trade.wfo.runner import run_walk_forward
from trade.wfo.splitter import purged_embargo_folds, walk_forward_splits


def test_walk_forward_rolling_produces_expected_folds() -> None:
    splits = walk_forward_splits(n_bars=100, train_bars=30, test_bars=10)
    assert splits[0].train_start == 0
    assert splits[0].train_end == 30
    assert splits[0].test_start == 30
    assert splits[0].test_end == 40
    # Rolling windows advance by test_bars.
    assert splits[1].train_start == 10
    assert splits[1].train_end == 40


def test_walk_forward_expanding_grows_train_window() -> None:
    splits = walk_forward_splits(n_bars=100, train_bars=30, test_bars=10, expanding=True)
    assert splits[0].train_start == 0
    assert splits[1].train_start == 0  # expanding: still starts at 0
    assert splits[1].train_end == 40


def test_walk_forward_step_bars_controls_stride() -> None:
    splits = walk_forward_splits(n_bars=100, train_bars=30, test_bars=10, step_bars=5)
    assert splits[1].test_start - splits[0].test_start == 5


def test_walk_forward_empty_when_data_too_short() -> None:
    assert walk_forward_splits(n_bars=10, train_bars=30, test_bars=10) == []


def test_walk_forward_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        walk_forward_splits(n_bars=100, train_bars=0, test_bars=10)
    with pytest.raises(ValueError):
        walk_forward_splits(n_bars=100, train_bars=10, test_bars=10, step_bars=0)


def test_purged_folds_cover_full_range_without_test_overlap() -> None:
    folds = purged_embargo_folds(n_bars=100, n_folds=5)
    test_ranges = [(f.test_start, f.test_end) for f in folds]
    assert test_ranges[0] == (0, 20)
    assert test_ranges[-1] == (80, 100)
    # Each fold's test region excluded from training.
    for f in folds:
        for start, end in f.train_ranges:
            assert not (start < f.test_end and end > f.test_start)


def test_purged_folds_purge_and_embargo() -> None:
    folds = purged_embargo_folds(n_bars=100, n_folds=5, purge_bars=3, embargo_bars=2)
    # Middle fold: test=[40,60). Train excludes [40-3=37, 60+2=62), so
    # train_ranges = [(0, 37), (62, 100)].
    middle = folds[2]
    assert middle.train_ranges == ((0, 37), (62, 100))


def test_purged_folds_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        purged_embargo_folds(n_bars=100, n_folds=1)
    with pytest.raises(ValueError):
        purged_embargo_folds(n_bars=100, n_folds=200)  # more folds than bars
    with pytest.raises(ValueError):
        purged_embargo_folds(n_bars=100, n_folds=5, purge_bars=-1)


def _bars(n: int) -> list[KlineRecord]:
    return [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol="BTCUSDT",
            interval="60",
            event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
            open=100.0 + i * 0.1,
            high=101.0,
            low=99.0,
            close=100.0 + i * 0.1,
            volume=1.0,
            turnover=100.0,
        )
        for i in range(n)
    ]


def test_run_walk_forward_produces_per_fold_reports() -> None:
    bars = _bars(200)
    splits = walk_forward_splits(n_bars=len(bars), train_bars=100, test_bars=25)
    report = run_walk_forward(
        bars=bars,
        strategy_factory=lambda: BuyAndHoldStrategy(symbol="BTCUSDT", notional_fraction=1.0),
        splits=splits,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
        bars_per_year=8760,
        interval="60",
    )
    assert len(report.folds) == len(splits) > 0
    for fold in report.folds:
        assert fold.report.n_bars > 0
        assert fold.report.strategy_name.startswith("buy_hold")


def test_run_walk_forward_is_deterministic() -> None:
    bars = _bars(200)
    splits = walk_forward_splits(n_bars=len(bars), train_bars=100, test_bars=25)
    cfg = BacktestConfig(initial_equity=1000.0, fee_bps=5.5, slippage_bps=5.0)
    a = run_walk_forward(
        bars=bars,
        strategy_factory=lambda: BuyAndHoldStrategy(symbol="BTCUSDT"),
        splits=splits,
        config=cfg,
        bars_per_year=8760,
        interval="60",
    )
    b = run_walk_forward(
        bars=bars,
        strategy_factory=lambda: BuyAndHoldStrategy(symbol="BTCUSDT"),
        splits=splits,
        config=cfg,
        bars_per_year=8760,
        interval="60",
    )
    assert [f.report.final_equity for f in a.folds] == [f.report.final_equity for f in b.folds]
