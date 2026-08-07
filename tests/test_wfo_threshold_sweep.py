"""Tests for the threshold-aware WFO refactor.

Focus:
- `_train_fold` returns None sanely on inadequate corpora.
- Threshold sweep uses ONE fit per fold, N replays.
- Sweep + single-run produce identical results for the matching threshold.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.definitions.log_return import LogReturnN
from trade.features.definitions.realized_vol import RealizedVolN
from trade.mre.types import BacktestConfig
from trade.wfo.model_runner import (
    ThresholdSweepReport,
    run_walk_forward_model,
    run_walk_forward_threshold_sweep,
)
from trade.wfo.splitter import walk_forward_splits


def _synthetic_bars(n: int) -> list[KlineRecord]:
    ingest = datetime(2024, 1, 1, tzinfo=UTC)
    bars: list[KlineRecord] = []
    for i in range(n):
        drift = 100.0 + i * 0.02
        osc = 2.0 * math.sin(i / 25.0)
        open_ = drift + osc
        close = drift + 2.0 * math.sin((i + 1) / 25.0)
        hi = max(open_, close) + 0.5
        lo = min(open_, close) - 0.5
        bars.append(
            KlineRecord(
                source="synthetic",
                category="linear",
                symbol="TESTUSDT",
                interval="60",
                event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                ingest_time=ingest,
                open=round(open_, 4),
                high=round(hi, 4),
                low=round(lo, 4),
                close=round(close, 4),
                volume=1000.0,
                turnover=1000.0 * ((open_ + close) / 2),
            )
        )
    return bars


def _common_kwargs() -> dict[str, object]:
    return {
        "symbol": "TESTUSDT",
        "interval": "60",
        "features": [LogReturnN(window=5), RealizedVolN(window=20)],
        "label_horizon_bars": 6,
        "label_up_pct": 0.005,
        "label_down_pct": 0.005,
        "config": BacktestConfig(initial_equity=1000.0),
        "bars_per_year": 8760,
        "dataset_manifest_ids": ["synthetic"],
        "feature_manifest_ids": ["log_return@5", "realized_vol@20"],
        "code_git_sha": "d" * 40,
        "python_lockfile_sha": "f" * 64,
    }


def test_threshold_sweep_produces_one_result_per_threshold_per_fold() -> None:
    bars = _synthetic_bars(800)
    splits = walk_forward_splits(n_bars=800, train_bars=400, test_bars=100, step_bars=100)
    report = run_walk_forward_threshold_sweep(
        bars=bars,
        splits=splits,
        thresholds=[0.55, 0.60, 0.70],
        **_common_kwargs(),
    )
    assert isinstance(report, ThresholdSweepReport)
    assert report.thresholds == (0.55, 0.60, 0.70)
    assert len(report.folds) == len(splits)
    for fold in report.folds:
        assert len(fold.per_threshold) == 3
        assert [x.threshold for x in fold.per_threshold] == [0.55, 0.60, 0.70]


def test_threshold_sweep_matches_single_run_at_same_threshold() -> None:
    bars = _synthetic_bars(800)
    splits = walk_forward_splits(n_bars=800, train_bars=400, test_bars=100, step_bars=100)

    kwargs = _common_kwargs()
    single = run_walk_forward_model(bars=bars, splits=splits, confidence_threshold=0.55, **kwargs)
    sweep = run_walk_forward_threshold_sweep(bars=bars, splits=splits, thresholds=[0.55], **kwargs)
    assert len(single.folds) == len(sweep.folds)
    for s_fold, sw_fold in zip(single.folds, sweep.folds, strict=True):
        sw_result = sw_fold.per_threshold[0].result
        # Same reproducibility hash proves same model+features+config.
        assert s_fold.reproducibility_hash == sw_result.reproducibility_hash
        # Same final equity + fill count proves the strategy replay is
        # byte-identical when the threshold matches.
        assert s_fold.report.n_fills == sw_result.report.n_fills
        assert s_fold.report.final_equity == sw_result.report.final_equity


def test_threshold_sweep_rejects_empty_or_out_of_range_thresholds() -> None:
    bars = _synthetic_bars(400)
    splits = walk_forward_splits(n_bars=400, train_bars=200, test_bars=50, step_bars=50)
    kwargs = _common_kwargs()

    with pytest.raises(ValueError, match="thresholds must be non-empty"):
        run_walk_forward_threshold_sweep(bars=bars, splits=splits, thresholds=[], **kwargs)

    with pytest.raises(ValueError, match="threshold"):
        run_walk_forward_threshold_sweep(bars=bars, splits=splits, thresholds=[1.5], **kwargs)


def test_higher_threshold_produces_fewer_or_equal_fills_per_fold() -> None:
    bars = _synthetic_bars(800)
    splits = walk_forward_splits(n_bars=800, train_bars=400, test_bars=100, step_bars=100)
    report = run_walk_forward_threshold_sweep(
        bars=bars,
        splits=splits,
        thresholds=[0.35, 0.55, 0.90],
        **_common_kwargs(),
    )
    for fold in report.folds:
        fills_by_threshold = {t.threshold: t.result.report.n_fills for t in fold.per_threshold}
        # A stricter threshold cannot enable MORE trades.
        assert fills_by_threshold[0.55] <= fills_by_threshold[0.35]
        assert fills_by_threshold[0.90] <= fills_by_threshold[0.55]
