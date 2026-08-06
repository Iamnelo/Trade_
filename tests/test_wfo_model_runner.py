"""End-to-end tests for run_walk_forward_model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from trade.data.schemas import KlineRecord
from trade.features.definitions.log_return import LogReturnN
from trade.features.definitions.realized_vol import RealizedVolN
from trade.mre.types import BacktestConfig
from trade.wfo.model_runner import run_walk_forward_model
from trade.wfo.splitter import walk_forward_splits


def _bars(n: int, seed: int = 42) -> list[KlineRecord]:
    rng = np.random.default_rng(seed)
    price = 100.0
    out: list[KlineRecord] = []
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(n):
        price *= 1.0 + float(rng.normal(0.0005, 0.01))
        h = price * (1 + abs(float(rng.normal(0, 0.005))))
        low = price * (1 - abs(float(rng.normal(0, 0.005))))
        out.append(
            KlineRecord(
                source="bybit",
                category="linear",
                symbol="BTCUSDT",
                interval="60",
                event_time=base + timedelta(hours=i),
                ingest_time=base + timedelta(hours=i, seconds=1),
                open=price,
                high=h,
                low=low,
                close=price,
                volume=1.0,
                turnover=price,
            )
        )
    return out


def _run(bars: list[KlineRecord]):
    feats = [LogReturnN(window=5), RealizedVolN(window=20)]
    splits = walk_forward_splits(n_bars=len(bars), train_bars=300, test_bars=100)
    return run_walk_forward_model(
        bars=bars,
        symbol="BTCUSDT",
        interval="60",
        features=feats,
        label_horizon_bars=6,
        label_up_pct=0.01,
        label_down_pct=0.01,
        splits=splits,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=5.5, slippage_bps=5.0),
        bars_per_year=8760,
        dataset_manifest_ids=["ds1"],
        feature_manifest_ids=[f.spec.full_id for f in feats],
        code_git_sha="deadbeef",
        python_lockfile_sha="cafef00d",
        confidence_threshold=0.35,
        notional_fraction=0.25,
    )


def test_wfo_model_runner_produces_per_fold_reports() -> None:
    bars = _bars(600)
    report = _run(bars)
    assert len(report.folds) > 0
    for fold in report.folds:
        assert fold.report.n_bars > 0
        assert fold.report.strategy_name == "model_lgbm(BTCUSDT)"
        assert len(fold.reproducibility_hash) == 64


def test_wfo_model_runner_is_reproducible() -> None:
    bars = _bars(500)
    a = _run(bars)
    b = _run(bars)
    assert a.reproducibility_hashes == b.reproducibility_hashes
    for fa, fb in zip(a.folds, b.folds, strict=True):
        assert fa.report.final_equity == fb.report.final_equity


def test_wfo_model_runner_returns_empty_on_no_splits() -> None:
    bars = _bars(50)  # too short for any 300/100 split
    report = _run(bars)
    assert report.folds == ()
    assert report.mean_cost_adjusted_sharpe == 0.0
