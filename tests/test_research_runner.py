"""Runner: CSV loader + end-to-end run on a tiny synthetic corpus.

This exercises the actual code path — CSV -> features -> WFO -> gates ->
result JSON — on a small deterministic corpus so it stays fast enough
for CI. The strategy itself makes no directional claim; we only assert
the wiring produces a well-formed ExperimentResult.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from trade.research.experiment import (
    BacktestSpec,
    DataSpec,
    ExperimentSpec,
    LabelSpec,
    ModelSpec,
    RobustnessGateSpec,
    StrategySpec,
    WFOSpec,
)
from trade.research.runner import load_klines_csv, run_experiment


def _write_synthetic_csv(path: Path, *, n_bars: int = 800) -> None:
    """Deterministic gently-drifting OHLC series with a sine oscillation."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_time_ms", "open", "high", "low", "close", "volume", "turnover"])
        base_ms = 1_700_000_000_000
        for i in range(n_bars):
            drift = 100.0 + i * 0.02
            osc = 2.0 * math.sin(i / 25.0)
            open_ = drift + osc
            close = drift + 2.0 * math.sin((i + 1) / 25.0)
            hi = max(open_, close) + 0.5
            lo = min(open_, close) - 0.5
            w.writerow(
                [
                    base_ms + i * 3_600_000,
                    round(open_, 4),
                    round(hi, 4),
                    round(lo, 4),
                    round(close, 4),
                    1000.0,
                    1000.0 * ((open_ + close) / 2),
                ]
            )


def test_load_klines_csv_parses_expected_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "small.csv"
    _write_synthetic_csv(csv_path, n_bars=10)
    bars = load_klines_csv(csv_path, symbol="BTCUSDT", interval="60")
    assert len(bars) == 10
    assert bars[0].symbol == "BTCUSDT"
    assert bars[0].interval == "60"
    assert bars[0].source == "csv"


def test_run_experiment_produces_well_formed_result(tmp_path: Path) -> None:
    csv_path = tmp_path / "syn.csv"
    _write_synthetic_csv(csv_path, n_bars=800)

    spec = ExperimentSpec(
        name="synthetic_smoke",
        data=DataSpec(csv_path=csv_path.name, symbol="TESTUSDT", interval="60"),
        features=("log_return@5", "realized_vol@20", "atr@14"),
        label=LabelSpec(horizon_bars=6, up_pct=0.005, down_pct=0.005),
        model=ModelSpec(n_estimators=50, num_leaves=7, min_data_in_leaf=10),
        strategy=StrategySpec(confidence_threshold=0.55, notional_fraction=0.3),
        wfo=WFOSpec(train_bars=400, test_bars=100, step_bars=100),
        backtest=BacktestSpec(initial_equity=1000.0),
        gates=RobustnessGateSpec(
            min_pct_folds_positive_cas=0.0,
            min_folds_with_trades=0,
            max_annualized_turnover=1e9,
        ),
    )

    result = run_experiment(
        spec,
        code_git_sha="deadbeef" * 5,
        lockfile_sha="cafef00d" * 8,
        data_root=tmp_path,
    )
    assert result.n_bars == 800
    assert result.robustness.n_folds >= 1
    # Result dict must survive a JSON roundtrip so the CLI can dump it.
    dumped = json.dumps(result.to_dict(), default=str)
    assert "spec" in dumped
    assert "robustness" in dumped
    assert result.spec_fingerprint == spec.fingerprint


def test_run_experiment_returns_error_result_when_corpus_too_short(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    _write_synthetic_csv(csv_path, n_bars=50)
    spec = ExperimentSpec(
        name="too_short",
        data=DataSpec(csv_path=csv_path.name, symbol="TESTUSDT"),
        features=("log_return@5",),
        wfo=WFOSpec(train_bars=100, test_bars=20, step_bars=20),
        gates=RobustnessGateSpec(min_pct_folds_positive_cas=0.0),
    )
    result = run_experiment(
        spec,
        code_git_sha="a" * 40,
        lockfile_sha="b" * 64,
        data_root=tmp_path,
    )
    assert result.error is not None
    assert result.robustness.n_folds == 0
    assert result.gate.passed is False
