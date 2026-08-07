"""Tests for leaderboard ranking and format."""

from __future__ import annotations

import json
from pathlib import Path

from trade.research.leaderboard import (
    LeaderboardRow,
    format_table,
    load_results_dir,
    rank,
)


def _row(
    *,
    name: str,
    passed: bool,
    consistency: float,
    mean_cas: float = 0.0,
    dd: float = 5.0,
    ann_tv: float = 5.0,
    reasons: tuple[str, ...] = (),
) -> LeaderboardRow:
    return LeaderboardRow(
        name=name,
        fingerprint="deadbeef",
        passed=passed,
        consistency_score=consistency,
        mean_cost_adjusted_sharpe=mean_cas,
        pct_folds_positive_cas=0.6,
        max_fold_drawdown_pct=dd,
        annualized_turnover=ann_tv,
        n_folds=6,
        n_folds_with_trades=6,
        reasons_failed=reasons,
    )


def test_passed_experiments_always_rank_above_failed() -> None:
    high_but_failed = _row(name="lucky_failed", passed=False, consistency=99.0)
    mid_passed = _row(name="steady_passed", passed=True, consistency=0.5)
    ordered = rank([high_but_failed, mid_passed])
    assert ordered[0].name == "steady_passed"
    assert ordered[1].name == "lucky_failed"


def test_within_passed_group_higher_consistency_wins() -> None:
    good = _row(name="good", passed=True, consistency=1.5)
    better = _row(name="better", passed=True, consistency=2.5)
    ordered = rank([good, better])
    assert [r.name for r in ordered] == ["better", "good"]


def test_ties_broken_by_mean_cas_then_dd_then_turnover() -> None:
    a = _row(name="a", passed=True, consistency=1.0, mean_cas=0.5, dd=5.0, ann_tv=5.0)
    b = _row(name="b", passed=True, consistency=1.0, mean_cas=0.7, dd=5.0, ann_tv=5.0)
    ordered = rank([a, b])
    assert ordered[0].name == "b"


def test_load_results_dir_ignores_non_result_json(tmp_path: Path) -> None:
    (tmp_path / "not_a_result.json").write_text(json.dumps({"foo": "bar"}))
    (tmp_path / "bad.json").write_text("this isn't json")
    rows = load_results_dir(tmp_path)
    assert rows == []


def test_load_results_dir_reads_valid_result(tmp_path: Path) -> None:
    result = {
        "spec": {
            "name": "sample",
            "data": {"csv_path": "x.csv", "symbol": "BTCUSDT", "interval": "60"},
            "features": ["log_return@5"],
            "label": {
                "kind": "triple_barrier",
                "horizon_bars": 6,
                "up_pct": 0.01,
                "down_pct": 0.01,
            },
            "model": {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "num_leaves": 15,
                "min_data_in_leaf": 5,
                "max_depth": -1,
                "calibration_fraction": 0.2,
            },
            "strategy": {
                "confidence_threshold": 0.55,
                "notional_fraction": 0.5,
                "allow_short": True,
            },
            "wfo": {"train_bars": 100, "test_bars": 20, "step_bars": 20, "expanding": False},
            "backtest": {
                "initial_equity": 10000.0,
                "fee_bps": 5.5,
                "slippage_bps": 5.0,
                "bars_per_year": 8760,
            },
            "gates": {
                "max_fold_drawdown_pct": 15.0,
                "min_pct_folds_positive_cas": 0.5,
                "min_fills_per_fold": 0,
                "min_folds_with_trades": 1,
                "max_annualized_turnover": 50.0,
            },
        },
        "spec_fingerprint": "abc123",
        "robustness": {
            "n_folds": 6,
            "n_folds_with_trades": 5,
            "pct_folds_positive_cas": 0.7,
            "mean_cost_adjusted_sharpe": 1.2,
            "median_cost_adjusted_sharpe": 1.1,
            "min_cost_adjusted_sharpe": -0.3,
            "std_cost_adjusted_sharpe": 0.4,
            "max_fold_drawdown_pct": 4.5,
            "mean_hit_rate": 0.55,
            "total_fills": 33,
            "mean_turnover_per_fold": 1.5,
            "annualized_turnover": 9.0,
            "consistency_score": 0.44,
        },
        "gate": {"passed": True, "reasons_failed": []},
    }
    (tmp_path / "sample.json").write_text(json.dumps(result))
    rows = load_results_dir(tmp_path)
    assert len(rows) == 1
    assert rows[0].name == "sample"
    assert rows[0].passed is True
    assert rows[0].consistency_score == 0.44


def test_format_table_returns_ranked_lines() -> None:
    rows = [
        _row(name="second", passed=True, consistency=0.5),
        _row(name="first", passed=True, consistency=2.5),
    ]
    text = format_table(rows)
    lines = text.splitlines()
    # Third line onwards are ranked rows; the top row should mention 'first'.
    assert "first" in lines[2]
    assert "second" in lines[3]
