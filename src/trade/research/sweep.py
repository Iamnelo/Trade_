"""Confidence-threshold sweep on top of a base ExperimentSpec.

Given a base ExperimentSpec and a list of thresholds, `run_threshold_sweep`
produces one `ThresholdSweepResult` combining:

- Per-fold classifier diagnostics (AUC-ROC / AUC-PR / ECE) — properties
  of the trained model only, invariant across thresholds.
- Per-fold Oracle capture ratio at the SPEC's original threshold — kept
  as a reference point so you can see what fraction of achievable P&L
  each fold has room to reach.
- Per-(fold, threshold) trading metrics: cost_adj_sharpe, sharpe,
  sortino, max_dd, turnover, n_fills, hit_rate, expectancy, win_rate,
  profit_factor, avg_win, avg_loss, expected_return_per_trade,
  oracle_capture_long_short.
- Per-threshold aggregate: robustness metrics + gate result + a
  human-readable recommendation.

The recommendation criterion (per user spec): NOT the highest raw
return. Instead the threshold that maximises consistency_score and
passes gates; among passers, ties broken by lowest max DD.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade.features.catalog import build_features
from trade.metrics.trade_metrics import TradeMetrics, compute_trade_metrics
from trade.mre.types import BacktestConfig
from trade.research.diagnostics import (
    ClassifierDiagnostics,
    OracleCapture,
    compute_classifier_diagnostics,
    compute_oracle_capture,
    score_holdout,
)
from trade.research.experiment import ExperimentSpec
from trade.research.robustness import (
    GateResult,
    RobustnessMetrics,
    compute_robustness,
    evaluate_gates,
)
from trade.research.runner import load_klines_csv
from trade.wfo.model_runner import run_walk_forward_threshold_sweep
from trade.wfo.splitter import walk_forward_splits


@dataclass(frozen=True, slots=True)
class FoldClassifierDiagnostics:
    fold_index: int
    reproducibility_hash: str
    diagnostics: ClassifierDiagnostics


@dataclass(frozen=True, slots=True)
class ThresholdFoldCell:
    """One (fold, threshold) cell of trading-side metrics."""

    fold_index: int
    threshold: float
    n_fills: int
    total_return_pct: float
    sharpe: float
    sortino: float
    cost_adjusted_sharpe: float
    max_drawdown_pct: float
    hit_rate: float
    turnover: float
    trade_metrics: TradeMetrics
    oracle_capture: OracleCapture


@dataclass(frozen=True, slots=True)
class ThresholdAggregate:
    """Per-threshold aggregate across all folds."""

    threshold: float
    robustness: RobustnessMetrics
    gate: GateResult


@dataclass(frozen=True, slots=True)
class ThresholdSweepResult:
    base_spec: ExperimentSpec
    thresholds: tuple[float, ...]
    generated_at: str
    code_git_sha: str
    lockfile_sha: str
    n_bars: int
    per_fold_classifier: tuple[FoldClassifierDiagnostics, ...]
    per_cell: tuple[ThresholdFoldCell, ...]
    per_threshold: tuple[ThresholdAggregate, ...]
    recommended_threshold: float | None
    recommendation_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_spec": self.base_spec.to_dict(),
            "thresholds": list(self.thresholds),
            "generated_at": self.generated_at,
            "code_git_sha": self.code_git_sha,
            "lockfile_sha": self.lockfile_sha,
            "n_bars": self.n_bars,
            "per_fold_classifier": [
                {
                    "fold_index": f.fold_index,
                    "reproducibility_hash": f.reproducibility_hash,
                    "diagnostics": asdict(f.diagnostics),
                }
                for f in self.per_fold_classifier
            ],
            "per_cell": [
                {
                    **{
                        k: v
                        for k, v in asdict(c).items()
                        if k not in {"trade_metrics", "oracle_capture"}
                    },
                    "trade_metrics": asdict(c.trade_metrics),
                    "oracle_capture": asdict(c.oracle_capture),
                }
                for c in self.per_cell
            ],
            "per_threshold": [
                {
                    "threshold": t.threshold,
                    "robustness": asdict(t.robustness),
                    "gate": {
                        "passed": t.gate.passed,
                        "reasons_failed": list(t.gate.reasons_failed),
                    },
                }
                for t in self.per_threshold
            ],
            "recommended_threshold": self.recommended_threshold,
            "recommendation_notes": self.recommendation_notes,
        }


def _pick_recommendation(aggregates: Sequence[ThresholdAggregate]) -> tuple[float | None, str]:
    """Recommend the threshold with best robustness AMONG passers.

    If none pass, recommend the highest-consistency non-passer as an
    "least-broken" reference — but call it out explicitly.
    """
    if not aggregates:
        return None, "no aggregates to rank"
    passers = [a for a in aggregates if a.gate.passed]
    if passers:
        best = max(
            passers,
            key=lambda a: (
                a.robustness.consistency_score,
                -a.robustness.max_fold_drawdown_pct,
                -a.robustness.annualized_turnover,
            ),
        )
        return (
            best.threshold,
            f"selected {best.threshold:.2f}: highest consistency among gate-passers "
            f"(cons={best.robustness.consistency_score:.3f}, "
            f"max_dd={best.robustness.max_fold_drawdown_pct:.2f}%, "
            f"ann_tv={best.robustness.annualized_turnover:.2f})",
        )
    best = max(aggregates, key=lambda a: a.robustness.consistency_score)
    return (
        None,
        f"NO threshold passed the robustness gates; least-broken was "
        f"{best.threshold:.2f} (cons={best.robustness.consistency_score:.3f}, "
        f"gate reasons: {'; '.join(best.gate.reasons_failed)})",
    )


def run_threshold_sweep(
    base_spec: ExperimentSpec,
    *,
    thresholds: Sequence[float],
    code_git_sha: str,
    lockfile_sha: str,
    data_root: Path,
) -> ThresholdSweepResult:
    csv_path = data_root / base_spec.data.csv_path
    bars = load_klines_csv(csv_path, symbol=base_spec.data.symbol, interval=base_spec.data.interval)
    if not bars:
        raise ValueError(f"no bars loaded from {csv_path}")
    features = build_features(list(base_spec.features))
    config = BacktestConfig(
        initial_equity=base_spec.backtest.initial_equity,
        fee_bps=base_spec.backtest.fee_bps,
        slippage_bps=base_spec.backtest.slippage_bps,
    )
    splits = walk_forward_splits(
        n_bars=len(bars),
        train_bars=base_spec.wfo.train_bars,
        test_bars=base_spec.wfo.test_bars,
        step_bars=base_spec.wfo.step_bars,
        expanding=base_spec.wfo.expanding,
    )
    dataset_id = f"csv:{base_spec.data.symbol}:{base_spec.data.interval}:{base_spec.data.csv_path}"
    sweep_report = run_walk_forward_threshold_sweep(
        bars=bars,
        symbol=base_spec.data.symbol,
        interval=base_spec.data.interval,
        features=features,
        label_horizon_bars=base_spec.label.horizon_bars,
        label_up_pct=base_spec.label.up_pct,
        label_down_pct=base_spec.label.down_pct,
        splits=splits,
        config=config,
        bars_per_year=base_spec.backtest.bars_per_year,
        dataset_manifest_ids=[dataset_id],
        feature_manifest_ids=list(base_spec.features),
        code_git_sha=code_git_sha,
        python_lockfile_sha=lockfile_sha,
        thresholds=list(thresholds),
        model_config=base_spec.model.to_lightgbm_config(),
        notional_fraction=base_spec.strategy.notional_fraction,
        allow_short=base_spec.strategy.allow_short,
        calibration_fraction=base_spec.model.calibration_fraction,
        cost_bps_per_side=base_spec.backtest.fee_bps,
    )

    per_fold_clf: list[FoldClassifierDiagnostics] = []
    per_cell: list[ThresholdFoldCell] = []

    for fold_idx, fold in enumerate(sweep_report.folds):
        probs, y_true = score_holdout(
            sorted_bars=fold.trained.sorted_bars,
            symbol=base_spec.data.symbol,
            features=list(fold.trained.features),
            model=fold.trained.artifacts.model,
            calibrator=fold.trained.artifacts.calibrator,
            test_start_idx=fold.split.test_start,
            test_end_idx=fold.split.test_end,
            label_horizon_bars=base_spec.label.horizon_bars,
            label_up_pct=base_spec.label.up_pct,
            label_down_pct=base_spec.label.down_pct,
        )
        per_fold_clf.append(
            FoldClassifierDiagnostics(
                fold_index=fold_idx,
                reproducibility_hash=fold.reproducibility_hash,
                diagnostics=compute_classifier_diagnostics(probs=probs, y_true_int=y_true),
            )
        )

        test_bars_slice = fold.trained.sorted_bars[fold.split.test_start : fold.split.test_end]
        for cell in fold.per_threshold:
            trade_stats = compute_trade_metrics(cell.result.backtest.fills)
            oracle = compute_oracle_capture(
                test_bars=test_bars_slice,
                strategy_final_equity=cell.result.backtest.final_equity,
                initial_equity=cell.result.backtest.initial_equity,
            )
            per_cell.append(
                ThresholdFoldCell(
                    fold_index=fold_idx,
                    threshold=cell.threshold,
                    n_fills=cell.result.report.n_fills,
                    total_return_pct=cell.result.report.total_return_pct,
                    sharpe=cell.result.report.sharpe,
                    sortino=cell.result.report.sortino,
                    cost_adjusted_sharpe=cell.result.report.cost_adjusted_sharpe,
                    max_drawdown_pct=cell.result.report.max_drawdown_pct,
                    hit_rate=cell.result.report.hit_rate,
                    turnover=cell.result.report.turnover,
                    trade_metrics=trade_stats,
                    oracle_capture=oracle,
                )
            )

    per_threshold: list[ThresholdAggregate] = []
    for t in sweep_report.thresholds:
        fold_reports = [
            fold.per_threshold[i].result.report
            for fold in sweep_report.folds
            for i, cell in enumerate(fold.per_threshold)
            if cell.threshold == t
        ]
        rb = compute_robustness(
            fold_reports,
            bars_per_year=base_spec.backtest.bars_per_year,
            test_bars_per_fold=base_spec.wfo.test_bars,
        )
        gate = evaluate_gates(rb, gates=base_spec.gates, fold_reports=fold_reports)
        per_threshold.append(ThresholdAggregate(threshold=t, robustness=rb, gate=gate))

    recommended, notes = _pick_recommendation(per_threshold)

    return ThresholdSweepResult(
        base_spec=base_spec,
        thresholds=sweep_report.thresholds,
        generated_at=datetime.now(tz=UTC).isoformat(),
        code_git_sha=code_git_sha,
        lockfile_sha=lockfile_sha,
        n_bars=len(bars),
        per_fold_classifier=tuple(per_fold_clf),
        per_cell=tuple(per_cell),
        per_threshold=tuple(per_threshold),
        recommended_threshold=recommended,
        recommendation_notes=notes,
    )
