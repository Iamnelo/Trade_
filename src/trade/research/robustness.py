"""Cross-fold robustness scoring and hard-fail gates.

Given the per-fold `PerformanceReport`s from a walk-forward run, this
module distills them into a single `RobustnessMetrics` bundle that the
leaderboard can rank against, and evaluates the experiment's
`RobustnessGateSpec` — the pass/fail rules that keep the experiment
from being shortlisted at all if a fold blew up or nothing traded.

The `consistency_score` is the primary composite: it rewards a high
mean cost-adjusted Sharpe when a MAJORITY of folds are positive, and
penalises variability across folds. Concretely:

    consistency_score = mean_cas * pct_positive_folds - std_cas

so a strategy that averages Sharpe 2 on one fold and -2 on the next
(mean=0) scores much worse than one that hovers around 0.8 on every
fold — exactly the "avoid a single lucky backtest" property we want.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from trade.metrics.performance import PerformanceReport
from trade.research.experiment import RobustnessGateSpec


@dataclass(frozen=True, slots=True)
class RobustnessMetrics:
    n_folds: int
    n_folds_with_trades: int
    pct_folds_positive_cas: float
    mean_cost_adjusted_sharpe: float
    median_cost_adjusted_sharpe: float
    min_cost_adjusted_sharpe: float
    std_cost_adjusted_sharpe: float
    max_fold_drawdown_pct: float
    mean_hit_rate: float
    total_fills: int
    mean_turnover_per_fold: float
    annualized_turnover: float
    consistency_score: float


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reasons_failed: tuple[str, ...]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _stdev(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def compute_robustness(
    fold_reports: Sequence[PerformanceReport],
    *,
    bars_per_year: int,
    test_bars_per_fold: int,
) -> RobustnessMetrics:
    if not fold_reports:
        return RobustnessMetrics(
            n_folds=0,
            n_folds_with_trades=0,
            pct_folds_positive_cas=0.0,
            mean_cost_adjusted_sharpe=0.0,
            median_cost_adjusted_sharpe=0.0,
            min_cost_adjusted_sharpe=0.0,
            std_cost_adjusted_sharpe=0.0,
            max_fold_drawdown_pct=0.0,
            mean_hit_rate=0.0,
            total_fills=0,
            mean_turnover_per_fold=0.0,
            annualized_turnover=0.0,
            consistency_score=0.0,
        )

    cas = [r.cost_adjusted_sharpe for r in fold_reports]
    dds = [r.max_drawdown_pct for r in fold_reports]
    hits = [r.hit_rate for r in fold_reports if r.n_fills > 0]
    turnovers = [r.turnover for r in fold_reports]
    n_folds = len(fold_reports)
    n_folds_with_trades = sum(1 for r in fold_reports if r.n_fills > 0)
    pct_positive = sum(1 for x in cas if x > 0) / n_folds
    mean_cas = _mean(cas)
    std_cas = _stdev(cas)

    # Annualise mean turnover: turnover is per-fold; folds cover test_bars_per_fold
    # bars each, so scale to a year.
    mean_turnover_per_fold = _mean(turnovers)
    year_scale = bars_per_year / test_bars_per_fold if test_bars_per_fold > 0 else 0.0
    annual_turnover = mean_turnover_per_fold * year_scale

    consistency_score = mean_cas * pct_positive - std_cas

    return RobustnessMetrics(
        n_folds=n_folds,
        n_folds_with_trades=n_folds_with_trades,
        pct_folds_positive_cas=pct_positive,
        mean_cost_adjusted_sharpe=mean_cas,
        median_cost_adjusted_sharpe=_median(cas),
        min_cost_adjusted_sharpe=min(cas),
        std_cost_adjusted_sharpe=std_cas,
        max_fold_drawdown_pct=max(dds),
        mean_hit_rate=_mean(hits),
        total_fills=sum(r.n_fills for r in fold_reports),
        mean_turnover_per_fold=mean_turnover_per_fold,
        annualized_turnover=annual_turnover,
        consistency_score=consistency_score,
    )


def evaluate_gates(
    metrics: RobustnessMetrics,
    *,
    gates: RobustnessGateSpec,
    fold_reports: Sequence[PerformanceReport],
) -> GateResult:
    reasons: list[str] = []

    if metrics.max_fold_drawdown_pct > gates.max_fold_drawdown_pct:
        reasons.append(
            f"max_fold_drawdown_pct={metrics.max_fold_drawdown_pct:.2f} "
            f"> {gates.max_fold_drawdown_pct:.2f}"
        )
    if metrics.pct_folds_positive_cas < gates.min_pct_folds_positive_cas:
        reasons.append(
            f"pct_folds_positive_cas={metrics.pct_folds_positive_cas:.2f} "
            f"< {gates.min_pct_folds_positive_cas:.2f}"
        )
    if metrics.n_folds_with_trades < gates.min_folds_with_trades:
        reasons.append(
            f"n_folds_with_trades={metrics.n_folds_with_trades} < {gates.min_folds_with_trades}"
        )
    if metrics.annualized_turnover > gates.max_annualized_turnover:
        reasons.append(
            f"annualized_turnover={metrics.annualized_turnover:.2f} "
            f"> {gates.max_annualized_turnover:.2f}"
        )
    if gates.min_fills_per_fold > 0:
        for i, r in enumerate(fold_reports):
            if r.n_fills > 0 and r.n_fills < gates.min_fills_per_fold:
                reasons.append(f"fold {i} n_fills={r.n_fills} < {gates.min_fills_per_fold}")
                break

    return GateResult(passed=not reasons, reasons_failed=tuple(reasons))
