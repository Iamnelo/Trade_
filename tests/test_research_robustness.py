"""Tests for RobustnessMetrics + gate evaluation."""

from __future__ import annotations

import pytest

from trade.metrics.performance import PerformanceReport
from trade.research.experiment import RobustnessGateSpec
from trade.research.robustness import compute_robustness, evaluate_gates


def _report(
    *,
    cas: float,
    dd: float = 2.0,
    turnover: float = 1.0,
    hit_rate: float = 0.5,
    n_fills: int = 10,
) -> PerformanceReport:
    return PerformanceReport(
        initial_equity=10_000.0,
        final_equity=10_000.0 * (1 + cas / 100.0),
        total_return_pct=cas,
        sharpe=cas,
        sortino=cas,
        calmar=cas,
        max_drawdown_pct=dd,
        ulcer_index_pct=0.1,
        hit_rate=hit_rate,
        turnover=turnover,
        cvar_5pct=-0.001,
        cost_adjusted_sharpe=cas,
        n_bars=1440,
        n_fills=n_fills,
    )


def test_empty_reports_produce_zero_metrics() -> None:
    m = compute_robustness([], bars_per_year=8760, test_bars_per_fold=1440)
    assert m.n_folds == 0
    assert m.consistency_score == 0.0


def test_all_positive_folds_gives_positive_consistency() -> None:
    reports = [_report(cas=1.0), _report(cas=1.2), _report(cas=0.9)]
    m = compute_robustness(reports, bars_per_year=8760, test_bars_per_fold=1440)
    assert m.pct_folds_positive_cas == 1.0
    assert m.mean_cost_adjusted_sharpe == pytest.approx((1.0 + 1.2 + 0.9) / 3)
    assert m.consistency_score > 0.0


def test_high_variance_hurts_consistency_more_than_mean_helps() -> None:
    stable = [_report(cas=0.8) for _ in range(4)]
    lucky = [_report(cas=3.0), _report(cas=-2.0), _report(cas=-2.0), _report(cas=3.0)]
    m_stable = compute_robustness(stable, bars_per_year=8760, test_bars_per_fold=1440)
    m_lucky = compute_robustness(lucky, bars_per_year=8760, test_bars_per_fold=1440)
    # Both have similar mean CAS, but 'stable' should score much higher.
    assert m_stable.consistency_score > m_lucky.consistency_score


def test_median_and_min_computed_correctly() -> None:
    reports = [_report(cas=-1.0), _report(cas=0.0), _report(cas=2.0)]
    m = compute_robustness(reports, bars_per_year=8760, test_bars_per_fold=1440)
    assert m.median_cost_adjusted_sharpe == 0.0
    assert m.min_cost_adjusted_sharpe == -1.0


def test_annualized_turnover_scales_correctly() -> None:
    # 6 folds x 1440 test_bars = 1 year of hourly bars. Mean turnover 1.0/fold
    # -> annualised turnover = 1.0 * (8760/1440) = 6.083...
    reports = [_report(cas=0.5, turnover=1.0) for _ in range(6)]
    m = compute_robustness(reports, bars_per_year=8760, test_bars_per_fold=1440)
    assert m.mean_turnover_per_fold == pytest.approx(1.0)
    assert m.annualized_turnover == pytest.approx(8760 / 1440)


def test_gate_fails_on_max_drawdown_breach() -> None:
    reports = [_report(cas=1.0, dd=25.0)]
    m = compute_robustness(reports, bars_per_year=8760, test_bars_per_fold=1440)
    gate = evaluate_gates(
        m,
        gates=RobustnessGateSpec(max_fold_drawdown_pct=15.0, min_pct_folds_positive_cas=0.0),
        fold_reports=reports,
    )
    assert gate.passed is False
    assert any("max_fold_drawdown_pct" in r for r in gate.reasons_failed)


def test_gate_fails_when_too_few_folds_positive() -> None:
    reports = [_report(cas=-1.0), _report(cas=-1.0), _report(cas=1.0)]
    m = compute_robustness(reports, bars_per_year=8760, test_bars_per_fold=1440)
    gate = evaluate_gates(
        m,
        gates=RobustnessGateSpec(min_pct_folds_positive_cas=0.5),
        fold_reports=reports,
    )
    assert gate.passed is False
    assert any("pct_folds_positive_cas" in r for r in gate.reasons_failed)


def test_gate_fails_when_too_many_folds_had_no_trades() -> None:
    reports = [_report(cas=0.0, n_fills=0) for _ in range(6)]
    m = compute_robustness(reports, bars_per_year=8760, test_bars_per_fold=1440)
    gate = evaluate_gates(
        m,
        gates=RobustnessGateSpec(min_folds_with_trades=3, min_pct_folds_positive_cas=0.0),
        fold_reports=reports,
    )
    assert gate.passed is False
    assert any("n_folds_with_trades" in r for r in gate.reasons_failed)


def test_gate_fails_on_excessive_annualized_turnover() -> None:
    reports = [_report(cas=0.5, turnover=20.0) for _ in range(6)]
    m = compute_robustness(reports, bars_per_year=8760, test_bars_per_fold=1440)
    gate = evaluate_gates(
        m,
        gates=RobustnessGateSpec(max_annualized_turnover=50.0, min_pct_folds_positive_cas=0.0),
        fold_reports=reports,
    )
    assert gate.passed is False
    assert any("annualized_turnover" in r for r in gate.reasons_failed)


def test_gate_passes_when_all_criteria_met() -> None:
    reports = [_report(cas=0.8, dd=3.0, turnover=1.0) for _ in range(6)]
    m = compute_robustness(reports, bars_per_year=8760, test_bars_per_fold=1440)
    gate = evaluate_gates(m, gates=RobustnessGateSpec(), fold_reports=reports)
    assert gate.passed is True
    assert gate.reasons_failed == ()
