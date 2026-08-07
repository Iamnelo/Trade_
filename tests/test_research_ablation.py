"""Tests for AblationSpec + spec-with-added-features + report shape.

The end-to-end ablation run is exercised by the runner-level tests
(via a small synthetic corpus) — here we focus on the wiring: spec
validation, JSON roundtrip, deduplicated feature merge, and delta math.
"""

from __future__ import annotations

import pytest

from trade.metrics.performance import PerformanceReport
from trade.research.ablation import (
    AblationReport,
    AblationSpec,
    AblationVariant,
    DeltaMetrics,
    VariantOutcome,
    format_ablation_table,
    spec_with_added_features,
)
from trade.research.experiment import DataSpec, ExperimentSpec, WFOSpec
from trade.research.robustness import (
    GateResult,
    RobustnessMetrics,
    compute_robustness,
)
from trade.research.runner import ExperimentResult


def _base_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="base",
        data=DataSpec(csv_path="x.csv", symbol="BTCUSDT"),
        features=("log_return@5", "realized_vol@20"),
        wfo=WFOSpec(train_bars=100, test_bars=20, step_bars=20),
    )


def test_ablation_spec_json_roundtrip() -> None:
    spec = AblationSpec(
        name="test_ablation",
        base_spec=_base_spec(),
        variants=(
            AblationVariant(name="add_time", add_features=("time_of_day@sin", "time_of_day@cos")),
            AblationVariant(name="add_vol", add_features=("realized_vol@50",)),
        ),
    )
    reloaded = AblationSpec.from_dict(spec.to_dict())
    assert reloaded == spec


def test_ablation_rejects_duplicate_variant_names() -> None:
    with pytest.raises(ValueError, match="duplicate variant name"):
        AblationSpec(
            name="dup",
            base_spec=_base_spec(),
            variants=(
                AblationVariant(name="v", add_features=("a@1",)),
                AblationVariant(name="v", add_features=("b@1",)),
            ),
        )


def test_ablation_rejects_empty_add_features() -> None:
    with pytest.raises(ValueError, match="empty add_features"):
        AblationSpec(
            name="empty",
            base_spec=_base_spec(),
            variants=(AblationVariant(name="v", add_features=()),),
        )


def test_ablation_rejects_empty_variants() -> None:
    with pytest.raises(ValueError, match="at least one variant"):
        AblationSpec(name="e", base_spec=_base_spec(), variants=())


def test_spec_with_added_features_appends_new_and_skips_duplicates() -> None:
    base = _base_spec()
    variant = spec_with_added_features(
        base,
        variant_name="mix",
        add_features=("log_return@5", "atr@14", "atr@14"),  # dupes + already-present
    )
    assert variant.features == ("log_return@5", "realized_vol@20", "atr@14")
    # Name is derived; other fields preserved.
    assert variant.name == "base__mix"
    assert variant.data == base.data
    assert variant.wfo == base.wfo


def test_spec_with_added_features_produces_new_fingerprint() -> None:
    base = _base_spec()
    variant = spec_with_added_features(base, variant_name="add_atr", add_features=("atr@14",))
    assert variant.fingerprint != base.fingerprint


def _make_result(cas_per_fold: list[float]) -> ExperimentResult:
    reports = [
        PerformanceReport(
            initial_equity=10_000.0,
            final_equity=10_000.0 * (1 + cas / 100.0),
            total_return_pct=cas,
            sharpe=cas,
            sortino=cas,
            calmar=cas,
            max_drawdown_pct=2.0,
            ulcer_index_pct=0.1,
            hit_rate=0.5,
            turnover=1.0,
            cvar_5pct=-0.001,
            cost_adjusted_sharpe=cas,
            n_bars=1440,
            n_fills=10,
        )
        for cas in cas_per_fold
    ]
    rb = compute_robustness(reports, bars_per_year=8760, test_bars_per_fold=1440)
    return ExperimentResult(
        spec=_base_spec(),
        spec_fingerprint="deadbeef",
        generated_at="2026-01-01T00:00:00+00:00",
        code_git_sha="a" * 40,
        lockfile_sha="b" * 64,
        n_bars=8000,
        folds=(),
        robustness=rb,
        gate=GateResult(passed=False, reasons_failed=("stub",)),
    )


def test_format_ablation_table_orders_by_delta_consistency() -> None:
    baseline = _make_result([0.0, 0.0, 0.0])
    variant_good = _make_result([1.0, 1.0, 1.0])
    variant_bad = _make_result([-1.0, -1.0, -1.0])

    def _delta_from(base: RobustnessMetrics, other: RobustnessMetrics) -> DeltaMetrics:
        return DeltaMetrics(
            delta_mean_cost_adjusted_sharpe=other.mean_cost_adjusted_sharpe
            - base.mean_cost_adjusted_sharpe,
            delta_consistency_score=other.consistency_score - base.consistency_score,
            delta_pct_folds_positive_cas=other.pct_folds_positive_cas - base.pct_folds_positive_cas,
            delta_max_fold_drawdown_pct=other.max_fold_drawdown_pct - base.max_fold_drawdown_pct,
            delta_annualized_turnover=other.annualized_turnover - base.annualized_turnover,
            baseline_gate_passed=False,
            variant_gate_passed=False,
            crossed_gate_boundary=False,
        )

    report = AblationReport(
        name="a",
        generated_at="2026-01-01T00:00:00+00:00",
        baseline=baseline,
        variants=(
            VariantOutcome(
                variant=AblationVariant(name="bad", add_features=("x@1",)),
                result=variant_bad,
                delta=_delta_from(baseline.robustness, variant_bad.robustness),
            ),
            VariantOutcome(
                variant=AblationVariant(name="good", add_features=("y@1",)),
                result=variant_good,
                delta=_delta_from(baseline.robustness, variant_good.robustness),
            ),
        ),
    )
    text = format_ablation_table(report)
    # 'good' appears before 'bad' because it wins on Δconsistency.
    good_line = next(i for i, line in enumerate(text.splitlines()) if "good" in line)
    bad_line = next(i for i, line in enumerate(text.splitlines()) if "bad" in line)
    assert good_line < bad_line
