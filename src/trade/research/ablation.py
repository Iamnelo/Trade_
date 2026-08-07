"""Forward-ablation runner.

An `AblationSpec` bundles one baseline `ExperimentSpec` together with a
list of `AblationVariant`s. Each variant declares which feature ids to
ADD to the baseline (never subtract — starting from the baseline
guarantees the delta is a pure marginal contribution of the added
family, not a re-ordering of a superset).

`run_ablation` executes the baseline plus every variant, then measures
each variant's ΔCAS, ΔConsistency, ΔMaxDD, ΔTurnover, and whether the
variant crosses a gate boundary (fail → pass) relative to the baseline.

Results serialise to a single JSON that the leaderboard can also
ingest — variants are stored as regular `ExperimentResult`s alongside
the delta metadata, so no other code needs to know about ablations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade.research.experiment import ExperimentSpec
from trade.research.robustness import RobustnessMetrics
from trade.research.runner import ExperimentResult, run_experiment


@dataclass(frozen=True, slots=True)
class AblationVariant:
    name: str
    add_features: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True, slots=True)
class AblationSpec:
    name: str
    base_spec: ExperimentSpec
    variants: tuple[AblationVariant, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ablation name must be non-empty")
        if not self.variants:
            raise ValueError("must supply at least one variant")
        seen: set[str] = set()
        for v in self.variants:
            if v.name in seen:
                raise ValueError(f"duplicate variant name {v.name!r}")
            seen.add(v.name)
            if not v.add_features:
                raise ValueError(f"variant {v.name!r} has empty add_features")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_spec": self.base_spec.to_dict(),
            "variants": [
                {
                    "name": v.name,
                    "add_features": list(v.add_features),
                    "description": v.description,
                }
                for v in self.variants
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AblationSpec:
        return cls(
            name=d["name"],
            base_spec=ExperimentSpec.from_dict(d["base_spec"]),
            variants=tuple(
                AblationVariant(
                    name=v["name"],
                    add_features=tuple(v["add_features"]),
                    description=v.get("description", ""),
                )
                for v in d["variants"]
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> AblationSpec:
        return cls.from_dict(json.loads(text))


def spec_with_added_features(
    base: ExperimentSpec, *, variant_name: str, add_features: Sequence[str]
) -> ExperimentSpec:
    """Return a new ExperimentSpec with `add_features` appended (deduplicated)."""
    existing = tuple(base.features)
    merged = list(existing)
    for fid in add_features:
        if fid not in merged:
            merged.append(fid)
    return ExperimentSpec(
        name=f"{base.name}__{variant_name}",
        data=base.data,
        features=tuple(merged),
        label=base.label,
        model=base.model,
        strategy=base.strategy,
        wfo=base.wfo,
        backtest=base.backtest,
        gates=base.gates,
    )


@dataclass(frozen=True, slots=True)
class DeltaMetrics:
    delta_mean_cost_adjusted_sharpe: float
    delta_consistency_score: float
    delta_pct_folds_positive_cas: float
    delta_max_fold_drawdown_pct: float
    delta_annualized_turnover: float
    baseline_gate_passed: bool
    variant_gate_passed: bool
    crossed_gate_boundary: bool  # baseline failed AND variant passed (win) — or opposite (loss)


def _delta(
    base: RobustnessMetrics, other: RobustnessMetrics
) -> tuple[float, float, float, float, float]:
    return (
        other.mean_cost_adjusted_sharpe - base.mean_cost_adjusted_sharpe,
        other.consistency_score - base.consistency_score,
        other.pct_folds_positive_cas - base.pct_folds_positive_cas,
        other.max_fold_drawdown_pct - base.max_fold_drawdown_pct,
        other.annualized_turnover - base.annualized_turnover,
    )


@dataclass(frozen=True, slots=True)
class VariantOutcome:
    variant: AblationVariant
    result: ExperimentResult
    delta: DeltaMetrics


@dataclass(frozen=True, slots=True)
class AblationReport:
    name: str
    generated_at: str
    baseline: ExperimentResult
    variants: tuple[VariantOutcome, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generated_at": self.generated_at,
            "baseline": self.baseline.to_dict(),
            "variants": [
                {
                    "variant": {
                        "name": v.variant.name,
                        "add_features": list(v.variant.add_features),
                        "description": v.variant.description,
                    },
                    "result": v.result.to_dict(),
                    "delta": asdict(v.delta),
                }
                for v in self.variants
            ],
        }


def run_ablation(
    spec: AblationSpec,
    *,
    code_git_sha: str,
    lockfile_sha: str,
    data_root: Path,
    on_progress: object | None = None,
) -> AblationReport:
    total_runs = 1 + len(spec.variants)
    if on_progress is not None:
        on_progress(0, total_runs, "baseline")  # type: ignore[operator]
    baseline = run_experiment(
        spec.base_spec, code_git_sha=code_git_sha, lockfile_sha=lockfile_sha, data_root=data_root
    )
    variants: list[VariantOutcome] = []
    for i, variant in enumerate(spec.variants):
        if on_progress is not None:
            on_progress(i + 1, total_runs, variant.name)  # type: ignore[operator]
        v_spec = spec_with_added_features(
            spec.base_spec, variant_name=variant.name, add_features=variant.add_features
        )
        v_result = run_experiment(
            v_spec, code_git_sha=code_git_sha, lockfile_sha=lockfile_sha, data_root=data_root
        )
        d_cas, d_cons, d_pct, d_dd, d_tv = _delta(baseline.robustness, v_result.robustness)
        crossed = baseline.gate.passed != v_result.gate.passed
        variants.append(
            VariantOutcome(
                variant=variant,
                result=v_result,
                delta=DeltaMetrics(
                    delta_mean_cost_adjusted_sharpe=d_cas,
                    delta_consistency_score=d_cons,
                    delta_pct_folds_positive_cas=d_pct,
                    delta_max_fold_drawdown_pct=d_dd,
                    delta_annualized_turnover=d_tv,
                    baseline_gate_passed=baseline.gate.passed,
                    variant_gate_passed=v_result.gate.passed,
                    crossed_gate_boundary=crossed,
                ),
            )
        )
    return AblationReport(
        name=spec.name,
        generated_at=datetime.now(tz=UTC).isoformat(),
        baseline=baseline,
        variants=tuple(variants),
    )


def format_ablation_table(report: AblationReport) -> str:
    header = (
        f"{'variant':<32} {'ΔCAS':>8} {'ΔCons':>8} {'ΔPct+':>7} {'ΔDD%':>7} {'ΔTV':>8} {'gate':>7}"
    )
    lines = [
        f"# ablation: {report.name}",
        f"# baseline: {report.baseline.spec.name}",
        f"#   cas={report.baseline.robustness.mean_cost_adjusted_sharpe:.3f}  "
        f"cons={report.baseline.robustness.consistency_score:.3f}  "
        f"pct+={report.baseline.robustness.pct_folds_positive_cas:.2f}  "
        f"gate={'PASS' if report.baseline.gate.passed else 'FAIL'}",
        "",
        header,
        "-" * len(header),
    ]
    # Order variants by ΔConsistency (best marginal first).
    ranked = sorted(report.variants, key=lambda v: v.delta.delta_consistency_score, reverse=True)
    for v in ranked:
        d = v.delta
        gate_str = ("PASS" if d.variant_gate_passed else "FAIL") + (
            "*" if d.crossed_gate_boundary else ""
        )
        lines.append(
            f"{v.variant.name[:32]:<32} "
            f"{d.delta_mean_cost_adjusted_sharpe:>8.3f} "
            f"{d.delta_consistency_score:>8.3f} "
            f"{d.delta_pct_folds_positive_cas:>7.2f} "
            f"{d.delta_max_fold_drawdown_pct:>7.2f} "
            f"{d.delta_annualized_turnover:>8.2f} "
            f"{gate_str:>7}"
        )
    return "\n".join(lines)
