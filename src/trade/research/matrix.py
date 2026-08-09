"""Cartesian-product grid runner.

A `MatrixSpec` bundles a base `ExperimentSpec` with a `grid` — a mapping
from dotted paths (e.g. `label.mode`) to lists of override values. The
expander returns one concrete `ExperimentSpec` per cell of the Cartesian
product, with a derived name and independent fingerprint.

The runner is intentionally naive about anti-leakage: it never inspects
per-cell test results to select follow-up cells. Every spec is chosen
ahead of time from the declarative `grid`. Threshold and other decision
knobs are held fixed by the base spec — the caller is responsible for
not letting a "best across cells" pick leak backward into future
matrices.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade.research.experiment import ExperimentSpec
from trade.research.robustness import GateResult, RobustnessMetrics
from trade.research.runner import ExperimentResult, run_experiment

# Nested dataclass paths supported by the grid — everything else is
# rejected up front so a typo does not silently produce N identical cells.
_ALLOWED_SECTIONS = {"label", "model", "strategy", "wfo", "backtest", "gates", "data"}
_ALLOWED_TOPLEVEL = {"features"}
_CELL_NAME_SAFE = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class MatrixSpec:
    name: str
    base_spec: ExperimentSpec
    grid: dict[str, tuple[Any, ...]]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("matrix name must be non-empty")
        if not self.grid:
            raise ValueError("matrix grid must be non-empty")
        for path, values in self.grid.items():
            if not values:
                raise ValueError(f"grid path {path!r} has no values")
            head = path.split(".", 1)[0]
            if head not in _ALLOWED_SECTIONS and head not in _ALLOWED_TOPLEVEL:
                raise ValueError(
                    f"grid path {path!r} targets unknown section {head!r}; "
                    f"allowed: {_ALLOWED_SECTIONS | _ALLOWED_TOPLEVEL}"
                )

    @property
    def n_cells(self) -> int:
        n = 1
        for values in self.grid.values():
            n *= len(values)
        return n

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_spec": self.base_spec.to_dict(),
            "grid": {k: list(v) for k, v in self.grid.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MatrixSpec:
        return cls(
            name=d["name"],
            base_spec=ExperimentSpec.from_dict(d["base_spec"]),
            grid={k: tuple(v) for k, v in d["grid"].items()},
        )

    @classmethod
    def from_json(cls, text: str) -> MatrixSpec:
        return cls.from_dict(json.loads(text))


def _apply_override(spec: ExperimentSpec, path: str, value: Any) -> ExperimentSpec:
    """Return a copy of `spec` with the field at `path` set to `value`.

    Supported paths:
      - "features" (top-level) — value must be a sequence of feature ids.
      - "<section>.<field>" where section ∈ {data,label,model,strategy,
        wfo,backtest,gates}.
    """
    if path == "features":
        return dataclasses.replace(spec, features=tuple(value))
    section, field_name = path.split(".", 1)
    section_obj = getattr(spec, section)
    new_section = dataclasses.replace(section_obj, **{field_name: value})
    return dataclasses.replace(spec, **{section: new_section})


def _slug_for_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        s = repr(v)
        return _CELL_NAME_SAFE.sub("_", s).strip("_")
    if isinstance(v, str):
        return _CELL_NAME_SAFE.sub("_", v).strip("_")[:32]
    if isinstance(v, (list, tuple)):
        return f"n{len(v)}"
    return _CELL_NAME_SAFE.sub("_", str(v)).strip("_")[:32]


def _cell_name(base_name: str, overrides: dict[str, Any]) -> str:
    parts = [base_name]
    for path, value in overrides.items():
        short = path.split(".")[-1]
        parts.append(f"{short}={_slug_for_value(value)}")
    return "__".join(parts)


def expand_matrix(matrix: MatrixSpec) -> list[ExperimentSpec]:
    paths = list(matrix.grid.keys())
    value_lists = [list(matrix.grid[p]) for p in paths]
    out: list[ExperimentSpec] = []
    for combo in itertools.product(*value_lists):
        overrides = dict(zip(paths, combo, strict=True))
        spec = matrix.base_spec
        for path, value in overrides.items():
            spec = _apply_override(spec, path, value)
        spec = dataclasses.replace(spec, name=_cell_name(matrix.base_spec.name, overrides))
        out.append(spec)
    return out


@dataclass(frozen=True, slots=True)
class MatrixCellRow:
    cell_index: int
    name: str
    fingerprint: str
    passed: bool
    mean_cost_adjusted_sharpe: float
    consistency_score: float
    pct_folds_positive_cas: float
    max_fold_drawdown_pct: float
    annualized_turnover: float
    total_fills: int
    reasons_failed: tuple[str, ...]
    spec: ExperimentSpec
    result: ExperimentResult


@dataclass(frozen=True, slots=True)
class MatrixReport:
    name: str
    generated_at: str
    n_cells: int
    rows: tuple[MatrixCellRow, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generated_at": self.generated_at,
            "n_cells": self.n_cells,
            "rows": [
                {
                    "cell_index": r.cell_index,
                    "name": r.name,
                    "fingerprint": r.fingerprint,
                    "passed": r.passed,
                    "mean_cost_adjusted_sharpe": r.mean_cost_adjusted_sharpe,
                    "consistency_score": r.consistency_score,
                    "pct_folds_positive_cas": r.pct_folds_positive_cas,
                    "max_fold_drawdown_pct": r.max_fold_drawdown_pct,
                    "annualized_turnover": r.annualized_turnover,
                    "total_fills": r.total_fills,
                    "reasons_failed": list(r.reasons_failed),
                    "spec": r.spec.to_dict(),
                    "result": r.result.to_dict(),
                }
                for r in self.rows
            ],
        }


def _row_from_result(
    *, cell_index: int, spec: ExperimentSpec, result: ExperimentResult
) -> MatrixCellRow:
    return MatrixCellRow(
        cell_index=cell_index,
        name=spec.name,
        fingerprint=spec.fingerprint,
        passed=result.gate.passed,
        mean_cost_adjusted_sharpe=result.robustness.mean_cost_adjusted_sharpe,
        consistency_score=result.robustness.consistency_score,
        pct_folds_positive_cas=result.robustness.pct_folds_positive_cas,
        max_fold_drawdown_pct=result.robustness.max_fold_drawdown_pct,
        annualized_turnover=result.robustness.annualized_turnover,
        total_fills=result.robustness.total_fills,
        reasons_failed=result.gate.reasons_failed,
        spec=spec,
        result=result,
    )


def _rank_key(row: MatrixCellRow) -> tuple[int, float, float, float, float]:
    return (
        0 if row.passed else 1,
        -row.consistency_score,
        -row.mean_cost_adjusted_sharpe,
        row.max_fold_drawdown_pct,
        row.annualized_turnover,
    )


def rank_matrix(rows: Sequence[MatrixCellRow]) -> list[MatrixCellRow]:
    return sorted(rows, key=_rank_key)


def _placeholder_result(spec: ExperimentSpec, message: str) -> ExperimentResult:
    """Build a stub ExperimentResult so a failed cell still lands in the leaderboard."""
    rb = RobustnessMetrics(
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
    return ExperimentResult(
        spec=spec,
        spec_fingerprint=spec.fingerprint,
        generated_at=datetime.now(tz=UTC).isoformat(),
        code_git_sha="",
        lockfile_sha="",
        n_bars=0,
        folds=(),
        robustness=rb,
        gate=GateResult(passed=False, reasons_failed=(f"crashed: {message[:200]}",)),
        error=message,
    )


def run_matrix(
    matrix: MatrixSpec,
    *,
    code_git_sha: str,
    lockfile_sha: str,
    data_root: Path,
    on_progress: object | None = None,
) -> MatrixReport:
    specs = expand_matrix(matrix)
    rows: list[MatrixCellRow] = []
    for i, spec in enumerate(specs):
        if on_progress is not None:
            on_progress(i, len(specs), spec.name)  # type: ignore[operator]
        try:
            result = run_experiment(
                spec,
                code_git_sha=code_git_sha,
                lockfile_sha=lockfile_sha,
                data_root=data_root,
            )
        except Exception as exc:
            result = _placeholder_result(spec, f"{type(exc).__name__}: {exc}")
        rows.append(_row_from_result(cell_index=i, spec=spec, result=result))
    return MatrixReport(
        name=matrix.name,
        generated_at=datetime.now(tz=UTC).isoformat(),
        n_cells=len(specs),
        rows=tuple(rows),
    )


def format_matrix_table(report: MatrixReport, *, top_n: int | None = None) -> str:
    ranked = rank_matrix(report.rows)
    if top_n is not None:
        ranked = ranked[:top_n]
    header = (
        f"{'rk':<3} {'pass':<5} {'cell':<52} "
        f"{'mean_cas':>9} {'cons':>8} {'pct_pos':>8} {'max_dd%':>8} {'ann_tv':>8}"
    )
    lines = [f"# matrix: {report.name}  ({report.n_cells} cells)", "", header, "-" * len(header)]
    for i, row in enumerate(ranked):
        lines.append(
            f"{i + 1:<3} {'PASS' if row.passed else 'FAIL':<5} "
            f"{row.name[:52]:<52} "
            f"{row.mean_cost_adjusted_sharpe:>9.3f} "
            f"{row.consistency_score:>8.3f} "
            f"{row.pct_folds_positive_cas:>8.2f} "
            f"{row.max_fold_drawdown_pct:>8.2f} "
            f"{row.annualized_turnover:>8.2f}"
        )
    return "\n".join(lines)
