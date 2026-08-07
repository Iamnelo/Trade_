"""Rank a collection of ExperimentResults.

Ordering:

1. `gate.passed=True` sorts strictly before `gate.passed=False`.
2. Within each group, higher `consistency_score` is better.
3. Ties broken by higher `mean_cost_adjusted_sharpe`, then lower
   `max_fold_drawdown_pct`, then lower `annualized_turnover`.

`format_table` returns a fixed-width text table suitable for CLI dump.
`load_results_dir(dir)` reads every `*.json` in the directory that
parses as an `ExperimentResult`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trade.research.experiment import ExperimentSpec
from trade.research.robustness import GateResult, RobustnessMetrics


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    name: str
    fingerprint: str
    passed: bool
    consistency_score: float
    mean_cost_adjusted_sharpe: float
    pct_folds_positive_cas: float
    max_fold_drawdown_pct: float
    annualized_turnover: float
    n_folds: int
    n_folds_with_trades: int
    reasons_failed: tuple[str, ...]
    source_path: Path | None = None


def _sort_key(row: LeaderboardRow) -> tuple[int, float, float, float, float]:
    return (
        0 if row.passed else 1,
        -row.consistency_score,
        -row.mean_cost_adjusted_sharpe,
        row.max_fold_drawdown_pct,
        row.annualized_turnover,
    )


def rank(rows: Iterable[LeaderboardRow]) -> list[LeaderboardRow]:
    return sorted(rows, key=_sort_key)


def row_from_result_dict(d: dict[str, Any], *, source: Path | None = None) -> LeaderboardRow:
    spec = ExperimentSpec.from_dict(d["spec"])
    rb: dict[str, Any] = d["robustness"]
    gate: dict[str, Any] = d["gate"]
    return LeaderboardRow(
        name=spec.name,
        fingerprint=d["spec_fingerprint"],
        passed=bool(gate["passed"]),
        consistency_score=float(rb["consistency_score"]),
        mean_cost_adjusted_sharpe=float(rb["mean_cost_adjusted_sharpe"]),
        pct_folds_positive_cas=float(rb["pct_folds_positive_cas"]),
        max_fold_drawdown_pct=float(rb["max_fold_drawdown_pct"]),
        annualized_turnover=float(rb["annualized_turnover"]),
        n_folds=int(rb["n_folds"]),
        n_folds_with_trades=int(rb["n_folds_with_trades"]),
        reasons_failed=tuple(gate.get("reasons_failed", [])),
        source_path=source,
    )


def row_from_result(
    *,
    spec: ExperimentSpec,
    spec_fingerprint: str,
    robustness: RobustnessMetrics,
    gate: GateResult,
    source: Path | None = None,
) -> LeaderboardRow:
    return LeaderboardRow(
        name=spec.name,
        fingerprint=spec_fingerprint,
        passed=gate.passed,
        consistency_score=robustness.consistency_score,
        mean_cost_adjusted_sharpe=robustness.mean_cost_adjusted_sharpe,
        pct_folds_positive_cas=robustness.pct_folds_positive_cas,
        max_fold_drawdown_pct=robustness.max_fold_drawdown_pct,
        annualized_turnover=robustness.annualized_turnover,
        n_folds=robustness.n_folds,
        n_folds_with_trades=robustness.n_folds_with_trades,
        reasons_failed=gate.reasons_failed,
        source_path=source,
    )


def load_results_dir(directory: Path) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            if "spec" not in data or "robustness" not in data:
                continue
            rows.append(row_from_result_dict(data, source=path))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return rows


def format_table(rows: Sequence[LeaderboardRow], *, top_n: int | None = None) -> str:
    ranked = rank(rows)
    if top_n is not None:
        ranked = ranked[:top_n]
    header = (
        f"{'rank':<5} {'pass':<5} {'name':<32} "
        f"{'cons':>8} {'mean_cas':>10} {'pct_pos':>8} {'max_dd%':>9} "
        f"{'ann_tv':>8} {'folds':>6}"
    )
    lines = [header, "-" * len(header)]
    for i, row in enumerate(ranked):
        lines.append(
            f"{i + 1:<5} {'PASS' if row.passed else 'FAIL':<5} "
            f"{row.name[:32]:<32} "
            f"{row.consistency_score:>8.3f} "
            f"{row.mean_cost_adjusted_sharpe:>10.3f} "
            f"{row.pct_folds_positive_cas:>8.2f} "
            f"{row.max_fold_drawdown_pct:>9.2f} "
            f"{row.annualized_turnover:>8.2f} "
            f"{row.n_folds_with_trades}/{row.n_folds:<3}"
        )
        if not row.passed and row.reasons_failed:
            lines.append(f"       └─ failed: {'; '.join(row.reasons_failed)}")
    return "\n".join(lines)
