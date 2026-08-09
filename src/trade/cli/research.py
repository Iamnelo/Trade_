"""Research CLI: run one experiment, run an ablation, rank a directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer

from trade.reproducibility.git import current_git_sha, lockfile_sha
from trade.research.ablation import AblationSpec, format_ablation_table, run_ablation
from trade.research.experiment import ExperimentSpec
from trade.research.leaderboard import format_table, load_results_dir
from trade.research.matrix import (
    MatrixSpec,
    format_matrix_table,
    rank_matrix,
    run_matrix,
)
from trade.research.runner import run_experiment
from trade.research.sweep import run_threshold_sweep
from trade.research.sweep_reports import (
    write_cell_csv,
    write_classifier_csv,
    write_html_report,
    write_markdown_summary,
    write_threshold_csv,
)

research_app = typer.Typer(no_args_is_help=True)


@research_app.command("run")
def run(
    spec_path: Path = typer.Option(..., help="Path to an ExperimentSpec JSON file."),
    out_dir: Path = typer.Option(
        Path("eval_reports/experiments"),
        help="Directory to write the ExperimentResult JSON into.",
    ),
    data_root: Path = typer.Option(
        Path.cwd(), help="Root against which spec.data.csv_path resolves."
    ),
    lockfile: Path | None = typer.Option(
        None, help="Path to a Python lockfile; sha256 feeds the reproducibility hash."
    ),
) -> None:
    """Run one experiment end-to-end and dump the result JSON."""
    spec = ExperimentSpec.from_json(spec_path.read_text())
    typer.echo(f"experiment: {spec.name}  fingerprint={spec.fingerprint[:12]}")

    git_sha = current_git_sha()
    lock_sha = lockfile_sha(lockfile) if lockfile else "no-lockfile-supplied"
    if lockfile is None:
        typer.echo(
            "WARNING: no --lockfile supplied; reproducibility hash uses a placeholder.",
            err=True,
        )

    result = run_experiment(spec, code_git_sha=git_sha, lockfile_sha=lock_sha, data_root=data_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}__{spec.fingerprint[:12]}.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))

    rb = result.robustness
    typer.echo(
        f"folds={rb.n_folds}  with_trades={rb.n_folds_with_trades}  "
        f"mean_cas={rb.mean_cost_adjusted_sharpe:.3f}  "
        f"pct_pos={rb.pct_folds_positive_cas:.2f}  "
        f"max_dd={rb.max_fold_drawdown_pct:.2f}%  "
        f"ann_tv={rb.annualized_turnover:.2f}  "
        f"consistency={rb.consistency_score:.3f}"
    )
    typer.echo(f"gate: {'PASS' if result.gate.passed else 'FAIL'}")
    for r in result.gate.reasons_failed:
        typer.echo(f"  └─ {r}")
    typer.echo(f"wrote {out_path}")


@research_app.command("rank")
def rank_dir(
    directory: Path = typer.Option(..., help="Directory of ExperimentResult JSON files."),
    top_n: int | None = typer.Option(None, help="Show only the top N results."),
) -> None:
    """Print a leaderboard from every ExperimentResult in a directory."""
    rows = load_results_dir(directory)
    if not rows:
        typer.echo(f"no result files found in {directory}", err=True)
        raise typer.Exit(code=1)
    typer.echo(format_table(rows, top_n=top_n))


@research_app.command("ablation")
def ablation(
    spec_path: Path = typer.Option(..., help="Path to an AblationSpec JSON file."),
    out_dir: Path = typer.Option(
        Path("eval_reports/ablations"),
        help="Directory to write the AblationReport JSON into.",
    ),
    data_root: Path = typer.Option(
        Path.cwd(), help="Root against which spec.base_spec.data.csv_path resolves."
    ),
    lockfile: Path | None = typer.Option(
        None, help="Path to a Python lockfile; sha256 feeds the reproducibility hash."
    ),
) -> None:
    """Run a baseline + N variants, ranked by marginal Δconsistency."""
    spec = AblationSpec.from_json(spec_path.read_text())
    typer.echo(f"ablation: {spec.name}  variants={len(spec.variants)}")

    git_sha = current_git_sha()
    lock_sha = lockfile_sha(lockfile) if lockfile else "no-lockfile-supplied"
    if lockfile is None:
        typer.echo(
            "WARNING: no --lockfile supplied; reproducibility hash uses a placeholder.",
            err=True,
        )

    def _progress(i: int, total: int, name: str) -> None:
        typer.echo(f"  [{i + 1}/{total}] {name}")

    report = run_ablation(
        spec,
        code_git_sha=git_sha,
        lockfile_sha=lock_sha,
        data_root=data_root,
        on_progress=_progress,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    typer.echo("")
    typer.echo(format_ablation_table(report))
    typer.echo(f"\nwrote {out_path}")


@research_app.command("sweep")
def sweep(
    spec_path: Path = typer.Option(..., help="Base ExperimentSpec JSON to sweep."),
    thresholds: str = typer.Option(
        "0.55,0.60,0.65,0.70,0.75,0.80",
        help="Comma-separated confidence thresholds to evaluate.",
    ),
    out_dir: Path = typer.Option(
        Path("eval_reports/sweeps"),
        help="Directory to write CSVs + markdown + HTML into.",
    ),
    data_root: Path = typer.Option(
        Path.cwd(), help="Root against which spec.data.csv_path resolves."
    ),
    lockfile: Path | None = typer.Option(
        None, help="Path to a Python lockfile; sha256 feeds the reproducibility hash."
    ),
) -> None:
    """Sweep confidence thresholds against a base spec.

    Trains each fold ONCE, replays the strategy at every threshold, then
    emits CSV (per-cell, per-threshold, per-fold classifier), a markdown
    summary, and an interactive plotly HTML.
    """
    base_spec = ExperimentSpec.from_json(spec_path.read_text())
    thr_list = tuple(float(x) for x in thresholds.split(",") if x.strip())
    typer.echo(
        f"sweep base={base_spec.name}  thresholds={thr_list}  "
        f"fingerprint={base_spec.fingerprint[:12]}"
    )

    git_sha = current_git_sha()
    lock_sha = lockfile_sha(lockfile) if lockfile else "no-lockfile-supplied"
    if lockfile is None:
        typer.echo(
            "WARNING: no --lockfile supplied; reproducibility hash uses a placeholder.",
            err=True,
        )

    result = run_threshold_sweep(
        base_spec,
        thresholds=thr_list,
        code_git_sha=git_sha,
        lockfile_sha=lock_sha,
        data_root=data_root,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{base_spec.name}_sweep"
    (out_dir / f"{stem}.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))
    write_cell_csv(result, out_dir / f"{stem}__cells.csv")
    write_threshold_csv(result, out_dir / f"{stem}__thresholds.csv")
    write_classifier_csv(result, out_dir / f"{stem}__classifier.csv")
    write_markdown_summary(result, out_dir / f"{stem}.md")
    write_html_report(result, out_dir / f"{stem}.html")

    typer.echo("")
    typer.echo(f"recommendation: {result.recommendation_notes}")
    typer.echo("")
    typer.echo("per-threshold aggregate:")
    for t in result.per_threshold:
        star = " ⭐" if result.recommended_threshold == t.threshold else ""
        typer.echo(
            f"  θ={t.threshold:.2f}{star}  "
            f"mean_cas={t.robustness.mean_cost_adjusted_sharpe:>7.3f}  "
            f"pct_pos={t.robustness.pct_folds_positive_cas:>4.2f}  "
            f"max_dd={t.robustness.max_fold_drawdown_pct:>5.2f}%  "
            f"ann_tv={t.robustness.annualized_turnover:>6.2f}  "
            f"cons={t.robustness.consistency_score:>7.3f}  "
            f"{'PASS' if t.gate.passed else 'FAIL'}"
        )
    typer.echo(
        f"\nwrote {out_dir}/{stem}.* (json, cells.csv, thresholds.csv, classifier.csv, md, html)"
    )


@research_app.command("matrix")
def matrix(
    spec_path: Path = typer.Option(..., help="Path to a MatrixSpec JSON file."),
    out_dir: Path = typer.Option(
        Path("eval_reports/matrices"),
        help="Directory to write the MatrixReport JSON + CSV + markdown.",
    ),
    data_root: Path = typer.Option(
        Path.cwd(), help="Root against which base_spec.data.csv_path resolves."
    ),
    lockfile: Path | None = typer.Option(
        None, help="Path to a Python lockfile; sha256 feeds the reproducibility hash."
    ),
) -> None:
    """Expand a MatrixSpec into its Cartesian cells and run every experiment.

    Each cell runs at the base spec's fixed threshold — the matrix does
    NOT tune thresholds per cell, which would leak test-fold information
    into the model-selection loop.
    """
    matrix_spec = MatrixSpec.from_json(spec_path.read_text())
    typer.echo(
        f"matrix: {matrix_spec.name}  cells={matrix_spec.n_cells}  "
        f"grid_paths={list(matrix_spec.grid.keys())}"
    )

    git_sha = current_git_sha()
    lock_sha = lockfile_sha(lockfile) if lockfile else "no-lockfile-supplied"
    if lockfile is None:
        typer.echo(
            "WARNING: no --lockfile supplied; reproducibility hash uses a placeholder.",
            err=True,
        )

    def _progress(i: int, total: int, name: str) -> None:
        typer.echo(f"  [{i + 1}/{total}] {name}")

    report = run_matrix(
        matrix_spec,
        code_git_sha=git_sha,
        lockfile_sha=lock_sha,
        data_root=data_root,
        on_progress=_progress,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = matrix_spec.name
    (out_dir / f"{stem}.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))

    # CSV: one row per cell with the primary ranking metrics.
    with (out_dir / f"{stem}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cell_index",
                "name",
                "fingerprint",
                "passed",
                "mean_cost_adjusted_sharpe",
                "consistency_score",
                "pct_folds_positive_cas",
                "max_fold_drawdown_pct",
                "annualized_turnover",
                "total_fills",
                "reasons_failed",
            ]
        )
        for row in rank_matrix(report.rows):
            w.writerow(
                [
                    row.cell_index,
                    row.name,
                    row.fingerprint,
                    row.passed,
                    row.mean_cost_adjusted_sharpe,
                    row.consistency_score,
                    row.pct_folds_positive_cas,
                    row.max_fold_drawdown_pct,
                    row.annualized_turnover,
                    row.total_fills,
                    " ; ".join(row.reasons_failed),
                ]
            )

    (out_dir / f"{stem}.md").write_text(format_matrix_table(report))
    typer.echo("")
    typer.echo(format_matrix_table(report, top_n=10))
    typer.echo(f"\nwrote {out_dir}/{stem}.* (json, csv, md)")
