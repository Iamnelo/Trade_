"""Research CLI: run one experiment, run an ablation, rank a directory."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from trade.reproducibility.git import current_git_sha, lockfile_sha
from trade.research.ablation import AblationSpec, format_ablation_table, run_ablation
from trade.research.experiment import ExperimentSpec
from trade.research.leaderboard import format_table, load_results_dir
from trade.research.runner import run_experiment

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
