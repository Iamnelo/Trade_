"""Model training CLI.

`python -m trade.cli train ...` loads a kline manifest, materialises a
configured feature set into the FeatureStore, computes triple-barrier
labels, trains a LightGBM + isotonic classifier, and writes the resulting
artifacts to a directory (with the reproducibility hash for later
verification).
"""

from __future__ import annotations

from pathlib import Path

import typer

from trade.data.manifest import DatasetManifest
from trade.data.storage import LocalStore
from trade.features.catalog import build_features
from trade.features.store import InMemoryFeatureStore
from trade.labels.triple_barrier import triple_barrier_labels
from trade.model.persistence import save_training_artifacts
from trade.mre.clock import SimClock
from trade.mre.source import MarketReplaySource
from trade.reproducibility.git import current_git_sha, lockfile_sha
from trade.training.pipeline import train_model

train_app = typer.Typer(no_args_is_help=True)


@train_app.command("model")
def train_model_cmd(
    symbol: str = typer.Option(..., help="Trading symbol, e.g., BTCUSDT."),
    interval: str = typer.Option("60"),
    manifest_path: Path = typer.Option(..., help="Path to a DatasetManifest JSON."),
    local_dir: Path = typer.Option(..., help="LocalStore root holding the parquet partitions."),
    output_dir: Path = typer.Option(..., help="Directory to write the trained artifacts into."),
    features: list[str] = typer.Option(
        ["log_return@5", "realized_vol@20", "atr@14"],
        help="Feature ids (name@version). Must be in the catalog.",
    ),
    horizon_bars: int = typer.Option(6),
    up_pct: float = typer.Option(0.01),
    down_pct: float = typer.Option(0.01),
    calibration_fraction: float = typer.Option(0.2),
    code_git_sha: str | None = typer.Option(
        None,
        help="Override the git sha in the reproducibility hash (default: current HEAD).",
    ),
    lockfile: Path | None = typer.Option(
        None,
        help="Path to a python lockfile whose sha256 feeds the reproducibility hash.",
    ),
) -> None:
    """Train a LightGBM model against a kline manifest and save the artifacts."""
    manifest = DatasetManifest.from_json(manifest_path.read_text())
    if not manifest.partitions:
        raise typer.BadParameter(f"empty manifest: {manifest_path}")

    store_root = LocalStore(local_dir)
    clock_start = manifest.coverage_start
    if clock_start is None:
        raise typer.BadParameter("manifest has no coverage window")
    source = MarketReplaySource.from_manifest(
        manifest=manifest,
        store=store_root,
        clock=SimClock(clock_start),
        interval=interval,
        symbol_filter=symbol,
    )
    bars = list(source.iter_bars())
    if not bars:
        raise typer.BadParameter(
            f"manifest yielded 0 bars for symbol={symbol!r} interval={interval!r}"
        )

    feats = build_features(features)
    feature_store = InMemoryFeatureStore()
    for feat in feats:
        feature_store.materialize(feature=feat, entity_id=symbol, bars=bars)

    labels = triple_barrier_labels(
        bars, horizon_bars=horizon_bars, up_pct=up_pct, down_pct=down_pct
    )
    # Drop labels whose horizon extends past the corpus so their outcome is
    # observable — matches the WFO runner's convention.
    labels = labels[: max(0, len(labels) - horizon_bars)]
    if not labels:
        raise typer.BadParameter(
            "corpus too short after horizon trim; extend the manifest or lower horizon"
        )

    resolved_git = code_git_sha or current_git_sha()
    resolved_lock = lockfile_sha(lockfile) if lockfile is not None else "no-lockfile-supplied"
    if lockfile is None:
        typer.echo(
            "WARNING: no --lockfile supplied; the reproducibility hash will use a "
            "placeholder sha and cannot be verified against committed artifacts.",
            err=True,
        )

    artifacts = train_model(
        feature_store=feature_store,
        feature_ids=[f.spec.full_id for f in feats],
        labels=labels,
        dataset_manifest_ids=[manifest.dataset],
        feature_manifest_ids=[f.spec.full_id for f in feats],
        code_git_sha=resolved_git,
        python_lockfile_sha=resolved_lock,
        calibration_fraction=calibration_fraction,
    )
    save_training_artifacts(artifacts, output_dir)
    typer.echo(
        f"Wrote artifacts to {output_dir}\n"
        f"  reproducibility_hash: {artifacts.reproducibility_hash}\n"
        f"  train_rows: {artifacts.train_rows}, calibration_rows: {artifacts.calibration_rows}"
    )
