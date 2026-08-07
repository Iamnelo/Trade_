"""One-shot experiment runner.

`run_experiment(spec, ...)` takes an `ExperimentSpec`, loads the raw
klines from the spec's data path, wires the requested features,
computes walk-forward splits, runs the model WFO, aggregates per-fold
metrics into `RobustnessMetrics`, applies the spec's gates, and returns
an `ExperimentResult` that can be dumped straight to disk as JSON for
the leaderboard.

Loading klines from CSV keeps this runner independent of the parquet
storage layer — for real evaluations the CSV can be materialised from
the FeatureStore-backed manifests too.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade.data.schemas import KlineRecord
from trade.features.catalog import build_features
from trade.mre.types import BacktestConfig
from trade.research.experiment import ExperimentSpec
from trade.research.robustness import (
    GateResult,
    RobustnessMetrics,
    compute_robustness,
    evaluate_gates,
)
from trade.wfo.model_runner import run_walk_forward_model
from trade.wfo.splitter import Split, walk_forward_splits


@dataclass(frozen=True, slots=True)
class FoldRecord:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    reproducibility_hash: str
    n_fills: int
    total_return_pct: float
    sharpe: float
    cost_adjusted_sharpe: float
    max_drawdown_pct: float
    hit_rate: float
    turnover: float


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    spec: ExperimentSpec
    spec_fingerprint: str
    generated_at: str
    code_git_sha: str
    lockfile_sha: str
    n_bars: int
    folds: tuple[FoldRecord, ...]
    robustness: RobustnessMetrics
    gate: GateResult
    error: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "spec_fingerprint": self.spec_fingerprint,
            "generated_at": self.generated_at,
            "code_git_sha": self.code_git_sha,
            "lockfile_sha": self.lockfile_sha,
            "n_bars": self.n_bars,
            "folds": [asdict(f) for f in self.folds],
            "robustness": asdict(self.robustness),
            "gate": {
                "passed": self.gate.passed,
                "reasons_failed": list(self.gate.reasons_failed),
            },
            "error": self.error,
        }


def load_klines_csv(path: Path, *, symbol: str, interval: str) -> list[KlineRecord]:
    """CSV columns: event_time_ms, open, high, low, close, volume, turnover."""
    ingest_time = datetime.now(tz=UTC)
    bars: list[KlineRecord] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_time = datetime.fromtimestamp(int(row["event_time_ms"]) / 1000, tz=UTC)
            bars.append(
                KlineRecord(
                    source="csv",
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    event_time=event_time,
                    ingest_time=ingest_time,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    turnover=float(row["turnover"]),
                )
            )
    bars.sort(key=lambda b: b.event_time)
    return bars


def _build_splits(n_bars: int, spec: ExperimentSpec) -> list[Split]:
    return walk_forward_splits(
        n_bars=n_bars,
        train_bars=spec.wfo.train_bars,
        test_bars=spec.wfo.test_bars,
        step_bars=spec.wfo.step_bars,
        expanding=spec.wfo.expanding,
    )


def run_experiment(
    spec: ExperimentSpec,
    *,
    code_git_sha: str,
    lockfile_sha: str,
    data_root: Path,
) -> ExperimentResult:
    csv_path = data_root / spec.data.csv_path
    bars = load_klines_csv(csv_path, symbol=spec.data.symbol, interval=spec.data.interval)
    if not bars:
        raise ValueError(f"no bars loaded from {csv_path}")

    splits = _build_splits(len(bars), spec)
    if not splits:
        return ExperimentResult(
            spec=spec,
            spec_fingerprint=spec.fingerprint,
            generated_at=datetime.now(tz=UTC).isoformat(),
            code_git_sha=code_git_sha,
            lockfile_sha=lockfile_sha,
            n_bars=len(bars),
            folds=(),
            robustness=compute_robustness(
                [],
                bars_per_year=spec.backtest.bars_per_year,
                test_bars_per_fold=spec.wfo.test_bars,
            ),
            gate=GateResult(passed=False, reasons_failed=("no folds produced",)),
            error="wfo produced 0 splits — corpus too short for train_bars+test_bars",
        )

    features = build_features(list(spec.features))
    config = BacktestConfig(
        initial_equity=spec.backtest.initial_equity,
        fee_bps=spec.backtest.fee_bps,
        slippage_bps=spec.backtest.slippage_bps,
    )
    dataset_id = f"csv:{spec.data.symbol}:{spec.data.interval}:{spec.data.csv_path}"

    wfo_report = run_walk_forward_model(
        bars=bars,
        symbol=spec.data.symbol,
        interval=spec.data.interval,
        features=features,
        label_horizon_bars=spec.label.horizon_bars,
        label_up_pct=spec.label.up_pct,
        label_down_pct=spec.label.down_pct,
        splits=splits,
        config=config,
        bars_per_year=spec.backtest.bars_per_year,
        dataset_manifest_ids=[dataset_id],
        feature_manifest_ids=list(spec.features),
        code_git_sha=code_git_sha,
        python_lockfile_sha=lockfile_sha,
        model_config=spec.model.to_lightgbm_config(),
        confidence_threshold=spec.strategy.confidence_threshold,
        notional_fraction=spec.strategy.notional_fraction,
        label_mode=spec.label.mode,
        allow_short=spec.strategy.allow_short,
        calibration_fraction=spec.model.calibration_fraction,
        cost_bps_per_side=spec.backtest.fee_bps,
    )

    fold_records = tuple(
        FoldRecord(
            train_start=f.split.train_start,
            train_end=f.split.train_end,
            test_start=f.split.test_start,
            test_end=f.split.test_end,
            reproducibility_hash=f.reproducibility_hash,
            n_fills=f.report.n_fills,
            total_return_pct=f.report.total_return_pct,
            sharpe=f.report.sharpe,
            cost_adjusted_sharpe=f.report.cost_adjusted_sharpe,
            max_drawdown_pct=f.report.max_drawdown_pct,
            hit_rate=f.report.hit_rate,
            turnover=f.report.turnover,
        )
        for f in wfo_report.folds
    )
    fold_reports = [f.report for f in wfo_report.folds]
    robustness = compute_robustness(
        fold_reports,
        bars_per_year=spec.backtest.bars_per_year,
        test_bars_per_fold=spec.wfo.test_bars,
    )
    gate = evaluate_gates(robustness, gates=spec.gates, fold_reports=fold_reports)

    return ExperimentResult(
        spec=spec,
        spec_fingerprint=spec.fingerprint,
        generated_at=datetime.now(tz=UTC).isoformat(),
        code_git_sha=code_git_sha,
        lockfile_sha=lockfile_sha,
        n_bars=len(bars),
        folds=fold_records,
        robustness=robustness,
        gate=gate,
        error=None,
    )


def run_experiments(
    specs: Sequence[ExperimentSpec],
    *,
    code_git_sha: str,
    lockfile_sha: str,
    data_root: Path,
    on_progress: object | None = None,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for i, spec in enumerate(specs):
        if on_progress is not None:
            on_progress(i, len(specs), spec)  # type: ignore[operator]
        results.append(
            run_experiment(
                spec,
                code_git_sha=code_git_sha,
                lockfile_sha=lockfile_sha,
                data_root=data_root,
            )
        )
    return results
