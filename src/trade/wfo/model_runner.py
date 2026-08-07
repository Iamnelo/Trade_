"""Walk-forward runner for model-driven strategies.

Per fold:

1. Materialise every feature into a fresh in-memory FeatureStore over the
   FULL bar timeline (both train and test bars — the PIT join enforces
   no-lookahead by only returning features with availability_time <=
   label.event_time, and the trained model then queries the store online
   via `ModelDrivenStrategy` during the test replay).
2. Compute triple-barrier labels on the TRAIN slice only.
3. Train a LightGBM model + isotonic calibrator via `train_model`.
4. Instantiate `ModelDrivenStrategy` from the trained artifacts.
5. Run the MRE backtest over the TEST bars (plus a small warmup so the
   longest-lookback feature has enough history to produce values on the
   first test decision).
6. Emit per-fold BacktestResult, PerformanceReport, and reproducibility
   hash.

Each fold's `reproducibility_hash` is deterministic given identical inputs
so a WFO re-run can be verified byte-for-byte.

The fold body is factored into `_train_fold` + `_backtest_fold` so the
threshold-sweep runner can amortise the expensive fit step across many
cheap strategy replays at different confidence thresholds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from trade.data.schemas import KlineRecord
from trade.features.protocol import Feature
from trade.features.store import InMemoryFeatureStore
from trade.labels.triple_barrier import triple_barrier_labels
from trade.metrics.performance import PerformanceReport, summarize
from trade.mre.backtest import run_backtest
from trade.mre.clock import SimClock
from trade.mre.source import MarketReplaySource
from trade.mre.types import BacktestConfig, BacktestResult
from trade.strategies.model_driven import ModelDrivenStrategy
from trade.training.pipeline import TrainingArtifacts, train_model
from trade.wfo.splitter import Split


@dataclass(frozen=True, slots=True)
class ModelFoldResult:
    split: Split
    reproducibility_hash: str
    backtest: BacktestResult
    report: PerformanceReport


@dataclass(frozen=True, slots=True)
class ModelWFOReport:
    folds: tuple[ModelFoldResult, ...]

    @property
    def mean_cost_adjusted_sharpe(self) -> float:
        if not self.folds:
            return 0.0
        return sum(f.report.cost_adjusted_sharpe for f in self.folds) / len(self.folds)

    @property
    def reproducibility_hashes(self) -> tuple[str, ...]:
        return tuple(f.reproducibility_hash for f in self.folds)


@dataclass(frozen=True, slots=True)
class TrainedFold:
    """Everything the threshold-sweep runner needs to re-run the strategy
    at multiple confidence thresholds without retraining."""

    split: Split
    artifacts: TrainingArtifacts
    features: tuple[Feature, ...]
    sorted_bars: tuple[KlineRecord, ...]
    symbol: str
    interval: str
    warmup_start_idx: int
    max_lookback: int


def _train_fold(
    *,
    sorted_bars: Sequence[KlineRecord],
    symbol: str,
    interval: str,
    features: Sequence[Feature],
    split: Split,
    label_horizon_bars: int,
    label_up_pct: float,
    label_down_pct: float,
    dataset_manifest_ids: Sequence[str],
    feature_manifest_ids: Sequence[str],
    code_git_sha: str,
    python_lockfile_sha: str,
    model_config: dict[str, Any] | None,
    calibration_fraction: float,
) -> TrainedFold | None:
    """Materialise features, generate train-slice labels, and fit the model.

    Returns None when the fold produces no usable labels (train slice too
    short after horizon trim) or no test bars — the caller should skip.
    """
    max_lookback = max(f.spec.lookback_bars for f in features)

    store = InMemoryFeatureStore()
    for feat in features:
        store.materialize(feature=feat, entity_id=symbol, bars=sorted_bars)

    train_bars = sorted_bars[split.train_start : split.train_end]
    labels = triple_barrier_labels(
        train_bars,
        horizon_bars=label_horizon_bars,
        up_pct=label_up_pct,
        down_pct=label_down_pct,
    )
    labels = labels[: max(0, len(labels) - label_horizon_bars)]
    if not labels:
        return None

    warmup_start = max(0, split.test_start - max_lookback)
    if not sorted_bars[warmup_start : split.test_end]:
        return None

    artifacts = train_model(
        feature_store=store,
        feature_ids=[f.spec.full_id for f in features],
        labels=labels,
        dataset_manifest_ids=dataset_manifest_ids,
        feature_manifest_ids=feature_manifest_ids,
        code_git_sha=code_git_sha,
        python_lockfile_sha=python_lockfile_sha,
        model_config=model_config,
        calibration_fraction=calibration_fraction,
    )
    return TrainedFold(
        split=split,
        artifacts=artifacts,
        features=tuple(features),
        sorted_bars=tuple(sorted_bars),
        symbol=symbol,
        interval=interval,
        warmup_start_idx=warmup_start,
        max_lookback=max_lookback,
    )


def _backtest_fold(
    trained: TrainedFold,
    *,
    config: BacktestConfig,
    bars_per_year: int,
    confidence_threshold: float,
    notional_fraction: float,
    allow_short: bool,
    cost_bps_per_side: float,
) -> ModelFoldResult:
    """Replay the test slice with the strategy at the given threshold.

    Cheap relative to `_train_fold` — the trained model + features are
    reused unchanged.
    """
    source_bars = trained.sorted_bars[trained.warmup_start_idx : trained.split.test_end]
    source = MarketReplaySource(
        bars=list(source_bars),
        clock=SimClock(source_bars[0].event_time),
        interval=trained.interval,
    )
    strategy = ModelDrivenStrategy(
        symbol=trained.symbol,
        interval=trained.interval,
        model=trained.artifacts.model,
        features=list(trained.features),
        calibrator=trained.artifacts.calibrator,
        confidence_threshold=confidence_threshold,
        notional_fraction=notional_fraction,
        allow_short=allow_short,
    )
    result = run_backtest(source=source, strategy=strategy, config=config)
    report = summarize(
        equity_curve=result.equity_curve,
        fills=result.fills,
        initial_equity=result.initial_equity,
        bars_per_year=bars_per_year,
        cost_bps_per_side=cost_bps_per_side,
        strategy_name=result.strategy_name,
        halted_reasons_seen=result.halted_reasons_seen,
    )
    return ModelFoldResult(
        split=trained.split,
        reproducibility_hash=trained.artifacts.reproducibility_hash,
        backtest=result,
        report=report,
    )


def run_walk_forward_model(
    *,
    bars: Sequence[KlineRecord],
    symbol: str,
    interval: str,
    features: Sequence[Feature],
    label_horizon_bars: int,
    label_up_pct: float,
    label_down_pct: float,
    splits: Sequence[Split],
    config: BacktestConfig,
    bars_per_year: int,
    dataset_manifest_ids: Sequence[str],
    feature_manifest_ids: Sequence[str],
    code_git_sha: str,
    python_lockfile_sha: str,
    model_config: dict[str, Any] | None = None,
    confidence_threshold: float = 0.55,
    notional_fraction: float = 0.5,
    allow_short: bool = True,
    calibration_fraction: float = 0.2,
    cost_bps_per_side: float = 5.5,
) -> ModelWFOReport:
    sorted_bars = sorted(bars, key=lambda b: b.event_time)
    fold_results: list[ModelFoldResult] = []

    for split in splits:
        trained = _train_fold(
            sorted_bars=sorted_bars,
            symbol=symbol,
            interval=interval,
            features=features,
            split=split,
            label_horizon_bars=label_horizon_bars,
            label_up_pct=label_up_pct,
            label_down_pct=label_down_pct,
            dataset_manifest_ids=dataset_manifest_ids,
            feature_manifest_ids=feature_manifest_ids,
            code_git_sha=code_git_sha,
            python_lockfile_sha=python_lockfile_sha,
            model_config=model_config,
            calibration_fraction=calibration_fraction,
        )
        if trained is None:
            continue
        fold_results.append(
            _backtest_fold(
                trained,
                config=config,
                bars_per_year=bars_per_year,
                confidence_threshold=confidence_threshold,
                notional_fraction=notional_fraction,
                allow_short=allow_short,
                cost_bps_per_side=cost_bps_per_side,
            )
        )
    return ModelWFOReport(folds=tuple(fold_results))


@dataclass(frozen=True, slots=True)
class ThresholdFoldResult:
    threshold: float
    result: ModelFoldResult


@dataclass(frozen=True, slots=True)
class ThresholdSweepFoldReport:
    split: Split
    reproducibility_hash: str
    trained: TrainedFold
    per_threshold: tuple[ThresholdFoldResult, ...]


@dataclass(frozen=True, slots=True)
class ThresholdSweepReport:
    thresholds: tuple[float, ...]
    folds: tuple[ThresholdSweepFoldReport, ...]


def run_walk_forward_threshold_sweep(
    *,
    bars: Sequence[KlineRecord],
    symbol: str,
    interval: str,
    features: Sequence[Feature],
    label_horizon_bars: int,
    label_up_pct: float,
    label_down_pct: float,
    splits: Sequence[Split],
    config: BacktestConfig,
    bars_per_year: int,
    dataset_manifest_ids: Sequence[str],
    feature_manifest_ids: Sequence[str],
    code_git_sha: str,
    python_lockfile_sha: str,
    thresholds: Sequence[float],
    model_config: dict[str, Any] | None = None,
    notional_fraction: float = 0.5,
    allow_short: bool = True,
    calibration_fraction: float = 0.2,
    cost_bps_per_side: float = 5.5,
) -> ThresholdSweepReport:
    """Train each fold ONCE and replay the strategy at every threshold.

    Because the classifier + calibrator are threshold-independent, this
    is dramatically cheaper than running `run_walk_forward_model` N times
    at N different thresholds — expensive fits are amortised across the
    (cheap) backtest replays.
    """
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    for t in thresholds:
        if not 0.0 < t <= 1.0:
            raise ValueError(f"threshold {t!r} must be in (0.0, 1.0]")

    sorted_bars = sorted(bars, key=lambda b: b.event_time)
    fold_reports: list[ThresholdSweepFoldReport] = []

    for split in splits:
        trained = _train_fold(
            sorted_bars=sorted_bars,
            symbol=symbol,
            interval=interval,
            features=features,
            split=split,
            label_horizon_bars=label_horizon_bars,
            label_up_pct=label_up_pct,
            label_down_pct=label_down_pct,
            dataset_manifest_ids=dataset_manifest_ids,
            feature_manifest_ids=feature_manifest_ids,
            code_git_sha=code_git_sha,
            python_lockfile_sha=python_lockfile_sha,
            model_config=model_config,
            calibration_fraction=calibration_fraction,
        )
        if trained is None:
            continue
        per_threshold = tuple(
            ThresholdFoldResult(
                threshold=t,
                result=_backtest_fold(
                    trained,
                    config=config,
                    bars_per_year=bars_per_year,
                    confidence_threshold=t,
                    notional_fraction=notional_fraction,
                    allow_short=allow_short,
                    cost_bps_per_side=cost_bps_per_side,
                ),
            )
            for t in thresholds
        )
        fold_reports.append(
            ThresholdSweepFoldReport(
                split=split,
                reproducibility_hash=trained.artifacts.reproducibility_hash,
                trained=trained,
                per_threshold=per_threshold,
            )
        )
    return ThresholdSweepReport(thresholds=tuple(thresholds), folds=tuple(fold_reports))
