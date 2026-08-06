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
from trade.training.pipeline import train_model
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
    max_lookback = max(f.spec.lookback_bars for f in features)
    fold_results: list[ModelFoldResult] = []

    for split in splits:
        # 1. Materialise features across the WHOLE timeline. PIT semantics in
        # `point_in_time_join` guarantee no test-time bars leak into training
        # even though the store holds them.
        store = InMemoryFeatureStore()
        for feat in features:
            store.materialize(feature=feat, entity_id=symbol, bars=sorted_bars)

        # 2. Labels on the train slice only.
        train_bars = sorted_bars[split.train_start : split.train_end]
        labels = triple_barrier_labels(
            train_bars,
            horizon_bars=label_horizon_bars,
            up_pct=label_up_pct,
            down_pct=label_down_pct,
        )
        # Drop labels whose horizon would extend past the train slice
        # (their outcome cannot be observed inside training data).
        labels = labels[: max(0, len(labels) - label_horizon_bars)]
        if not labels:
            continue

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

        # 3. Backtest on TEST bars, with warmup so features can compute on the
        # first decision. Warmup bars produce no trades (strategy naturally
        # returns [] until enough history exists) but keep equity marked.
        warmup_start = max(0, split.test_start - max_lookback)
        source_bars = sorted_bars[warmup_start : split.test_end]
        if not source_bars:
            continue
        source = MarketReplaySource(
            bars=source_bars,
            clock=SimClock(source_bars[0].event_time),
            interval=interval,
        )
        strategy = ModelDrivenStrategy(
            symbol=symbol,
            interval=interval,
            model=artifacts.model,
            features=list(features),
            calibrator=artifacts.calibrator,
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
        fold_results.append(
            ModelFoldResult(
                split=split,
                reproducibility_hash=artifacts.reproducibility_hash,
                backtest=result,
                report=report,
            )
        )
    return ModelWFOReport(folds=tuple(fold_results))
