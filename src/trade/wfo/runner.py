"""Walk-forward runner: repeats `run_backtest` over rolling test windows.

Given a bar sequence and a `strategy_factory` (a callable that returns a
fresh strategy per fold — important so learned state does not leak across
folds), the runner:

1. Splits the timeline via `walk_forward_splits`.
2. For each split, builds a MarketReplaySource over the TEST bars and runs
   the backtest, seeding portfolio equity from the original config each
   fold.
3. Returns a `WFOReport` bundling per-fold BacktestResult + PerformanceReport.

Training-time model fitting will slot in at Phase 3 by using the split's
`train_start`..`train_end` bars; V1 benchmarks are stateless so the runner
only exercises the test window.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from trade.data.schemas import KlineRecord
from trade.metrics.performance import PerformanceReport, summarize
from trade.mre.backtest import run_backtest
from trade.mre.clock import SimClock
from trade.mre.source import MarketReplaySource
from trade.mre.strategy import Strategy
from trade.mre.types import BacktestConfig, BacktestResult
from trade.wfo.splitter import Split


@dataclass(frozen=True, slots=True)
class FoldResult:
    split: Split
    backtest: BacktestResult
    report: PerformanceReport


@dataclass(frozen=True, slots=True)
class WFOReport:
    folds: tuple[FoldResult, ...]

    @property
    def mean_sharpe(self) -> float:
        return sum(f.report.sharpe for f in self.folds) / len(self.folds) if self.folds else 0.0

    @property
    def mean_cost_adjusted_sharpe(self) -> float:
        if not self.folds:
            return 0.0
        return sum(f.report.cost_adjusted_sharpe for f in self.folds) / len(self.folds)


StrategyFactory = Callable[[], Strategy]


def run_walk_forward(
    *,
    bars: Sequence[KlineRecord],
    strategy_factory: StrategyFactory,
    splits: Sequence[Split],
    config: BacktestConfig,
    bars_per_year: int,
    interval: str,
    cost_bps_per_side: float = 5.5,
) -> WFOReport:
    """Run `run_backtest` on each split's test window and collect reports."""
    sorted_bars = sorted(bars, key=lambda b: b.event_time)
    fold_results: list[FoldResult] = []

    for split in splits:
        test_bars = sorted_bars[split.test_start : split.test_end]
        if not test_bars:
            continue
        clock = SimClock(test_bars[0].event_time)
        source = MarketReplaySource(bars=test_bars, clock=clock, interval=interval)
        strategy = strategy_factory()
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
        fold_results.append(FoldResult(split=split, backtest=result, report=report))

    return WFOReport(folds=tuple(fold_results))
