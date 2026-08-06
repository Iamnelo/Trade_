"""Backtest CLI: run one strategy or the full benchmark suite, produce a report.

Strategy selection is a small registry so the CLI itself does not import each
strategy's construction knobs — this keeps the CLI stable while the strategy
catalogue grows in Phase 3+.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from trade.data.manifest import DatasetManifest
from trade.data.storage import LocalStore
from trade.features.catalog import build_features
from trade.metrics.performance import HOURS_PER_YEAR, PerformanceReport, summarize
from trade.model.persistence import load_training_artifacts
from trade.mre.backtest import run_backtest
from trade.mre.clock import SimClock
from trade.mre.source import MarketReplaySource
from trade.mre.strategy import Strategy
from trade.mre.types import BacktestConfig
from trade.research.oracle import capture_ratio, oracle_max_pnl
from trade.strategies.buy_hold import BuyAndHoldStrategy
from trade.strategies.ma_cross import MACrossStrategy
from trade.strategies.model_driven import ModelDrivenStrategy
from trade.strategies.momentum import Momentum12_1Strategy
from trade.strategies.random_signal import RandomSignalStrategy
from trade.strategies.risk_parity import RiskParityStrategy
from trade.tracking.mlflow import log_backtest

backtest_app = typer.Typer(no_args_is_help=True)


StrategyBuilder = Callable[[str, str], Strategy]

_BUILDERS: dict[str, StrategyBuilder] = {
    "buy_hold": lambda symbol, _interval: BuyAndHoldStrategy(symbol=symbol),
    "ma_cross": lambda symbol, interval: MACrossStrategy(symbol=symbol, interval=interval),
    "momentum": lambda symbol, interval: Momentum12_1Strategy(symbol=symbol, interval=interval),
    "random": lambda symbol, interval: RandomSignalStrategy(symbol=symbol, interval=interval),
}


def _build_model_driven(
    *,
    symbol: str,
    interval: str,
    model_path: Path,
    confidence_threshold: float,
    notional_fraction: float,
    allow_short: bool,
) -> Strategy:
    artifacts = load_training_artifacts(model_path)
    features = build_features(artifacts.feature_ids)
    return ModelDrivenStrategy(
        symbol=symbol,
        interval=interval,
        model=artifacts.model,
        features=features,
        calibrator=artifacts.calibrator,
        confidence_threshold=confidence_threshold,
        notional_fraction=notional_fraction,
        allow_short=allow_short,
    )


def _load_source(
    *, manifest_path: Path, local_dir: Path, interval: str, symbol: str | None
) -> tuple[MarketReplaySource, DatasetManifest]:
    manifest = DatasetManifest.from_json(manifest_path.read_text())
    if not manifest.partitions:
        raise typer.BadParameter(f"manifest has no partitions: {manifest_path}")
    store = LocalStore(local_dir)
    clock_start = manifest.coverage_start
    if clock_start is None:
        raise typer.BadParameter("manifest has no coverage window")
    source = MarketReplaySource.from_manifest(
        manifest=manifest,
        store=store,
        clock=SimClock(clock_start),
        interval=interval,
        symbol_filter=symbol,
    )
    if len(source) == 0:
        raise typer.BadParameter(
            f"manifest yielded 0 bars for interval={interval!r} symbol={symbol!r}"
        )
    return source, manifest


def _report_to_dict(report: PerformanceReport) -> dict[str, Any]:
    d: dict[str, Any] = asdict(report) if is_dataclass(report) else {}
    d["halted_reasons_seen"] = list(report.halted_reasons_seen)
    return d


@backtest_app.command("run")
def run(
    strategy: str = typer.Option(..., help="Strategy key from the registry, or 'model_driven'."),
    symbol: str = typer.Option("BTCUSDT"),
    interval: str = typer.Option("60"),
    manifest_path: Path = typer.Option(...),
    local_dir: Path = typer.Option(...),
    initial_equity: float = typer.Option(1000.0),
    fee_bps: float = typer.Option(5.5),
    slippage_bps: float = typer.Option(5.0),
    output_json: Path | None = typer.Option(None, help="Write PerformanceReport JSON here."),
    mlflow_uri: str | None = typer.Option(None, help="MLflow tracking URI to log into."),
    experiment: str = typer.Option("backtests"),
    model_path: Path | None = typer.Option(
        None, help="Required when --strategy model_driven: path to saved TrainingArtifacts."
    ),
    confidence_threshold: float = typer.Option(
        0.55, help="Model strategy: minimum argmax prob to take a position."
    ),
    notional_fraction: float = typer.Option(
        0.5, help="Model strategy: fraction of equity to deploy per position."
    ),
    allow_short: bool = typer.Option(True, help="Model strategy: allow short positions."),
) -> None:
    """Run one backtest against a stored manifest."""
    if strategy == "model_driven":
        if model_path is None:
            raise typer.BadParameter("--model-path is required when --strategy model_driven")
        strategy_instance = _build_model_driven(
            symbol=symbol,
            interval=interval,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            notional_fraction=notional_fraction,
            allow_short=allow_short,
        )
    else:
        builder = _BUILDERS.get(strategy)
        if builder is None:
            raise typer.BadParameter(
                f"unknown strategy {strategy!r}; choose from: "
                f"{sorted([*_BUILDERS, 'model_driven'])}"
            )
        strategy_instance = builder(symbol, interval)

    source, manifest = _load_source(
        manifest_path=manifest_path, local_dir=local_dir, interval=interval, symbol=symbol
    )
    config = BacktestConfig(
        initial_equity=initial_equity, fee_bps=fee_bps, slippage_bps=slippage_bps
    )
    result = run_backtest(source=source, strategy=strategy_instance, config=config)
    report = summarize(
        equity_curve=result.equity_curve,
        fills=result.fills,
        initial_equity=result.initial_equity,
        bars_per_year=HOURS_PER_YEAR,
        strategy_name=result.strategy_name,
        halted_reasons_seen=result.halted_reasons_seen,
    )

    if mlflow_uri is not None:
        log_backtest(
            result=result,
            report=report,
            dataset_manifest_ids=[manifest.dataset],
            interval=interval,
            tracking_uri=mlflow_uri,
            experiment_name=experiment,
        )
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(_report_to_dict(report), indent=2, default=str))

    typer.echo(json.dumps(_report_to_dict(report), indent=2, default=str))


@backtest_app.command("benchmark-suite")
def benchmark_suite(
    symbol: str = typer.Option("BTCUSDT"),
    interval: str = typer.Option("60"),
    manifest_path: Path = typer.Option(...),
    local_dir: Path = typer.Option(...),
    initial_equity: float = typer.Option(1000.0),
    fee_bps: float = typer.Option(5.5),
    slippage_bps: float = typer.Option(5.0),
    output_json: Path = typer.Option(...),
    include_risk_parity: bool = typer.Option(False, help="Requires --secondary-symbol"),
    secondary_symbol: str = typer.Option("ETHUSDT"),
    model_path: Path | None = typer.Option(
        None, help="Path to saved model artifacts; when set, includes model_lgbm in the suite."
    ),
    model_confidence_threshold: float = typer.Option(0.55),
    model_notional_fraction: float = typer.Option(0.5),
    model_allow_short: bool = typer.Option(True),
) -> None:
    """Run every V1 benchmark against the same manifest and write a JSON baseline report.

    Also computes the Oracle capture ratio per strategy. The Oracle uses future
    data and is research-only — it is never accessible from a strategy's on_bar.
    """
    source, manifest = _load_source(
        manifest_path=manifest_path, local_dir=local_dir, interval=interval, symbol=None
    )
    config = BacktestConfig(
        initial_equity=initial_equity, fee_bps=fee_bps, slippage_bps=slippage_bps
    )

    # Oracle uses the primary symbol's bars only.
    primary_bars = [b for b in source.iter_bars() if b.symbol == symbol]
    oracle = oracle_max_pnl(primary_bars, initial_equity=initial_equity)

    # Fresh source per strategy so clocks/state do not leak across runs.
    def fresh_source(sym_filter: str | None) -> MarketReplaySource:
        s, _ = _load_source(
            manifest_path=manifest_path,
            local_dir=local_dir,
            interval=interval,
            symbol=sym_filter,
        )
        return s

    strategies: list[tuple[str, Strategy, MarketReplaySource]] = [
        ("buy_hold", BuyAndHoldStrategy(symbol=symbol), fresh_source(symbol)),
        ("ma_cross", MACrossStrategy(symbol=symbol, interval=interval), fresh_source(symbol)),
        (
            "momentum",
            Momentum12_1Strategy(symbol=symbol, interval=interval),
            fresh_source(symbol),
        ),
        ("random", RandomSignalStrategy(symbol=symbol, interval=interval), fresh_source(symbol)),
    ]
    if include_risk_parity:
        strategies.append(
            (
                "risk_parity",
                RiskParityStrategy(symbols=[symbol, secondary_symbol], interval=interval),
                fresh_source(None),
            )
        )
    if model_path is not None:
        strategies.append(
            (
                "model_lgbm",
                _build_model_driven(
                    symbol=symbol,
                    interval=interval,
                    model_path=model_path,
                    confidence_threshold=model_confidence_threshold,
                    notional_fraction=model_notional_fraction,
                    allow_short=model_allow_short,
                ),
                fresh_source(symbol),
            )
        )

    per_strategy: list[dict[str, Any]] = []
    for key, strat, src in strategies:
        result = run_backtest(source=src, strategy=strat, config=config)
        report = summarize(
            equity_curve=result.equity_curve,
            fills=result.fills,
            initial_equity=result.initial_equity,
            bars_per_year=HOURS_PER_YEAR,
            strategy_name=result.strategy_name,
            halted_reasons_seen=result.halted_reasons_seen,
        )
        cr_long_only = capture_ratio(
            strategy_final_equity=result.final_equity,
            oracle_final_equity=oracle.final_equity_long_only,
            initial_equity=initial_equity,
        )
        per_strategy.append(
            {
                "key": key,
                "report": _report_to_dict(report),
                "oracle_capture_ratio_long_only": cr_long_only,
            }
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "symbol": symbol,
            "interval": interval,
            "initial_equity": initial_equity,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
        },
        "dataset_manifest": manifest.dataset,
        "oracle": {
            "final_equity_long_only": oracle.final_equity_long_only,
            "final_equity_long_short": oracle.final_equity_long_short,
            "total_return_long_only": oracle.total_return_long_only,
            "total_return_long_short": oracle.total_return_long_short,
        },
        "strategies": per_strategy,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, default=str))
    typer.echo(f"Wrote baseline report: {output_json}")
