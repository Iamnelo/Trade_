"""MLflow tracking helpers for backtests.

Every backtest is logged with the ExperimentRecord shape referenced in
V1_SPEC:

    params    : strategy_name, initial_equity, fee_bps, slippage_bps,
                dataset_manifest_ids, interval
    metrics   : sharpe, sortino, calmar, max_drawdown_pct,
                ulcer_index_pct, hit_rate, turnover, cvar_5pct,
                cost_adjusted_sharpe, final_equity, total_return_pct,
                n_bars, n_fills
    tags      : halted_reasons_seen (comma-joined; "" if none)

This is Phase 2 wiring. The Phase 2.5 reproducibility hash and feature-set
manifest IDs will land alongside the feature store rollout.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlflow

from trade.metrics.performance import PerformanceReport
from trade.mre.types import BacktestResult


def _mlflow_ready(tracking_uri: str | None, experiment_name: str) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_backtest(
    *,
    result: BacktestResult,
    report: PerformanceReport,
    dataset_manifest_ids: Sequence[str] = (),
    interval: str = "",
    tracking_uri: str | None = None,
    experiment_name: str = "backtests",
    run_name: str | None = None,
) -> str:
    """Log a single backtest run and return the mlflow run_id."""
    _mlflow_ready(tracking_uri, experiment_name)
    with mlflow.start_run(run_name=run_name or result.strategy_name) as run:
        mlflow.log_params(
            {
                "strategy": result.strategy_name,
                "initial_equity": result.config.initial_equity,
                "fee_bps": result.config.fee_bps,
                "slippage_bps": result.config.slippage_bps,
                "max_gross_notional_fraction": result.config.max_gross_notional_fraction,
                "interval": interval,
                "dataset_manifest_ids": ",".join(dataset_manifest_ids),
            }
        )
        mlflow.log_metrics(
            {
                "sharpe": report.sharpe,
                "sortino": report.sortino,
                "calmar": report.calmar,
                "max_drawdown_pct": report.max_drawdown_pct,
                "ulcer_index_pct": report.ulcer_index_pct,
                "hit_rate": report.hit_rate,
                "turnover": report.turnover,
                "cvar_5pct": report.cvar_5pct,
                "cost_adjusted_sharpe": report.cost_adjusted_sharpe,
                "final_equity": report.final_equity,
                "total_return_pct": report.total_return_pct,
                "n_bars": float(report.n_bars),
                "n_fills": float(report.n_fills),
            }
        )
        mlflow.set_tag("halted_reasons_seen", ",".join(result.halted_reasons_seen))
        return str(run.info.run_id)
