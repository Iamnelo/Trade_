"""Tests for the MLflow logging helper. Uses a tmp file:// URI, no external server."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pytest

from trade.metrics.performance import PerformanceReport
from trade.mre.types import BacktestConfig, BacktestResult, EquityPoint
from trade.tracking.mlflow import log_backtest


@pytest.fixture(autouse=True)
def _allow_file_store(monkeypatch: pytest.MonkeyPatch) -> None:
    # New MLflow versions require this opt-in to use the file:// tracking
    # backend that tests want (no server, no SQL). Production deployments
    # should use sqlite:/// or a real database.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    # Prevent test pollution from any leftover global URI.
    if "MLFLOW_TRACKING_URI" in os.environ:
        monkeypatch.delenv("MLFLOW_TRACKING_URI")


def _fake_result() -> BacktestResult:
    return BacktestResult(
        equity_curve=(EquityPoint(timestamp=datetime(2024, 1, 1, tzinfo=UTC), equity=1000.0),),
        fills=(),
        initial_equity=1000.0,
        final_equity=1100.0,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=5.5, slippage_bps=5.0),
        strategy_name="test_strategy",
        dataset_manifest_ids=("bybit_kline_BTCUSDT_60",),
        halted_reasons_seen=(),
    )


def _fake_report() -> PerformanceReport:
    return PerformanceReport(
        initial_equity=1000.0,
        final_equity=1100.0,
        total_return_pct=10.0,
        sharpe=1.2,
        sortino=1.5,
        calmar=0.8,
        max_drawdown_pct=5.0,
        ulcer_index_pct=3.0,
        hit_rate=0.55,
        turnover=1.5,
        cvar_5pct=-0.02,
        cost_adjusted_sharpe=1.0,
        n_bars=1,
        n_fills=0,
        strategy_name="test_strategy",
    )


def test_log_backtest_writes_run_with_params_metrics_and_tag(tmp_path: Path) -> None:
    uri = f"file://{tmp_path.as_posix()}/mlruns"
    run_id = log_backtest(
        result=_fake_result(),
        report=_fake_report(),
        dataset_manifest_ids=("bybit_kline_BTCUSDT_60",),
        interval="60",
        tracking_uri=uri,
        experiment_name="phase2_test",
        run_name="test_run",
    )
    assert run_id  # non-empty

    # Fetch the run back and assert what we logged.
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)

    assert run.data.params["strategy"] == "test_strategy"
    assert run.data.params["interval"] == "60"
    assert run.data.params["dataset_manifest_ids"] == "bybit_kline_BTCUSDT_60"

    assert run.data.metrics["sharpe"] == 1.2
    assert run.data.metrics["max_drawdown_pct"] == 5.0
    assert run.data.metrics["total_return_pct"] == 10.0

    assert run.data.tags["halted_reasons_seen"] == ""
