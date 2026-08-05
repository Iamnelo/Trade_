"""Tests for the backtest CLI: run + benchmark-suite against a local manifest."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from trade.cli.app import app
from trade.data.manifest import DatasetManifest
from trade.data.parquet import klines_to_parquet_bytes
from trade.data.schemas import KlineRecord, partition_key
from trade.data.storage import LocalStore


def _make_manifest(tmp_path: Path, symbol: str = "BTCUSDT", n_bars: int = 60) -> Path:
    """Write a small parquet partition + manifest under tmp_path/data."""
    store = LocalStore(tmp_path / "data")
    bars = [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol=symbol,
            interval="60",
            event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
            open=100.0 + i * 0.5,
            high=100.0 + i * 0.5 + 0.1,
            low=100.0 + i * 0.5 - 0.1,
            close=100.0 + i * 0.5,
            volume=1.0,
            turnover=100.0,
        )
        for i in range(n_bars)
    ]
    key = partition_key(
        dataset="bybit_kline",
        source="bybit",
        category="linear",
        symbol=symbol,
        interval="60",
        year=2024,
        month=1,
    )
    data = klines_to_parquet_bytes(bars)
    store.put(key, data)

    manifest = DatasetManifest(dataset=f"bybit_kline_{symbol}_60")
    manifest.add_partition(
        key=key,
        data=data,
        rows=len(bars),
        event_time_min=bars[0].event_time,
        event_time_max=bars[-1].event_time,
    )
    manifest_path = tmp_path / f"manifest-{symbol}.json"
    manifest_path.write_text(manifest.to_json())
    return manifest_path


def test_backtest_run_writes_report_json(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    output = tmp_path / "reports" / "buy_hold.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "--strategy",
            "buy_hold",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "60",
            "--manifest-path",
            str(manifest_path),
            "--local-dir",
            str(tmp_path / "data"),
            "--initial-equity",
            "1000",
            "--fee-bps",
            "0",
            "--slippage-bps",
            "0",
            "--output-json",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    payload = json.loads(output.read_text())
    assert "sharpe" in payload
    assert payload["strategy_name"].startswith("buy_hold")


def test_backtest_run_unknown_strategy_errors(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "--strategy",
            "does_not_exist",
            "--symbol",
            "BTCUSDT",
            "--manifest-path",
            str(manifest_path),
            "--local-dir",
            str(tmp_path / "data"),
        ],
    )
    assert result.exit_code != 0


def test_backtest_benchmark_suite_writes_report(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path, n_bars=200)
    output = tmp_path / "suite.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "backtest",
            "benchmark-suite",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "60",
            "--manifest-path",
            str(manifest_path),
            "--local-dir",
            str(tmp_path / "data"),
            "--initial-equity",
            "1000",
            "--fee-bps",
            "0",
            "--slippage-bps",
            "0",
            "--output-json",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    payload = json.loads(output.read_text())
    assert "strategies" in payload
    assert "oracle" in payload
    keys = {s["key"] for s in payload["strategies"]}
    assert keys == {"buy_hold", "ma_cross", "momentum", "random"}
    for entry in payload["strategies"]:
        assert "oracle_capture_ratio_long_only" in entry
        assert "report" in entry
