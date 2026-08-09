"""End-to-end CLI test: train a model, then backtest and benchmark it."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from trade.cli.app import app
from trade.data.manifest import DatasetManifest
from trade.data.parquet import klines_to_parquet_bytes
from trade.data.schemas import KlineRecord, partition_key
from trade.data.storage import LocalStore


def _make_manifest(tmp_path: Path, symbol: str = "BTCUSDT", n_bars: int = 400) -> Path:
    """Write a synthetic kline parquet + manifest under tmp_path/data."""
    store = LocalStore(tmp_path / "data")
    rng = np.random.default_rng(42)
    price = 100.0
    bars: list[KlineRecord] = []
    for i in range(n_bars):
        price *= 1.0 + float(rng.normal(0.0005, 0.01))
        h = price * (1 + abs(float(rng.normal(0, 0.005))))
        lo = price * (1 - abs(float(rng.normal(0, 0.005))))
        bars.append(
            KlineRecord(
                source="bybit",
                category="linear",
                symbol=symbol,
                interval="60",
                event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
                open=price,
                high=h,
                low=lo,
                close=price,
                volume=1.0,
                turnover=price,
            )
        )
    # Group into monthly partitions.
    by_month: dict[tuple[int, int], list[KlineRecord]] = defaultdict(list)
    for b in bars:
        by_month[(b.event_time.year, b.event_time.month)].append(b)

    manifest = DatasetManifest(dataset=f"bybit_kline_{symbol}_60")
    for (year, month), records in sorted(by_month.items()):
        key = partition_key(
            dataset="bybit_kline",
            source="bybit",
            category="linear",
            symbol=symbol,
            interval="60",
            year=year,
            month=month,
        )
        data = klines_to_parquet_bytes(records)
        store.put(key, data)
        manifest.add_partition(
            key=key,
            data=data,
            rows=len(records),
            event_time_min=records[0].event_time,
            event_time_max=records[-1].event_time,
        )
    manifest_path = tmp_path / f"manifest-{symbol}.json"
    manifest_path.write_text(manifest.to_json())
    return manifest_path


def test_train_then_backtest_run_model_driven(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    model_dir = tmp_path / "model"
    lockfile = tmp_path / "lock"
    lockfile.write_text("fake-lockfile-content")

    runner = CliRunner()
    train_result = runner.invoke(
        app,
        [
            "train",
            "model",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "60",
            "--manifest-path",
            str(manifest_path),
            "--local-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(model_dir),
            "--features",
            "log_return@5",
            "--features",
            "realized_vol@20",
            "--horizon-bars",
            "6",
            "--code-git-sha",
            "test-sha",
            "--lockfile",
            str(lockfile),
        ],
    )
    assert train_result.exit_code == 0, train_result.output
    assert (model_dir / "manifest.json").exists()
    assert (model_dir / "model.joblib").exists()

    # Now backtest against the same manifest using the trained model.
    output = tmp_path / "reports" / "model_driven.json"
    backtest_result = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "--strategy",
            "model_driven",
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
            "--model-path",
            str(model_dir),
            "--confidence-threshold",
            "0.35",
            "--notional-fraction",
            "0.25",
            "--output-json",
            str(output),
        ],
    )
    assert backtest_result.exit_code == 0, backtest_result.output
    payload = json.loads(output.read_text())
    assert payload["strategy_name"] == "model_lgbm(BTCUSDT)"


def test_backtest_run_model_driven_requires_model_path(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "--strategy",
            "model_driven",
            "--symbol",
            "BTCUSDT",
            "--manifest-path",
            str(manifest_path),
            "--local-dir",
            str(tmp_path / "data"),
        ],
    )
    # Must be rejected: model_driven with no --model-path cannot run. We assert
    # on the exit code, not the error wording — newer Click/Typer route the
    # usage-error text to stderr (separate from result.output), so matching on
    # the option name in .output is brittle across framework versions. A Click
    # usage error (missing required option) exits with code 2.
    assert result.exit_code == 2


def test_benchmark_suite_with_model_includes_model_row(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path, n_bars=300)
    model_dir = tmp_path / "model"
    lockfile = tmp_path / "lock"
    lockfile.write_text("fake-lockfile")

    runner = CliRunner()
    train_result = runner.invoke(
        app,
        [
            "train",
            "model",
            "--symbol",
            "BTCUSDT",
            "--manifest-path",
            str(manifest_path),
            "--local-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(model_dir),
            "--features",
            "log_return@5",
            "--horizon-bars",
            "3",
            "--code-git-sha",
            "test-sha",
            "--lockfile",
            str(lockfile),
        ],
    )
    assert train_result.exit_code == 0, train_result.output

    output = tmp_path / "suite.json"
    suite_result = runner.invoke(
        app,
        [
            "backtest",
            "benchmark-suite",
            "--symbol",
            "BTCUSDT",
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
            "--model-path",
            str(model_dir),
            "--model-confidence-threshold",
            "0.35",
        ],
    )
    assert suite_result.exit_code == 0, suite_result.output
    payload = json.loads(output.read_text())
    keys = {s["key"] for s in payload["strategies"]}
    assert keys == {"buy_hold", "ma_cross", "momentum", "random", "model_lgbm"}
    model_entry = next(s for s in payload["strategies"] if s["key"] == "model_lgbm")
    assert "oracle_capture_ratio_long_only" in model_entry
    assert model_entry["report"]["strategy_name"] == "model_lgbm(BTCUSDT)"
