"""Smoke tests for the typer CLI."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from trade.cli.app import app
from trade.config import get_settings
from trade.data.manifest import DatasetManifest
from trade.data.parquet import parquet_bytes_to_dataframe


def _bybit_kline_page() -> dict[str, object]:
    rows = [
        ["1704067200000", "100.0", "101.0", "99.0", "100.5", "10.0", "1005.0"],
        ["1704070800000", "100.5", "101.5", "99.5", "101.0", "11.0", "1111.0"],
    ]
    # Bybit returns newest-first.
    return {"retCode": 0, "retMsg": "OK", "result": {"list": list(reversed(rows))}}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_backfill_bybit_writes_partitions_and_manifest(tmp_path: Path) -> None:
    get_settings.cache_clear()

    runner = CliRunner()
    with respx.mock() as router:
        router.get("https://api.bybit.com/v5/market/kline").side_effect = [
            httpx.Response(200, json=_bybit_kline_page()),
            httpx.Response(200, json={"retCode": 0, "retMsg": "OK", "result": {"list": []}}),
        ]

        result = runner.invoke(
            app,
            [
                "backfill",
                "bybit",
                "--symbol",
                "BTCUSDT",
                "--interval",
                "60",
                "--start",
                "2024-01-01T00:00:00+00:00",
                "--end",
                "2024-01-01T02:00:00+00:00",
                "--local-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0, result.output

    partition_path = tmp_path / "raw/bybit_kline/bybit/linear/BTCUSDT/60/2024/01.parquet"
    manifest_path = tmp_path / "manifests/bybit_kline/BTCUSDT_60.json"
    assert partition_path.exists()
    assert manifest_path.exists()

    df = parquet_bytes_to_dataframe(partition_path.read_bytes())
    assert df.height == 2
    assert df["symbol"].to_list() == ["BTCUSDT", "BTCUSDT"]

    manifest = DatasetManifest.from_json(manifest_path.read_text())
    assert manifest.dataset == "bybit_kline_BTCUSDT_60"
    assert manifest.total_rows == 2
    assert len(manifest.partitions) == 1
    assert manifest.partitions[0].sha256 == _sha256(partition_path.read_bytes())
