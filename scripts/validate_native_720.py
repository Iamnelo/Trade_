"""Validate isolated native 720-minute candle artifacts.

This script performs data-quality checks only. It does not load models, create
signals, simulate fills, or interact with any private/authenticated endpoint.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

EXPECTED_STEP = timedelta(hours=12)
REQUIRED_COLUMNS = {
    "symbol",
    "interval",
    "event_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def validate_symbol(root: Path, symbol: str) -> dict[str, Any]:
    files = sorted(root.glob(f"raw/bybit_kline/bybit/linear/{symbol}/720/**/*.parquet"))
    if not files:
        raise ValueError(f"No native 720 parquet files found for {symbol}")

    rows: list[dict[str, Any]] = []
    for path in files:
        table = pq.read_table(path)
        missing = REQUIRED_COLUMNS.difference(table.column_names)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows.extend(table.to_pylist())

    rows.sort(key=lambda row: row["event_time"])
    times = [row["event_time"] for row in rows]
    duplicates = len(times) - len(set(times))
    wrong_intervals = sum(str(row["interval"]) != "720" for row in rows)
    bad_ohlc = sum(
        row["low"] > min(row["open"], row["close"])
        or row["high"] < max(row["open"], row["close"])
        or row["low"] > row["high"]
        for row in rows
    )
    negative_volume = sum(row["volume"] < 0 for row in rows)
    gaps = [
        {
            "after": times[index - 1].isoformat(),
            "before": times[index].isoformat(),
            "hours": (times[index] - times[index - 1]).total_seconds() / 3600,
        }
        for index in range(1, len(times))
        if times[index] - times[index - 1] != EXPECTED_STEP
    ]

    passed = not any((duplicates, wrong_intervals, bad_ohlc, negative_volume, gaps))
    return {
        "symbol": symbol,
        "passed": passed,
        "rows": len(rows),
        "first": times[0].isoformat(),
        "last": times[-1].isoformat(),
        "duplicates": duplicates,
        "wrong_intervals": wrong_intervals,
        "bad_ohlc": bad_ohlc,
        "negative_volume": negative_volume,
        "gaps": gaps,
        "files": [str(path) for path in files],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("symbols", nargs="+")
    args = parser.parse_args()

    report = {
        "purpose": "native 720-minute public candle data validation only",
        "symbols": [validate_symbol(args.root, symbol) for symbol in args.symbols],
    }
    report["passed"] = all(item["passed"] for item in report["symbols"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
