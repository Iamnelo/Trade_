"""Resample committed hourly OHLCV files into aligned research candles.

Data transformation and quality reporting only: no models, signals, positions,
fills, exchange requests, credentials, or running services are involved.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HOUR_MS = 60 * 60 * 1000
FIELDS = ("event_time_ms", "open", "high", "low", "close", "volume", "turnover")


def _load_hourly(path: Path) -> tuple[list[dict[str, Any]], int]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        raw = list(reader)

    by_time: dict[int, dict[str, Any]] = {}
    for row in raw:
        timestamp = int(row["event_time_ms"])
        by_time[timestamp] = {
            "event_time_ms": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "turnover": float(row["turnover"]),
        }
    rows = [by_time[key] for key in sorted(by_time)]
    return rows, len(raw) - len(rows)


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()


def _resample(rows: list[dict[str, Any]], *, hours: int) -> tuple[list[dict[str, Any]], int]:
    interval_ms = hours * HOUR_MS
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        start = row["event_time_ms"] // interval_ms * interval_ms
        buckets[start].append(row)

    complete: list[dict[str, Any]] = []
    dropped = 0
    for start, group in sorted(buckets.items()):
        group.sort(key=lambda row: row["event_time_ms"])
        expected = [start + index * HOUR_MS for index in range(hours)]
        observed = [row["event_time_ms"] for row in group]
        if observed != expected:
            dropped += 1
            continue
        complete.append(
            {
                "event_time_ms": start,
                "open": group[0]["open"],
                "high": max(row["high"] for row in group),
                "low": min(row["low"] for row in group),
                "close": group[-1]["close"],
                "volume": sum(row["volume"] for row in group),
                "turnover": sum(row["turnover"] for row in group),
            }
        )
    return complete, dropped


def _gap_count(rows: list[dict[str, Any]], expected_ms: int) -> int:
    return sum(
        rows[index]["event_time_ms"] - rows[index - 1]["event_time_ms"] != expected_ms
        for index in range(1, len(rows))
    )


def _bad_ohlc_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        row["low"] > min(row["open"], row["close"])
        or row["high"] < max(row["open"], row["close"])
        or row["low"] > row["high"]
        or row["volume"] < 0
        for row in rows
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def process_symbol(symbol: str, source: Path, output_dir: Path, *, hours: int) -> dict[str, Any]:
    hourly, duplicates = _load_hourly(source)
    resampled, dropped = _resample(hourly, hours=hours)
    if not resampled:
        raise ValueError(f"{source}: no complete {hours}-hour blocks")

    destination = output_dir / f"{symbol}_{hours}H_resampled.csv"
    _write_csv(destination, resampled)
    hourly_gaps = _gap_count(hourly, HOUR_MS)
    output_gaps = _gap_count(resampled, hours * HOUR_MS)
    bad_ohlc = _bad_ohlc_count(resampled)
    passed = duplicates == 0 and hourly_gaps == 0 and output_gaps == 0 and bad_ohlc == 0

    return {
        "symbol": symbol,
        "passed": passed,
        "source_file": str(source),
        "output_file": str(destination),
        "source_hourly_rows": len(hourly),
        "output_rows": len(resampled),
        "first_open": _iso(resampled[0]["event_time_ms"]),
        "last_open": _iso(resampled[-1]["event_time_ms"]),
        "duplicate_hourly_rows": duplicates,
        "hourly_gap_count": hourly_gaps,
        "incomplete_buckets_dropped": dropped,
        "output_gap_count": output_gaps,
        "bad_ohlc_or_volume_rows": bad_ohlc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hours", type=int, choices=(4, 12), default=12)
    args = parser.parse_args()

    results = [
        process_symbol("BTCUSDT", Path("BTCUSDT_60_2y.csv"), args.output_dir, hours=args.hours),
        process_symbol("ETHUSDT", Path("ETHUSDT_60_2y.csv"), args.output_dir, hours=args.hours),
    ]
    common_start = max(item["first_open"] for item in results)
    common_end = min(item["last_open"] for item in results)
    report = {
        "purpose": f"offline hourly-to-{args.hours}H data-pipeline validation only",
        "source": "committed hourly CSV files",
        "native_interval_comparison_performed": False,
        "passed": all(item["passed"] for item in results),
        "common_coverage": {"start": common_start, "end": common_end},
        "symbols": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
