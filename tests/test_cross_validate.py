"""Tests for cross-source kline comparison."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.backfill.cross_validate import compare_klines
from trade.data.schemas import KlineRecord


def _make(source: str, minutes: int, close: float) -> KlineRecord:
    return KlineRecord(
        source=source,
        category="linear",
        symbol="BTCUSDT",
        interval="60",
        event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes),
        ingest_time=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
        turnover=close,
    )


def test_identical_series_zero_delta() -> None:
    a = [_make("bybit", 0, 100.0), _make("bybit", 60, 101.0)]
    b = [_make("binance", 0, 100.0), _make("binance", 60, 101.0)]
    report = compare_klines(a, b)
    assert report.common_bars == 2
    assert report.a_only_bars == 0
    assert report.b_only_bars == 0
    assert report.max_abs_delta_bps == 0.0


def test_close_delta_in_bps() -> None:
    a = [_make("bybit", 0, 100.0)]
    b = [_make("binance", 0, 100.1)]  # 10 bps above
    report = compare_klines(a, b)
    assert report.common_bars == 1
    assert report.deltas[0].close_delta_bps == pytest.approx(10.0)


def test_coverage_mismatch_counted() -> None:
    a = [_make("bybit", 0, 100.0), _make("bybit", 60, 101.0)]
    b = [_make("binance", 60, 101.0), _make("binance", 120, 102.0)]
    report = compare_klines(a, b)
    assert report.common_bars == 1
    assert report.a_only_bars == 1
    assert report.b_only_bars == 1


def test_zero_price_does_not_divide_by_zero() -> None:
    a = [_make("bybit", 0, 0.0)]
    b = [_make("binance", 0, 100.0)]
    report = compare_klines(a, b)
    assert report.deltas[0].close_delta_bps == 0.0


def test_summary_stats() -> None:
    a = [_make("bybit", i, 100.0) for i in (0, 60, 120)]
    b = [_make("binance", 0, 100.0), _make("binance", 60, 100.05), _make("binance", 120, 100.10)]
    report = compare_klines(a, b)
    assert report.common_bars == 3
    assert report.max_abs_delta_bps == pytest.approx(10.0)
    assert report.median_abs_delta_bps == pytest.approx(5.0)


def test_empty_inputs_produce_empty_report() -> None:
    report = compare_klines([], [])
    assert report.common_bars == 0
    assert report.max_abs_delta_bps == 0.0
    assert report.median_abs_delta_bps == 0.0
