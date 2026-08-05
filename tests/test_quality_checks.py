"""Tests for pure data-quality checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.quality.checks import (
    find_missing_bars,
    gap_report,
    price_sanity,
    staleness_seconds,
    within_expected_interval,
)
from trade.data.schemas import KlineRecord


def _kline(minute: int, close: float = 100.0) -> KlineRecord:
    ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=minute)
    return KlineRecord(
        source="bybit",
        category="linear",
        symbol="BTCUSDT",
        interval="60",
        event_time=ts,
        ingest_time=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1.0,
        turnover=close,
    )


def test_gap_report_full_coverage() -> None:
    records = [_kline(i) for i in (0, 60, 120)]
    report = gap_report(
        records,
        interval="60",
        window_start=datetime(2024, 1, 1, tzinfo=UTC),
        window_end=datetime(2024, 1, 1, 3, tzinfo=UTC),
    )
    assert report.expected_bars == 3
    assert report.observed_bars == 3
    assert report.missing_bars == 0
    assert report.gap_percent == 0.0


def test_gap_report_missing_middle_bar() -> None:
    records = [_kline(0), _kline(120)]  # missing 01:00
    report = gap_report(
        records,
        interval="60",
        window_start=datetime(2024, 1, 1, tzinfo=UTC),
        window_end=datetime(2024, 1, 1, 3, tzinfo=UTC),
    )
    assert report.expected_bars == 3
    assert report.missing_bars == 1
    assert report.missing_event_times == (datetime(2024, 1, 1, 1, tzinfo=UTC),)
    assert report.gap_percent == pytest.approx(100.0 / 3)


def test_gap_report_rejects_naive_bounds() -> None:
    with pytest.raises(ValueError, match="UTC-aware"):
        gap_report(
            [],
            interval="60",
            window_start=datetime(2024, 1, 1),
            window_end=datetime(2024, 1, 2),
        )


def test_gap_report_rejects_inverted_window() -> None:
    with pytest.raises(ValueError, match="strictly before"):
        gap_report(
            [],
            interval="60",
            window_start=datetime(2024, 1, 2, tzinfo=UTC),
            window_end=datetime(2024, 1, 1, tzinfo=UTC),
        )


def test_gap_report_ignores_records_outside_window() -> None:
    outsider = _kline(-60)
    inside = _kline(0)
    report = gap_report(
        [outsider, inside],
        interval="60",
        window_start=datetime(2024, 1, 1, tzinfo=UTC),
        window_end=datetime(2024, 1, 1, 1, tzinfo=UTC),
    )
    assert report.expected_bars == 1
    assert report.observed_bars == 1


def test_staleness_seconds_basic() -> None:
    now = datetime(2024, 1, 1, 12, tzinfo=UTC)
    last = datetime(2024, 1, 1, 11, 55, tzinfo=UTC)
    assert staleness_seconds(last, now) == 300.0


def test_staleness_seconds_negative_clamped_to_zero() -> None:
    now = datetime(2024, 1, 1, 12, tzinfo=UTC)
    future = datetime(2024, 1, 1, 12, 5, tzinfo=UTC)
    assert staleness_seconds(future, now) == 0.0


def test_staleness_rejects_naive() -> None:
    with pytest.raises(ValueError, match="UTC-aware"):
        staleness_seconds(datetime(2024, 1, 1), datetime(2024, 1, 2, tzinfo=UTC))


def test_price_sanity_flags_extreme_bar() -> None:
    records = [_kline(i * 60, close=100.0) for i in range(20)]
    records.append(_kline(20 * 60, close=200.0))  # +100% spike
    result = price_sanity(records, window=10, band_pct=0.20)
    assert result.violation_count == 1
    assert result.violations[0].close == 200.0


def test_price_sanity_no_violations_under_band() -> None:
    records = [_kline(i * 60, close=100.0 + i * 0.5) for i in range(30)]
    result = price_sanity(records, window=10, band_pct=0.20)
    assert result.violation_count == 0


def test_price_sanity_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        price_sanity([], window=1)
    with pytest.raises(ValueError):
        price_sanity([], band_pct=0.0)
    with pytest.raises(ValueError):
        price_sanity([], band_pct=1.5)


def test_find_missing_bars() -> None:
    observed = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 2, tzinfo=UTC),
    ]
    missing = find_missing_bars(
        observed,
        interval="60",
        window_start=datetime(2024, 1, 1, tzinfo=UTC),
        window_end=datetime(2024, 1, 1, 3, tzinfo=UTC),
    )
    assert missing == [datetime(2024, 1, 1, 1, tzinfo=UTC)]


def test_within_expected_interval() -> None:
    a = datetime(2024, 1, 1, tzinfo=UTC)
    b = datetime(2024, 1, 1, 1, tzinfo=UTC)
    assert within_expected_interval(a, b, interval="60")
    c = datetime(2024, 1, 1, 2, tzinfo=UTC)
    assert not within_expected_interval(a, c, interval="60")
