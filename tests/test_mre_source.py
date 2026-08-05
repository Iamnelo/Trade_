"""Tests for MarketReplaySource — the PIT contract lives or dies here."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.mre.clock import SimClock
from trade.mre.source import MarketReplaySource


def _kline(hour: int, symbol: str = "BTCUSDT", close: float = 100.0) -> KlineRecord:
    ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour)
    return KlineRecord(
        source="bybit",
        category="linear",
        symbol=symbol,
        interval="60",
        event_time=ts,
        ingest_time=ts + timedelta(seconds=1),
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1.0,
        turnover=close,
    )


def test_interval_mismatch_rejected() -> None:
    clock = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    bad = KlineRecord(
        source="bybit",
        category="linear",
        symbol="BTCUSDT",
        interval="15",
        event_time=datetime(2024, 1, 1, tzinfo=UTC),
        ingest_time=datetime(2024, 1, 1, tzinfo=UTC),
        open=1,
        high=1,
        low=1,
        close=1,
        volume=1,
        turnover=1,
    )
    with pytest.raises(ValueError, match="interval"):
        MarketReplaySource(bars=[bad], clock=clock, interval="60")


def test_history_before_first_bar_close_is_empty() -> None:
    clock = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    bars = [_kline(0), _kline(1)]
    src = MarketReplaySource(bars=bars, clock=clock, interval="60")

    # Sim clock has not advanced past the first bar's close (01:00).
    assert src.history("BTCUSDT", "60", lookback=10) == []


def test_history_returns_only_closed_bars() -> None:
    clock = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    bars = [_kline(0), _kline(1), _kline(2)]
    src = MarketReplaySource(bars=bars, clock=clock, interval="60")

    # After the first bar closes (01:00), only bar 0 is available.
    clock.advance_to(datetime(2024, 1, 1, 1, tzinfo=UTC))
    got = src.history("BTCUSDT", "60", lookback=10)
    assert [b.event_time.hour for b in got] == [0]

    # After the second bar closes (02:00), bars 0 and 1.
    clock.advance_to(datetime(2024, 1, 1, 2, tzinfo=UTC))
    got = src.history("BTCUSDT", "60", lookback=10)
    assert [b.event_time.hour for b in got] == [0, 1]


def test_history_respects_lookback_limit() -> None:
    clock = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    bars = [_kline(h) for h in range(10)]
    src = MarketReplaySource(bars=bars, clock=clock, interval="60")
    clock.advance_to(datetime(2024, 1, 1, 10, tzinfo=UTC))
    got = src.history("BTCUSDT", "60", lookback=3)
    assert [b.event_time.hour for b in got] == [7, 8, 9]


def test_history_lookback_zero_rejected() -> None:
    clock = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    src = MarketReplaySource(bars=[_kline(0)], clock=clock, interval="60")
    with pytest.raises(ValueError, match="lookback"):
        src.history("BTCUSDT", "60", lookback=0)


def test_iter_bars_yields_in_event_time_order() -> None:
    clock = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    bars = [_kline(2), _kline(0), _kline(1, symbol="ETHUSDT")]
    src = MarketReplaySource(bars=bars, clock=clock, interval="60")
    hours = [b.event_time.hour for b in src.iter_bars()]
    assert hours == sorted(hours)


def test_latest_none_before_close() -> None:
    clock = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    src = MarketReplaySource(bars=[_kline(0)], clock=clock, interval="60")
    assert src.latest("BTCUSDT", "60") is None
    clock.advance_to(datetime(2024, 1, 1, 1, tzinfo=UTC))
    latest = src.latest("BTCUSDT", "60")
    assert latest is not None
    assert latest.event_time == datetime(2024, 1, 1, tzinfo=UTC)


def test_symbols_property_deduped_and_sorted() -> None:
    clock = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    bars = [_kline(0, "ETHUSDT"), _kline(1, "BTCUSDT"), _kline(2, "BTCUSDT")]
    src = MarketReplaySource(bars=bars, clock=clock, interval="60")
    assert src.symbols == ("BTCUSDT", "ETHUSDT")
