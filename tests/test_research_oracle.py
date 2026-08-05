"""Tests for the Oracle (perfect foresight) research benchmark."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.research.oracle import capture_ratio, oracle_max_pnl


def _bars(prices: list[float]) -> list[KlineRecord]:
    return [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol="BTCUSDT",
            interval="60",
            event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
            open=p,
            high=p * 1.01,
            low=p * 0.99,
            close=p,
            volume=1.0,
            turnover=p,
        )
        for i, p in enumerate(prices)
    ]


def test_oracle_flat_when_prices_constant() -> None:
    result = oracle_max_pnl(_bars([100.0] * 5), initial_equity=1000.0)
    assert result.final_equity_long_only == pytest.approx(1000.0)
    assert result.final_equity_long_short == pytest.approx(1000.0)


def test_oracle_captures_monotonic_up_move() -> None:
    # Prices +10% per bar for 3 bars = 1.331x total.
    result = oracle_max_pnl(_bars([100.0, 110.0, 121.0, 133.1]), initial_equity=1000.0)
    assert result.final_equity_long_only == pytest.approx(1331.0)
    assert result.final_equity_long_short == pytest.approx(1331.0)


def test_oracle_beats_long_only_with_downside_capture() -> None:
    # Down then up: long-only misses the down leg but long/short catches both.
    result = oracle_max_pnl(_bars([100.0, 80.0, 100.0]), initial_equity=1000.0)
    # long-only: 80 -> 100 = 1.25x -> 1250.
    assert result.final_equity_long_only == pytest.approx(1250.0)
    # long/short: short-cover 100->80 = 1.25x, then long 80->100 = 1.25x = 1562.5.
    assert result.final_equity_long_short == pytest.approx(1562.5)


def test_oracle_handles_short_series() -> None:
    result = oracle_max_pnl(_bars([100.0]), initial_equity=1000.0)
    assert result.final_equity_long_only == 1000.0
    assert result.final_equity_long_short == 1000.0


def test_oracle_rejects_bad_initial_equity() -> None:
    with pytest.raises(ValueError):
        oracle_max_pnl(_bars([100.0, 110.0]), initial_equity=0.0)


def test_capture_ratio_one_when_strategy_matches_oracle() -> None:
    ratio = capture_ratio(
        strategy_final_equity=1500.0,
        oracle_final_equity=1500.0,
        initial_equity=1000.0,
    )
    assert ratio == pytest.approx(1.0)


def test_capture_ratio_half_when_strategy_captures_half() -> None:
    ratio = capture_ratio(
        strategy_final_equity=1250.0,
        oracle_final_equity=1500.0,
        initial_equity=1000.0,
    )
    # strategy return = 0.25; oracle return = 0.5; ratio = 0.5.
    assert ratio == pytest.approx(0.5)


def test_capture_ratio_zero_when_oracle_return_is_zero() -> None:
    ratio = capture_ratio(
        strategy_final_equity=1000.0,
        oracle_final_equity=1000.0,
        initial_equity=1000.0,
    )
    assert ratio == 0.0


def test_capture_ratio_negative_when_strategy_loses_on_up_market() -> None:
    ratio = capture_ratio(
        strategy_final_equity=900.0,
        oracle_final_equity=1500.0,
        initial_equity=1000.0,
    )
    assert ratio < 0.0
