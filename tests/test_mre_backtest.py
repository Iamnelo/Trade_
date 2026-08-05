"""End-to-end MRE tests using BuyAndHoldStrategy and a synthetic price path."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.mre.backtest import run_backtest
from trade.mre.clock import SimClock
from trade.mre.source import MarketReplaySource
from trade.mre.types import BacktestConfig
from trade.strategies.buy_hold import BuyAndHoldStrategy


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


def test_buy_hold_equity_curve_tracks_price_up() -> None:
    bars = _bars([100.0, 110.0, 121.0, 133.1])  # +10% per bar
    src = MarketReplaySource(
        bars=bars, clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    strat = BuyAndHoldStrategy(symbol="BTCUSDT", notional_fraction=1.0)
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    # First bar: strategy sees close=100, decides at close, target=10 BTC @ close.
    # Second bar (open=110): fills 10 BTC at 110 -> cost 1100 > 1000 cash.
    # This over-sizes intentionally (spot-like bookkeeping) but reveals: final
    # equity ~= starting_equity * (last_close / decision_close). With fee=0,
    # slippage=0, entry at bar_1 open (110) then held; equity at end = final.
    assert result.final_equity > result.initial_equity
    # Buy-and-hold should place ONE fill: the initial buy after bar 0 close.
    assert len(result.fills) == 1
    assert result.fills[0].symbol == "BTCUSDT"


def test_buy_hold_zero_fee_zero_slippage_matches_price_ratio() -> None:
    prices = [100.0, 110.0, 121.0]
    bars = _bars(prices)
    src = MarketReplaySource(
        bars=bars, clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    strat = BuyAndHoldStrategy(symbol="BTCUSDT", notional_fraction=1.0)
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    # Bought 10 BTC at price 110 (next bar open after decision at close of bar 0)
    # Cash after fill = 1000 - 10*110 = -100. Position = 10 BTC.
    # Final equity = cash + qty * last_close = -100 + 10*121 = 1110.
    assert result.final_equity == pytest.approx(1110.0)


def test_backtest_records_equity_at_each_bar_close() -> None:
    bars = _bars([100.0, 101.0, 102.0, 103.0, 104.0])
    src = MarketReplaySource(
        bars=bars, clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    strat = BuyAndHoldStrategy(symbol="BTCUSDT", notional_fraction=0.5)
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    assert len(result.equity_curve) == len(bars)
    # Equity curve should be non-empty and end at final_equity.
    assert result.equity_curve[-1].equity == result.final_equity


def test_backtest_is_deterministic() -> None:
    bars = _bars([100.0 + i * 0.5 for i in range(50)])
    make_src = lambda: MarketReplaySource(  # noqa: E731
        bars=bars, clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    cfg = BacktestConfig(initial_equity=1000.0, fee_bps=5.5, slippage_bps=5.0)

    r1 = run_backtest(source=make_src(), strategy=BuyAndHoldStrategy(symbol="BTCUSDT"), config=cfg)
    r2 = run_backtest(source=make_src(), strategy=BuyAndHoldStrategy(symbol="BTCUSDT"), config=cfg)

    assert r1.final_equity == r2.final_equity
    assert [(p.timestamp, p.equity) for p in r1.equity_curve] == [
        (p.timestamp, p.equity) for p in r2.equity_curve
    ]
    assert len(r1.fills) == len(r2.fills)


def test_buy_hold_wrong_symbol_no_fills() -> None:
    bars = _bars([100.0, 101.0])
    src = MarketReplaySource(
        bars=bars, clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    strat = BuyAndHoldStrategy(symbol="ETHUSDT")
    result = run_backtest(source=src, strategy=strat, config=BacktestConfig(initial_equity=1000.0))
    assert result.fills == ()
    assert result.final_equity == pytest.approx(result.initial_equity)


def test_buy_hold_rejects_out_of_range_fraction() -> None:
    with pytest.raises(ValueError):
        BuyAndHoldStrategy(symbol="BTCUSDT", notional_fraction=0.0)
    with pytest.raises(ValueError):
        BuyAndHoldStrategy(symbol="BTCUSDT", notional_fraction=1.5)
