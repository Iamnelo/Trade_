"""Tests for the four V1 benchmark strategies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.mre.backtest import run_backtest
from trade.mre.clock import SimClock
from trade.mre.source import MarketReplaySource
from trade.mre.types import BacktestConfig, PortfolioSnapshot
from trade.strategies.ma_cross import MACrossStrategy
from trade.strategies.momentum import Momentum12_1Strategy
from trade.strategies.random_signal import RandomSignalStrategy
from trade.strategies.risk_parity import RiskParityStrategy


def _bars(prices: list[float], symbol: str = "BTCUSDT") -> list[KlineRecord]:
    return [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol=symbol,
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


def _snapshot(equity: float = 1000.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(cash=equity, positions={}, marks={})


# -----------------------------------------------------------------------------
# MA Cross
# -----------------------------------------------------------------------------


def test_ma_cross_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        MACrossStrategy(symbol="BTCUSDT", fast_window=50, slow_window=20)
    with pytest.raises(ValueError):
        MACrossStrategy(symbol="BTCUSDT", notional_fraction=0.0)


def test_ma_cross_no_trade_without_enough_history() -> None:
    prices = [100.0 + i * 0.1 for i in range(10)]
    src = MarketReplaySource(
        bars=_bars(prices), clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    strat = MACrossStrategy(symbol="BTCUSDT", fast_window=5, slow_window=50)
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    assert result.fills == ()


def test_ma_cross_takes_position_in_uptrend() -> None:
    # Sustained uptrend -> fast MA > slow MA -> long.
    prices = [100.0 + i * 1.0 for i in range(120)]
    src = MarketReplaySource(
        bars=_bars(prices), clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    strat = MACrossStrategy(symbol="BTCUSDT", fast_window=10, slow_window=50)
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    assert result.final_equity > result.initial_equity
    assert len(result.fills) >= 1


# -----------------------------------------------------------------------------
# Momentum 12-1 (short-window variant for tests)
# -----------------------------------------------------------------------------


def test_momentum_takes_long_when_momentum_positive() -> None:
    # 100 bars ramping up strongly. lookback=20, skip=2 -> positive momentum quickly.
    prices = [100.0 + i * 0.5 for i in range(100)]
    src = MarketReplaySource(
        bars=_bars(prices), clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    strat = Momentum12_1Strategy(
        symbol="BTCUSDT", lookback_bars=20, skip_bars=2, notional_fraction=0.5
    )
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    assert result.final_equity > result.initial_equity
    assert len(result.fills) >= 1


def test_momentum_flat_when_history_insufficient() -> None:
    prices = [100.0] * 5
    src = MarketReplaySource(
        bars=_bars(prices), clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    strat = Momentum12_1Strategy(symbol="BTCUSDT", lookback_bars=20, skip_bars=2)
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    assert result.fills == ()


def test_momentum_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        Momentum12_1Strategy(symbol="BTCUSDT", lookback_bars=1)
    with pytest.raises(ValueError):
        Momentum12_1Strategy(symbol="BTCUSDT", skip_bars=-1)


# -----------------------------------------------------------------------------
# Random signal (deterministic)
# -----------------------------------------------------------------------------


def test_random_signal_is_deterministic() -> None:
    prices = [100.0 + i * 0.1 for i in range(200)]
    make_src = lambda: MarketReplaySource(  # noqa: E731
        bars=_bars(prices), clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    cfg = BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0)

    r1 = run_backtest(
        source=make_src(),
        strategy=RandomSignalStrategy(symbol="BTCUSDT", seed=42, rebalance_period_bars=5),
        config=cfg,
    )
    r2 = run_backtest(
        source=make_src(),
        strategy=RandomSignalStrategy(symbol="BTCUSDT", seed=42, rebalance_period_bars=5),
        config=cfg,
    )
    assert r1.final_equity == r2.final_equity
    assert len(r1.fills) == len(r2.fills)


def test_random_signal_different_seeds_diverge() -> None:
    prices = [100.0 + i * 0.1 for i in range(200)]
    make_src = lambda: MarketReplaySource(  # noqa: E731
        bars=_bars(prices), clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    cfg = BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0)

    r1 = run_backtest(
        source=make_src(),
        strategy=RandomSignalStrategy(symbol="BTCUSDT", seed=1, rebalance_period_bars=5),
        config=cfg,
    )
    r2 = run_backtest(
        source=make_src(),
        strategy=RandomSignalStrategy(symbol="BTCUSDT", seed=99, rebalance_period_bars=5),
        config=cfg,
    )
    # Different seed sequences should typically produce different fill counts on 200 bars.
    assert (r1.final_equity, len(r1.fills)) != (r2.final_equity, len(r2.fills))


def test_random_signal_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        RandomSignalStrategy(symbol="BTCUSDT", rebalance_period_bars=0)
    with pytest.raises(ValueError):
        RandomSignalStrategy(symbol="BTCUSDT", notional_fraction=0.0)


# -----------------------------------------------------------------------------
# Risk parity
# -----------------------------------------------------------------------------


def _multi_symbol_bars(n: int) -> list[KlineRecord]:
    # Two symbols with different vol characteristics.
    btc_prices = [100.0 + i * 0.5 for i in range(n)]
    eth_prices = [50.0 + (i % 5) * 0.5 for i in range(n)]
    return _bars(btc_prices, symbol="BTCUSDT") + _bars(eth_prices, symbol="ETHUSDT")


def test_risk_parity_rejects_single_symbol() -> None:
    with pytest.raises(ValueError):
        RiskParityStrategy(symbols=["BTCUSDT"])


def test_risk_parity_no_trade_without_enough_history() -> None:
    bars = _multi_symbol_bars(20)
    src = MarketReplaySource(
        bars=bars, clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    strat = RiskParityStrategy(
        symbols=["BTCUSDT", "ETHUSDT"],
        vol_lookback_bars=200,
        rebalance_period_bars=5,
    )
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    assert result.fills == ()


def test_risk_parity_rebalances_when_ready() -> None:
    bars = _multi_symbol_bars(200)
    src = MarketReplaySource(
        bars=bars, clock=SimClock(datetime(2024, 1, 1, tzinfo=UTC)), interval="60"
    )
    strat = RiskParityStrategy(
        symbols=["BTCUSDT", "ETHUSDT"],
        vol_lookback_bars=30,
        rebalance_period_bars=10,
    )
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=10000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    assert len(result.fills) >= 2  # both symbols get filled at rebalance
    symbols_traded = {f.symbol for f in result.fills}
    assert symbols_traded == {"BTCUSDT", "ETHUSDT"}
