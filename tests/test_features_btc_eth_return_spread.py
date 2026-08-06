"""Contract + textbook tests for BTCETHReturnSpread."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.definitions.btc_eth_return_spread import BTCETHReturnSpread
from trade.features.multi_symbol import assert_multi_symbol_feature_contract


def _bars(symbol: str, closes: list[float]) -> list[KlineRecord]:
    return [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol=symbol,
            interval="60",
            event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
            open=c,
            high=c,
            low=c,
            close=c,
            volume=1.0,
            turnover=c,
        )
        for i, c in enumerate(closes)
    ]


def test_btc_eth_return_spread_contract() -> None:
    # 30 bars per symbol — plenty for a window=5 feature.
    btc_closes = [100.0 + 0.3 * i for i in range(30)]
    eth_closes = [50.0 + 0.4 * i for i in range(30)]
    histories = {
        "BTCUSDT": _bars("BTCUSDT", btc_closes),
        "ETHUSDT": _bars("ETHUSDT", eth_closes),
    }
    assert_multi_symbol_feature_contract(BTCETHReturnSpread(window=5), histories)


def test_return_spread_matches_manual_formula() -> None:
    btc = _bars("BTCUSDT", [100.0, 101.0, 102.0, 103.0, 104.0, 105.0])  # +5%
    eth = _bars("ETHUSDT", [50.0, 51.0, 52.0, 53.0, 54.0, 56.0])  # +12%
    v = BTCETHReturnSpread(window=5).compute({"BTCUSDT": btc, "ETHUSDT": eth})
    assert v == pytest.approx(math.log(56.0 / 50.0) - math.log(105.0 / 100.0))


def test_return_spread_returns_none_with_insufficient_history() -> None:
    btc = _bars("BTCUSDT", [100.0, 101.0])
    eth = _bars("ETHUSDT", [50.0, 51.0])
    assert BTCETHReturnSpread(window=5).compute({"BTCUSDT": btc, "ETHUSDT": eth}) is None


def test_return_spread_zero_when_both_move_identically() -> None:
    btc = _bars("BTCUSDT", [100.0 * (1.01**i) for i in range(7)])
    eth = _bars("ETHUSDT", [50.0 * (1.01**i) for i in range(7)])
    v = BTCETHReturnSpread(window=5).compute({"BTCUSDT": btc, "ETHUSDT": eth})
    assert v == pytest.approx(0.0, abs=1e-12)


def test_return_spread_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        BTCETHReturnSpread(window=0)
