"""Contract + textbook tests for ATR14."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.contract import assert_feature_contract
from trade.features.definitions.atr14 import ATR14


def _ohlc_bars(rows: list[tuple[float, float, float, float]]) -> list[KlineRecord]:
    out = []
    for i, (o, hi, lo, c) in enumerate(rows):
        out.append(
            KlineRecord(
                source="bybit",
                category="linear",
                symbol="BTCUSDT",
                interval="60",
                event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume=1.0,
                turnover=c,
            )
        )
    return out


def test_atr14_contract() -> None:
    # 30 well-formed OHLC bars.
    rows: list[tuple[float, float, float, float]] = []
    for i in range(30):
        c = 100.0 + i * 0.5
        rows.append((c, c + 1.0, c - 1.0, c))
    assert_feature_contract(ATR14(), _ohlc_bars(rows))


def test_atr14_matches_manual_over_15_bars() -> None:
    # 15 bars with constant HL range of 2 and no gap -> TR = 2 every bar
    # -> ATR14 = 2.
    rows = [(100.0, 101.0, 99.0, 100.0)] * 15
    v = ATR14().compute(_ohlc_bars(rows))
    assert v == pytest.approx(2.0)


def test_atr14_rejects_bad_period() -> None:
    with pytest.raises(ValueError):
        ATR14(period=1)
