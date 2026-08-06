"""Tests for the MultiSymbolFeature Protocol, materialisation, and contract helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import KlineRecord
from trade.features.multi_symbol import (
    assert_multi_symbol_feature_contract,
    assert_multi_symbol_feature_respects_lookback,
    materialize_multi_symbol_feature,
)
from trade.features.types import FeatureSpec


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


class _SumOfLastClosesFeature:
    """Well-behaved multi-symbol feature: sum of last close across two symbols."""

    def __init__(self) -> None:
        self.spec = FeatureSpec(
            name="sum_last_close",
            version="1",
            inputs=("close",),
            lookback_bars=2,
            availability_delay=timedelta(0),
        )

    @property
    def primary_symbol(self) -> str:
        return "BTCUSDT"

    @property
    def required_symbols(self) -> tuple[str, ...]:
        return ("BTCUSDT", "ETHUSDT")

    def compute(self, histories: Mapping[str, Sequence[KlineRecord]]) -> float | None:
        btc = histories.get("BTCUSDT")
        eth = histories.get("ETHUSDT")
        if not btc or not eth or len(btc) < 2 or len(eth) < 2:
            return None
        # Only look at the declared lookback tail (2 bars per symbol).
        return btc[-2:][-1].close + eth[-2:][-1].close


class _LeakyMultiSymbolFeature:
    """Cheats — uses the FULL history rather than the declared 2-bar tail."""

    spec = _SumOfLastClosesFeature().spec

    @property
    def primary_symbol(self) -> str:
        return "BTCUSDT"

    @property
    def required_symbols(self) -> tuple[str, ...]:
        return ("BTCUSDT", "ETHUSDT")

    def compute(self, histories):
        btc = histories.get("BTCUSDT")
        eth = histories.get("ETHUSDT")
        if not btc or not eth or len(btc) < 2 or len(eth) < 2:
            return None
        # ILLEGAL: reaches beyond lookback tail.
        return sum(b.close for b in btc) + sum(b.close for b in eth)


def _make_histories() -> dict[str, list[KlineRecord]]:
    return {
        "BTCUSDT": _bars("BTCUSDT", [100.0 + i for i in range(6)]),
        "ETHUSDT": _bars("ETHUSDT", [50.0 + 0.5 * i for i in range(6)]),
    }


def test_contract_passes_for_well_behaved_multi_symbol_feature() -> None:
    assert_multi_symbol_feature_contract(_SumOfLastClosesFeature(), _make_histories())


def test_contract_catches_leaky_multi_symbol_feature() -> None:
    with pytest.raises(AssertionError, match="lookback"):
        assert_multi_symbol_feature_respects_lookback(_LeakyMultiSymbolFeature(), _make_histories())


def test_materialize_walks_primary_bars_and_aligns_others() -> None:
    # ETH bars start one hour after BTC bars — the first ETH-available BTC bar
    # is the second BTC bar.
    btc = _bars("BTCUSDT", [100.0, 101.0, 102.0, 103.0])
    eth = [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol="ETHUSDT",
            interval="60",
            event_time=datetime(2024, 1, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            ingest_time=datetime(2024, 1, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
            open=50.0 + i,
            high=50.0 + i,
            low=50.0 + i,
            close=50.0 + i,
            volume=1.0,
            turnover=50.0 + i,
        )
        for i in range(4)
    ]

    rows = materialize_multi_symbol_feature(
        _SumOfLastClosesFeature(),
        bars_by_symbol={"BTCUSDT": btc, "ETHUSDT": eth},
        entity_id="BTCUSDT",
    )
    # BTC bar 0 has no ETH => skipped. BTC bar 1 sees ETH[0]. lookback=2 so BTC bar 1
    # (only 2 BTC bars available) is the earliest that could compute; ETH also needs 2.
    # BTC bars 2, 3 both compute (each aligns to 2 or more ETH bars).
    assert len(rows) == 2
    assert [r.event_time.hour for r in rows] == [2, 3]


def test_materialize_skips_when_secondary_has_no_bars_yet() -> None:
    btc = _bars("BTCUSDT", [100.0, 101.0, 102.0, 103.0])
    # ETH starts AFTER all BTC bars — no primary bar can align.
    eth = [
        KlineRecord(
            source="bybit",
            category="linear",
            symbol="ETHUSDT",
            interval="60",
            event_time=datetime(2025, 1, 1, tzinfo=UTC),
            ingest_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
            open=50.0,
            high=50.0,
            low=50.0,
            close=50.0,
            volume=1.0,
            turnover=50.0,
        )
    ]
    rows = materialize_multi_symbol_feature(
        _SumOfLastClosesFeature(),
        bars_by_symbol={"BTCUSDT": btc, "ETHUSDT": eth},
        entity_id="BTCUSDT",
    )
    assert rows == []


def test_materialize_rejects_missing_symbol() -> None:
    with pytest.raises(ValueError, match="required symbol"):
        materialize_multi_symbol_feature(
            _SumOfLastClosesFeature(),
            bars_by_symbol={"BTCUSDT": _bars("BTCUSDT", [100.0, 101.0])},
            entity_id="BTCUSDT",
        )
