"""Tests for FundingRate / FundingZScoreN / FundingRegime features.

Contract tests via `assert_feature_contract` don't cover these features
usefully — funding features don't read bar VALUES, only bar TIMESTAMPS,
so the standard "perturb old bars" checks are trivially satisfied. We
test the invariants that actually matter:

- PIT: a funding settlement at t=T does not leak into the feature value
  at a bar whose event_time < T.
- Deterministic: same inputs -> same outputs.
- Handles multiple symbols correctly (no cross-contamination).
- Returns None when the window is not yet full.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.data.schemas import FundingRecord, KlineRecord
from trade.features.definitions.funding import (
    FundingRate,
    FundingRegime,
    FundingZScoreN,
)


def _bar(symbol: str, at: datetime) -> KlineRecord:
    return KlineRecord(
        source="synthetic",
        category="linear",
        symbol=symbol,
        interval="D",
        event_time=at,
        ingest_time=at + timedelta(seconds=1),
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=1.0,
        turnover=100.0,
    )


def _funding(symbol: str, at: datetime, rate: float) -> FundingRecord:
    return FundingRecord(
        source="bybit",
        category="linear",
        symbol=symbol,
        event_time=at,
        ingest_time=at + timedelta(seconds=1),
        funding_rate=rate,
    )


def test_funding_rate_returns_latest_settled_value() -> None:
    t0 = datetime(2024, 1, 1, 0, tzinfo=UTC)
    fundings = [
        _funding("BTCUSDT", t0, 0.001),
        _funding("BTCUSDT", t0 + timedelta(hours=8), 0.002),
        _funding("BTCUSDT", t0 + timedelta(hours=16), 0.003),
    ]
    feat = FundingRate(fundings)
    # Bar at 12:00 → latest funding is the 08:00 one (0.002).
    got = feat.compute([_bar("BTCUSDT", t0 + timedelta(hours=12))])
    assert got == pytest.approx(0.002)


def test_funding_rate_never_reads_future_settlement() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    fundings = [
        _funding("BTCUSDT", t0, 0.001),
        _funding("BTCUSDT", t0 + timedelta(hours=8), 0.002),
    ]
    feat = FundingRate(fundings)
    # Bar at 04:00 — the 08:00 settlement is IN THE FUTURE.
    got = feat.compute([_bar("BTCUSDT", t0 + timedelta(hours=4))])
    assert got == pytest.approx(0.001)


def test_funding_rate_returns_none_before_any_settlement() -> None:
    t0 = datetime(2024, 1, 1, 12, tzinfo=UTC)
    fundings = [_funding("BTCUSDT", t0, 0.001)]
    feat = FundingRate(fundings)
    # Bar at t0 - 1h; no settlement has occurred yet.
    got = feat.compute([_bar("BTCUSDT", t0 - timedelta(hours=1))])
    assert got is None


def test_funding_rate_returns_none_for_symbol_without_data() -> None:
    fundings = [_funding("BTCUSDT", datetime(2024, 1, 1, tzinfo=UTC), 0.001)]
    feat = FundingRate(fundings)
    got = feat.compute([_bar("ETHUSDT", datetime(2024, 1, 2, tzinfo=UTC))])
    assert got is None


def test_funding_rate_isolates_symbols() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    fundings = [
        _funding("BTCUSDT", t0, 0.001),
        _funding("ETHUSDT", t0, 0.005),
    ]
    feat = FundingRate(fundings)
    btc_val = feat.compute([_bar("BTCUSDT", t0 + timedelta(hours=1))])
    eth_val = feat.compute([_bar("ETHUSDT", t0 + timedelta(hours=1))])
    assert btc_val == pytest.approx(0.001)
    assert eth_val == pytest.approx(0.005)


def test_funding_zscore_needs_window_full_before_emitting() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    fundings = [_funding("BTCUSDT", t0 + timedelta(hours=8 * i), 0.001) for i in range(3)]
    feat = FundingZScoreN(fundings, window=5)
    got = feat.compute([_bar("BTCUSDT", t0 + timedelta(hours=8 * 3))])
    assert got is None  # only 3 records but window=5


def test_funding_zscore_zero_when_history_constant() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    fundings = [_funding("BTCUSDT", t0 + timedelta(hours=8 * i), 0.001) for i in range(10)]
    feat = FundingZScoreN(fundings, window=5)
    got = feat.compute([_bar("BTCUSDT", t0 + timedelta(hours=8 * 10))])
    assert got == pytest.approx(0.0)


def test_funding_zscore_positive_on_spike() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    fundings = [
        _funding("BTCUSDT", t0 + timedelta(hours=8 * i), 0.001 if i < 4 else 0.05)
        for i in range(5)
    ]
    feat = FundingZScoreN(fundings, window=5)
    got = feat.compute([_bar("BTCUSDT", t0 + timedelta(hours=8 * 5))])
    assert got is not None and got > 0.0


def test_funding_regime_needs_long_window_full() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    fundings = [_funding("BTCUSDT", t0 + timedelta(hours=8 * i), 0.001) for i in range(30)]
    feat = FundingRegime(fundings, short_window=5, long_window=63)
    got = feat.compute([_bar("BTCUSDT", t0 + timedelta(hours=8 * 30))])
    assert got is None


def test_funding_regime_positive_when_short_run_hotter_than_long() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    fundings = [
        _funding("BTCUSDT", t0 + timedelta(hours=8 * i), 0.001 if i < 50 else 0.05)
        for i in range(70)
    ]
    feat = FundingRegime(fundings, short_window=5, long_window=63)
    got = feat.compute([_bar("BTCUSDT", t0 + timedelta(hours=8 * 70))])
    assert got is not None and got > 0.0


def test_funding_regime_negative_when_short_run_cooler_than_long() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    fundings = [
        _funding("BTCUSDT", t0 + timedelta(hours=8 * i), 0.05 if i < 50 else 0.001)
        for i in range(70)
    ]
    feat = FundingRegime(fundings, short_window=5, long_window=63)
    got = feat.compute([_bar("BTCUSDT", t0 + timedelta(hours=8 * 70))])
    assert got is not None and got < 0.0


def test_funding_regime_rejects_bad_windows() -> None:
    with pytest.raises(ValueError):
        FundingRegime([], short_window=1, long_window=63)
    with pytest.raises(ValueError):
        FundingRegime([], short_window=63, long_window=5)


def test_funding_zscore_deterministic() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    fundings = [_funding("BTCUSDT", t0 + timedelta(hours=8 * i), 0.001 * i) for i in range(10)]
    feat = FundingZScoreN(fundings, window=5)
    bar = _bar("BTCUSDT", t0 + timedelta(hours=8 * 10))
    v1 = feat.compute([bar])
    v2 = feat.compute([bar])
    assert v1 == v2
