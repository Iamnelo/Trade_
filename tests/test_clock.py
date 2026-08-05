"""Tests for trade.utils.clock — the UTC discipline lives or dies here."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.utils.clock import from_epoch_ms, to_epoch_ms, utcnow


def test_utcnow_is_utc() -> None:
    ts = utcnow()
    assert ts.tzinfo is not None
    assert ts.utcoffset() == timedelta(0)


def test_roundtrip_ms() -> None:
    original = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    restored = from_epoch_ms(to_epoch_ms(original))
    assert restored == original


def test_from_epoch_ms_returns_utc() -> None:
    ts = from_epoch_ms(1_700_000_000_000)
    assert ts.tzinfo is not None
    assert ts.utcoffset() == timedelta(0)
    assert ts == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)


def test_reject_naive() -> None:
    with pytest.raises(ValueError, match="Naive datetime"):
        to_epoch_ms(datetime(2025, 1, 1))
