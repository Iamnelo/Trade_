"""Tests for the deterministic client-order-id generator."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade.execution.client_ids import make_client_order_id


def test_same_inputs_produce_same_id() -> None:
    kwargs: dict[str, object] = {
        "symbol": "BTCUSDT",
        "submit_time": datetime(2024, 1, 1, tzinfo=UTC),
        "sequence": 1,
    }
    a = make_client_order_id(**kwargs)  # type: ignore[arg-type]
    b = make_client_order_id(**kwargs)  # type: ignore[arg-type]
    assert a == b


def test_different_sequence_produces_different_id() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    a = make_client_order_id(symbol="BTCUSDT", submit_time=ts, sequence=1)
    b = make_client_order_id(symbol="BTCUSDT", submit_time=ts, sequence=2)
    assert a != b


def test_different_symbol_produces_different_id() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    a = make_client_order_id(symbol="BTCUSDT", submit_time=ts, sequence=1)
    b = make_client_order_id(symbol="ETHUSDT", submit_time=ts, sequence=1)
    assert a != b


def test_id_length_fits_bybit_order_link_id() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    got = make_client_order_id(symbol="BTCUSDT", submit_time=ts, sequence=1)
    assert 1 <= len(got) <= 45


def test_salt_changes_id() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    a = make_client_order_id(symbol="BTCUSDT", submit_time=ts, sequence=1, salt="v1")
    b = make_client_order_id(symbol="BTCUSDT", submit_time=ts, sequence=1, salt="v2")
    assert a != b


def test_rejects_naive_submit_time() -> None:
    with pytest.raises(ValueError, match="UTC-aware"):
        make_client_order_id(symbol="BTCUSDT", submit_time=datetime(2024, 1, 1), sequence=1)


def test_rejects_negative_sequence() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_client_order_id(
            symbol="BTCUSDT", submit_time=datetime(2024, 1, 1, tzinfo=UTC), sequence=-1
        )
