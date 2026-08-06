"""Tests for execution value types."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade.execution.types import (
    ExchangeError,
    SanityGateConfig,
    SanityGateViolation,
    TransientExchangeError,
    VenueOrderAck,
    VenuePosition,
)


def test_venue_order_ack_frozen() -> None:
    ack = VenueOrderAck(
        client_order_id="abc",
        venue_order_id="v-1",
        accepted_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    with pytest.raises((AttributeError, TypeError)):
        ack.venue_order_id = "v-2"  # type: ignore[misc]


def test_venue_position_signed_quantity() -> None:
    long_pos = VenuePosition(symbol="BTCUSDT", quantity=0.5, entry_price=100.0, unrealized_pnl=1.0)
    short_pos = VenuePosition(
        symbol="ETHUSDT", quantity=-0.5, entry_price=200.0, unrealized_pnl=-2.0
    )
    assert long_pos.quantity > 0
    assert short_pos.quantity < 0


def test_sanity_gate_config_is_a_pod() -> None:
    cfg = SanityGateConfig(
        max_notional_per_order=100.0, max_gross_notional=500.0, staleness_seconds=90.0
    )
    assert cfg.max_notional_per_order == 100.0


def test_exchange_error_carries_code_and_message() -> None:
    exc = ExchangeError(code=10001, message="bad param")
    assert exc.code == 10001
    assert exc.message == "bad param"
    assert "10001" in str(exc)


def test_transient_is_subclass_of_exchange_error() -> None:
    assert issubclass(TransientExchangeError, ExchangeError)


def test_sanity_violation_is_runtime_error() -> None:
    assert issubclass(SanityGateViolation, RuntimeError)
