"""Tests for pre-submit sanity gates."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trade.execution.gates import SanityGate
from trade.execution.types import SanityGateConfig, SanityGateViolation
from trade.mre.types import Order, Side


def _cfg(**over: object) -> SanityGateConfig:
    base: dict[str, object] = {
        "max_notional_per_order": 1000.0,
        "max_gross_notional": 5000.0,
        "staleness_seconds": 90.0,
    }
    base.update(over)
    return SanityGateConfig(**base)  # type: ignore[arg-type]


def _order(qty: float = 0.5) -> Order:
    return Order(
        client_order_id="abc",
        symbol="BTCUSDT",
        side=Side.BUY,
        quantity=qty,
        submit_time=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_gate_allows_valid_order() -> None:
    gate = SanityGate(_cfg())
    gate.check(
        order=_order(0.5),
        mark_price=100.0,
        last_bar_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
        now=datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC),
        current_gross_notional=0.0,
    )


def test_gate_rejects_oversized_order() -> None:
    gate = SanityGate(_cfg(max_notional_per_order=50.0))
    with pytest.raises(SanityGateViolation, match="per-order notional"):
        gate.check(
            order=_order(1.0),
            mark_price=100.0,
            last_bar_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
            now=datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC),
            current_gross_notional=0.0,
        )


def test_gate_rejects_when_gross_cap_exceeded() -> None:
    gate = SanityGate(_cfg(max_gross_notional=100.0))
    with pytest.raises(SanityGateViolation, match="gross notional"):
        gate.check(
            order=_order(0.5),
            mark_price=100.0,
            last_bar_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
            now=datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC),
            current_gross_notional=80.0,
        )


def test_gate_rejects_stale_market_data() -> None:
    gate = SanityGate(_cfg(staleness_seconds=60.0))
    with pytest.raises(SanityGateViolation, match="stale"):
        gate.check(
            order=_order(0.1),
            mark_price=100.0,
            last_bar_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
            now=datetime(2024, 1, 1, 12, 5, tzinfo=UTC),  # 5 min stale
            current_gross_notional=0.0,
        )


def test_gate_rejects_nonpositive_mark_price() -> None:
    gate = SanityGate(_cfg())
    with pytest.raises(SanityGateViolation, match="mark price"):
        gate.check(
            order=_order(0.1),
            mark_price=0.0,
            last_bar_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
            now=datetime(2024, 1, 1, 12, tzinfo=UTC),
            current_gross_notional=0.0,
        )


def test_gate_rejects_naive_timestamps() -> None:
    gate = SanityGate(_cfg())
    with pytest.raises(SanityGateViolation, match="UTC-aware"):
        gate.check(
            order=_order(0.1),
            mark_price=100.0,
            last_bar_time=datetime(2024, 1, 1, 12),
            now=datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC),
            current_gross_notional=0.0,
        )


def test_gate_short_uses_absolute_quantity() -> None:
    gate = SanityGate(_cfg(max_notional_per_order=100.0))
    order = Order(
        client_order_id="abc",
        symbol="BTCUSDT",
        side=Side.SELL,
        quantity=1.5,
        submit_time=datetime(2024, 1, 1, tzinfo=UTC),
    )
    # 1.5 * 100 = 150 > 100, so rejected regardless of side.
    with pytest.raises(SanityGateViolation):
        gate.check(
            order=order,
            mark_price=100.0,
            last_bar_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
            now=datetime(2024, 1, 1, 12, tzinfo=UTC),
            current_gross_notional=timedelta(0).total_seconds(),
        )
