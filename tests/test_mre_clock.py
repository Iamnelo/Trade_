"""Tests for the monotonic simulation clock."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade.mre.clock import SimClock


def test_naive_start_rejected() -> None:
    with pytest.raises(ValueError, match="UTC-aware"):
        SimClock(datetime(2024, 1, 1))


def test_advance_forward_updates_now() -> None:
    clk = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    clk.advance_to(datetime(2024, 1, 1, 12, tzinfo=UTC))
    assert clk.now == datetime(2024, 1, 1, 12, tzinfo=UTC)


def test_advance_to_same_time_allowed() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    clk = SimClock(ts)
    clk.advance_to(ts)
    assert clk.now == ts


def test_backward_advance_rejected() -> None:
    clk = SimClock(datetime(2024, 1, 2, tzinfo=UTC))
    with pytest.raises(ValueError, match="cannot move backward"):
        clk.advance_to(datetime(2024, 1, 1, tzinfo=UTC))


def test_naive_advance_rejected() -> None:
    clk = SimClock(datetime(2024, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="UTC-aware"):
        clk.advance_to(datetime(2024, 1, 2))
