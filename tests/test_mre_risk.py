"""Tests for the RiskManager DD gates."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade.mre.risk import RiskManager
from trade.mre.types import TargetPosition


def test_initial_state_not_halted() -> None:
    rm = RiskManager(initial_equity=1000.0)
    assert not rm.is_halted
    assert rm.halted_reasons == ()


def test_rejects_misordered_limits() -> None:
    with pytest.raises(ValueError):
        RiskManager(initial_equity=1000.0, daily_pct=0.10, weekly_pct=0.05, monthly_pct=0.12)


def test_daily_halt_after_dd_exceeds() -> None:
    rm = RiskManager(initial_equity=1000.0, daily_pct=0.03)
    # Same UTC day, equity drops 5% (> 3%) → halt.
    rm.update(datetime(2024, 1, 1, 12, tzinfo=UTC), 1000.0)
    rm.update(datetime(2024, 1, 1, 13, tzinfo=UTC), 950.0)
    assert rm.is_halted
    assert "daily_dd" in rm.halted_reasons


def test_daily_halt_resets_on_new_day() -> None:
    rm = RiskManager(initial_equity=1000.0, daily_pct=0.03)
    rm.update(datetime(2024, 1, 1, 12, tzinfo=UTC), 1000.0)
    rm.update(datetime(2024, 1, 1, 13, tzinfo=UTC), 950.0)
    assert rm.is_halted
    # Next UTC day: halt clears (HWM reset).
    rm.update(datetime(2024, 1, 2, 1, tzinfo=UTC), 950.0)
    assert not rm.is_halted


def test_weekly_and_monthly_layers() -> None:
    rm = RiskManager(initial_equity=1000.0)  # defaults
    # Sustained slow drop that doesn't trip daily but trips weekly.
    rm.update(datetime(2024, 1, 1, 12, tzinfo=UTC), 1000.0)  # Monday
    rm.update(datetime(2024, 1, 2, 12, tzinfo=UTC), 985.0)  # -1.5%
    rm.update(datetime(2024, 1, 3, 12, tzinfo=UTC), 970.0)  # -3% from Mon
    rm.update(datetime(2024, 1, 4, 12, tzinfo=UTC), 950.0)  # -5%
    rm.update(datetime(2024, 1, 5, 12, tzinfo=UTC), 910.0)  # -9%  -> trips 8% weekly
    assert "weekly_dd" in rm.halted_reasons


def test_gate_blocks_new_positions_when_halted() -> None:
    rm = RiskManager(initial_equity=1000.0, daily_pct=0.03)
    rm.update(datetime(2024, 1, 1, tzinfo=UTC), 1000.0)
    rm.update(datetime(2024, 1, 1, 1, tzinfo=UTC), 900.0)  # -10% > 3% -> halt
    assert rm.is_halted

    targets = [
        TargetPosition(symbol="BTCUSDT", target_qty=1.0),  # blocked
        TargetPosition(symbol="ETHUSDT", target_qty=0.0),  # allowed (flatten)
    ]
    got = rm.gate_targets(targets)
    assert len(got) == 1
    assert got[0].symbol == "ETHUSDT"


def test_gate_passthrough_when_not_halted() -> None:
    rm = RiskManager(initial_equity=1000.0)
    targets = [TargetPosition(symbol="BTCUSDT", target_qty=1.0)]
    assert rm.gate_targets(targets) == targets


def test_reasons_ever_triggered_remembers_after_reset() -> None:
    # -5% trips daily (>3%) but stays under weekly (8%) and monthly (12%),
    # so a fresh UTC day releases the halt cleanly.
    rm = RiskManager(initial_equity=1000.0, daily_pct=0.03)
    rm.update(datetime(2024, 1, 1, tzinfo=UTC), 1000.0)
    rm.update(datetime(2024, 1, 1, 1, tzinfo=UTC), 950.0)  # daily halt fires
    rm.update(datetime(2024, 1, 2, 1, tzinfo=UTC), 950.0)  # new day → clears
    assert "daily_dd" in rm.reasons_ever_triggered
    assert not rm.is_halted
