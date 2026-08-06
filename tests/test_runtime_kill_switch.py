"""Tests for KillSwitchController + trip predicates + runtime metrics."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade.execution.types import VenuePosition
from trade.runtime.kill_switch import (
    KillSwitchController,
    check_data_staleness,
    check_exchange_error_rate,
    check_health_floor,
    check_reconciliation_drift,
)
from trade.runtime.metrics import build_runtime_metrics
from trade.runtime.reconciler import ReconciliationResult


def test_initial_state_not_halted() -> None:
    ks = KillSwitchController()
    assert not ks.is_halted
    assert ks.active_reasons == ()


def test_trip_records_reason_and_history() -> None:
    ks = KillSwitchController()
    ks.trip(reason="data_staleness", detail={"seconds": 120.0})
    assert ks.is_halted
    assert "data_staleness" in ks.active_reasons
    assert len(ks.trip_history()) == 1
    assert ks.reasons_ever_triggered == ("data_staleness",)


def test_trip_same_reason_twice_dedupes_active_but_appends_history() -> None:
    ks = KillSwitchController()
    ks.trip(reason="r", detail={"n": 1})
    ks.trip(reason="r", detail={"n": 2})
    assert ks.active_reasons == ("r",)
    assert len(ks.trip_history()) == 2


def test_clear_removes_reason() -> None:
    ks = KillSwitchController()
    ks.trip(reason="r", detail={})
    assert ks.clear("r") is True
    assert not ks.is_halted
    # Clear something that wasn't active.
    assert ks.clear("unknown") is False


def test_clear_all_wipes_active_reasons() -> None:
    ks = KillSwitchController()
    ks.trip(reason="a", detail={})
    ks.trip(reason="b", detail={})
    cleared = ks.clear_all()
    assert cleared == ("a", "b")
    assert not ks.is_halted


def test_reasons_ever_triggered_persists_after_clear() -> None:
    ks = KillSwitchController()
    ks.trip(reason="r", detail={})
    ks.clear("r")
    assert ks.reasons_ever_triggered == ("r",)


def test_trip_rejects_empty_reason() -> None:
    ks = KillSwitchController()
    with pytest.raises(ValueError):
        ks.trip(reason="", detail={})


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def test_data_staleness_predicate_trips_over_threshold() -> None:
    got = check_data_staleness(
        last_bar_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
        now=datetime(2024, 1, 1, 12, 3, tzinfo=UTC),
        threshold_seconds=90.0,
    )
    assert got is not None
    assert got[0] == "data_staleness"
    assert got[1]["seconds"] == 180.0


def test_data_staleness_predicate_none_under_threshold() -> None:
    got = check_data_staleness(
        last_bar_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
        now=datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC),
        threshold_seconds=90.0,
    )
    assert got is None


def test_exchange_error_rate_predicate_trips_over_threshold() -> None:
    got = check_exchange_error_rate(successes=10, errors=5, threshold_ratio=0.2, min_samples=5)
    assert got is not None
    assert got[0] == "exchange_error_rate"
    assert got[1]["ratio"] == pytest.approx(5 / 15)


def test_exchange_error_rate_predicate_skips_below_min_samples() -> None:
    got = check_exchange_error_rate(successes=0, errors=5, threshold_ratio=0.2, min_samples=10)
    assert got is None


def test_reconciliation_drift_predicate_fires_on_diverging_result() -> None:
    at = datetime(2024, 1, 1, tzinfo=UTC)
    results = [
        ReconciliationResult(
            symbol="BTCUSDT", local_qty=0.3, venue_qty=0.5, delta=0.2, diverges=True, at=at
        ),
        ReconciliationResult(
            symbol="ETHUSDT", local_qty=1.0, venue_qty=1.0, delta=0.0, diverges=False, at=at
        ),
    ]
    got = check_reconciliation_drift(results=results)
    assert got is not None
    assert got[0] == "reconciliation_drift"
    assert got[1]["n_diverging_symbols"] == 1
    assert got[1]["symbols"] == ["BTCUSDT"]


def test_reconciliation_drift_predicate_none_when_all_match() -> None:
    at = datetime(2024, 1, 1, tzinfo=UTC)
    got = check_reconciliation_drift(
        results=[
            ReconciliationResult(
                symbol="BTCUSDT",
                local_qty=0.5,
                venue_qty=0.5,
                delta=0.0,
                diverges=False,
                at=at,
            )
        ]
    )
    assert got is None


def test_health_floor_predicate_trips_below_floor() -> None:
    got = check_health_floor(composite_score=45.0, floor=60.0)
    assert got is not None
    assert got[0] == "ai_health_floor"


def test_health_floor_predicate_none_above_floor() -> None:
    got = check_health_floor(composite_score=75.0, floor=60.0)
    assert got is None


# ---------------------------------------------------------------------------
# Runtime metrics
# ---------------------------------------------------------------------------


def _sample_value(registry, name: str, labels: dict[str, str]) -> float:
    v = registry.get_sample_value(name, labels=labels)
    if v is None:
        raise AssertionError(f"missing sample {name} {labels!r}")
    return v


def test_metrics_trip_increments_counter_and_sets_active_gauge() -> None:
    m = build_runtime_metrics()
    m.observe_kill_switch_trip(reason="data_staleness")
    labels = {"reason": "data_staleness"}
    assert _sample_value(m.registry, "trade_kill_switch_trips_total", labels) == 1.0
    assert _sample_value(m.registry, "trade_kill_switch_active", labels) == 1.0


def test_metrics_clear_flips_active_gauge_back_to_zero() -> None:
    m = build_runtime_metrics()
    m.observe_kill_switch_trip(reason="r")
    m.observe_kill_switch_clear(reason="r")
    assert _sample_value(m.registry, "trade_kill_switch_active", {"reason": "r"}) == 0.0


def test_metrics_reconciler_and_audit_and_health() -> None:
    _ = VenuePosition  # keep import referenced
    m = build_runtime_metrics()
    m.observe_reconciler_iteration()
    m.observe_reconciler_divergence(symbol="BTCUSDT", delta_qty=0.2)
    m.observe_audit_event(kind="signal")
    m.observe_health(composite=82.5, sub_scores={"calibration": 90.0, "operational": 75.0})

    sym = {"symbol": "BTCUSDT"}
    assert _sample_value(m.registry, "trade_reconciler_iterations_total", {}) == 1.0
    assert _sample_value(m.registry, "trade_reconciler_divergences_total", sym) == 1.0
    assert _sample_value(m.registry, "trade_reconciler_last_divergence_qty", sym) == 0.2
    assert _sample_value(m.registry, "trade_audit_events_total", {"kind": "signal"}) == 1.0
    assert _sample_value(m.registry, "trade_ai_health_score", {}) == 82.5
    calib = {"name": "calibration"}
    assert _sample_value(m.registry, "trade_ai_health_sub_score", calib) == 90.0
