"""Kill-switch controller.

Halt state is a set of active reasons. Any active reason means the
runtime MUST refuse to open new positions; flatten-and-halt policy is
enforced by the orchestrator (this module is pure state, easy to test).

Trip predicates below are pure functions of runtime state that return
`(reason, detail)` when they fire and `None` when they don't. The
orchestrator threads runtime state (last bar time, error rates,
reconciler divergences, health score) into these predicates each tick
and forwards fires to the controller.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from trade.runtime.reconciler import ReconciliationResult
from trade.utils.clock import utcnow


@dataclass(frozen=True, slots=True)
class KillSwitchTrip:
    reason: str
    detail: dict[str, Any]
    at: datetime


class KillSwitchController:
    def __init__(self) -> None:
        self._active: set[str] = set()
        self._history: list[KillSwitchTrip] = []
        self._reasons_ever: set[str] = set()

    @property
    def is_halted(self) -> bool:
        return bool(self._active)

    @property
    def active_reasons(self) -> tuple[str, ...]:
        return tuple(sorted(self._active))

    @property
    def reasons_ever_triggered(self) -> tuple[str, ...]:
        return tuple(sorted(self._reasons_ever))

    def trip_history(self) -> tuple[KillSwitchTrip, ...]:
        return tuple(self._history)

    def trip(
        self,
        *,
        reason: str,
        detail: dict[str, Any],
        now: datetime | None = None,
    ) -> KillSwitchTrip:
        if not reason:
            raise ValueError("reason must be non-empty")
        trip = KillSwitchTrip(reason=reason, detail=detail, at=now or utcnow())
        self._history.append(trip)
        self._active.add(reason)
        self._reasons_ever.add(reason)
        return trip

    def clear(self, reason: str) -> bool:
        """Clear a reason. Returns True if it had been active."""
        if reason in self._active:
            self._active.remove(reason)
            return True
        return False

    def clear_all(self) -> tuple[str, ...]:
        cleared = tuple(sorted(self._active))
        self._active.clear()
        return cleared


# ---------------------------------------------------------------------------
# Trip predicates — pure functions of inputs; runtime loop calls them.
# ---------------------------------------------------------------------------


def check_data_staleness(
    *, last_bar_time: datetime, now: datetime, threshold_seconds: float
) -> tuple[str, dict[str, Any]] | None:
    if last_bar_time.tzinfo is None or now.tzinfo is None:
        raise ValueError("timestamps must be UTC-aware")
    staleness = (now - last_bar_time).total_seconds()
    if staleness > threshold_seconds:
        return (
            "data_staleness",
            {"seconds": staleness, "threshold_seconds": threshold_seconds},
        )
    return None


def check_exchange_error_rate(
    *,
    successes: int,
    errors: int,
    threshold_ratio: float,
    min_samples: int = 10,
) -> tuple[str, dict[str, Any]] | None:
    total = successes + errors
    if total < min_samples:
        return None
    ratio = errors / total
    if ratio > threshold_ratio:
        return (
            "exchange_error_rate",
            {
                "successes": successes,
                "errors": errors,
                "ratio": ratio,
                "threshold_ratio": threshold_ratio,
            },
        )
    return None


def check_reconciliation_drift(
    *,
    results: Sequence[ReconciliationResult],
) -> tuple[str, dict[str, Any]] | None:
    diverging = [r for r in results if r.diverges]
    if not diverging:
        return None
    return (
        "reconciliation_drift",
        {
            "n_diverging_symbols": len(diverging),
            "symbols": [r.symbol for r in diverging],
            "max_abs_delta": max(abs(r.delta) for r in diverging),
        },
    )


def check_health_floor(
    *,
    composite_score: float,
    floor: float,
) -> tuple[str, dict[str, Any]] | None:
    if composite_score < floor:
        return (
            "ai_health_floor",
            {"composite_score": composite_score, "floor": floor},
        )
    return None
