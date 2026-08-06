"""Live position reconciler.

Periodically compares the venue's reported positions against the local
OMS view. Emits one `ReconciliationResult` per symbol touched by either
side. A `diverges=True` result means |venue_qty - local_qty| > tolerance
and MUST be fed into the kill-switch controller by the runtime loop.

The reconciler itself does not decide anything — it only reports. That
keeps its testing surface small and lets the orchestrator apply
policy (e.g., "diverges twice in a row => trip kill switch").
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from trade.execution.venue import ExecutionVenue
from trade.utils.clock import utcnow

_DEFAULT_TOLERANCE_QTY = 1e-9


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    symbol: str
    local_qty: float
    venue_qty: float
    delta: float  # venue - local
    diverges: bool
    at: datetime


LocalPositionsFn = Callable[[], dict[str, float]]


class LivePositionReconciler:
    def __init__(
        self,
        *,
        venue: ExecutionVenue,
        local_positions_fn: LocalPositionsFn,
        tolerance_qty: float = _DEFAULT_TOLERANCE_QTY,
    ) -> None:
        if tolerance_qty < 0:
            raise ValueError("tolerance_qty must be non-negative")
        self._venue = venue
        self._local_positions_fn = local_positions_fn
        self._tolerance = tolerance_qty

    async def reconcile_once(self, *, now: datetime | None = None) -> list[ReconciliationResult]:
        at = now or utcnow()
        venue_positions = await self._venue.get_positions()
        venue_qty_by_symbol = {p.symbol: p.quantity for p in venue_positions}
        local_qty_by_symbol = self._local_positions_fn()

        all_symbols = set(venue_qty_by_symbol) | set(local_qty_by_symbol)
        out: list[ReconciliationResult] = []
        for symbol in sorted(all_symbols):
            local = local_qty_by_symbol.get(symbol, 0.0)
            venue = venue_qty_by_symbol.get(symbol, 0.0)
            delta = venue - local
            out.append(
                ReconciliationResult(
                    symbol=symbol,
                    local_qty=local,
                    venue_qty=venue,
                    delta=delta,
                    diverges=abs(delta) > self._tolerance,
                    at=at,
                )
            )
        return out


def divergent(results: Sequence[ReconciliationResult]) -> list[ReconciliationResult]:
    """Convenience: filter to results that violate the tolerance."""
    return [r for r in results if r.diverges]
