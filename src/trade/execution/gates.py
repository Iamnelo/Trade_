"""Pre-submit sanity gates for outgoing orders.

Every order must pass through `SanityGate.check` before it reaches the
venue. Rejections raise `SanityGateViolation` — the OMS is expected to
log the reason and drop the order (never retry a sanity-gate reject).

V1 checks:

- `max_notional_per_order` — cap on a single order's notional value
  (`quantity * mark_price`).
- `max_gross_notional` — cap on portfolio gross notional including the
  proposed order.
- `staleness_seconds` — reject if the most recent market bar is older
  than this threshold (bad data + trading is worse than not trading).
"""

from __future__ import annotations

from datetime import datetime

from trade.execution.types import SanityGateConfig, SanityGateViolation
from trade.mre.types import Order


class SanityGate:
    def __init__(self, config: SanityGateConfig) -> None:
        self._config = config

    @property
    def config(self) -> SanityGateConfig:
        return self._config

    def check(
        self,
        *,
        order: Order,
        mark_price: float,
        last_bar_time: datetime,
        now: datetime,
        current_gross_notional: float,
    ) -> None:
        if mark_price <= 0.0:
            raise SanityGateViolation(f"non-positive mark price {mark_price}")
        if now.tzinfo is None or last_bar_time.tzinfo is None:
            raise SanityGateViolation("timestamps must be UTC-aware")

        notional = abs(order.quantity) * mark_price
        if notional > self._config.max_notional_per_order:
            raise SanityGateViolation(
                f"per-order notional {notional:.2f} exceeds cap "
                f"{self._config.max_notional_per_order:.2f} "
                f"(symbol={order.symbol}, qty={order.quantity})"
            )

        projected_gross = current_gross_notional + notional
        if projected_gross > self._config.max_gross_notional:
            raise SanityGateViolation(
                f"gross notional would rise to {projected_gross:.2f}, exceeding cap "
                f"{self._config.max_gross_notional:.2f}"
            )

        staleness = (now - last_bar_time).total_seconds()
        if staleness > self._config.staleness_seconds:
            raise SanityGateViolation(
                f"market data stale by {staleness:.1f}s (> {self._config.staleness_seconds:.1f}s)"
            )
