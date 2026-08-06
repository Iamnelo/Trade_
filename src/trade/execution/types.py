"""Value types shared across execution venues (paper, live)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class VenueOrderAck:
    """Ack returned by a venue after accepting an order for execution."""

    client_order_id: str
    venue_order_id: str
    accepted_at: datetime


@dataclass(frozen=True, slots=True)
class VenuePosition:
    """Position snapshot as reported by the venue (source of truth)."""

    symbol: str
    quantity: float  # signed; +N long, -N short
    entry_price: float
    unrealized_pnl: float


@dataclass(frozen=True, slots=True)
class SanityGateConfig:
    """Pre-submit checks applied to every outgoing order."""

    max_notional_per_order: float
    max_gross_notional: float
    staleness_seconds: float


class SanityGateViolation(RuntimeError):  # noqa: N818 — domain-specific error name
    """Raised by SanityGate.check when an order fails a pre-submit check.

    Live path treats this as a HARD reject — the order does not go out and
    the OMS must log the rejection with the reason.
    """


class ExchangeError(RuntimeError):
    """Wrapper for venue-side business errors (e.g., Bybit non-zero retCode)."""

    def __init__(self, code: int | str, message: str) -> None:
        super().__init__(f"exchange error code={code!r} message={message!r}")
        self.code = code
        self.message = message


class TransientExchangeError(ExchangeError):
    """Signal to the retry layer that this call should be retried."""
