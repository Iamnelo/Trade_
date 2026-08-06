"""In-memory ExecutionVenue for paper trading.

Fills orders instantly at the current mark price adjusted by a per-side
slippage bps and a taker fee bps. Tracks positions internally so
`get_positions` returns a coherent view without external I/O.

This is the venue used by the paper-trading orchestrator in Phase 4c.
The MRE's `SimulatedVenue` is a separate concern — it drives the
event-loop-shaped backtest engine; this venue is request/response and
implements the same `ExecutionVenue` Protocol as the live Bybit adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from trade.execution.types import VenueOrderAck, VenuePosition
from trade.mre.types import Fill, Order, Position, Side
from trade.utils.clock import utcnow

_BPS = 10_000.0


MarkFn = Callable[[str], float]


class PaperExecutionVenue:
    def __init__(
        self,
        *,
        mark_fn: MarkFn,
        fee_bps: float = 5.5,
        slippage_bps: float = 5.0,
    ) -> None:
        if fee_bps < 0 or slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be non-negative")
        self._mark_fn = mark_fn
        self._fee_bps = fee_bps
        self._slippage_bps = slippage_bps
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._acked_ids: set[str] = set()
        self._venue_seq = 0

    def fills(self) -> Sequence[Fill]:
        return tuple(self._fills)

    def _mark_to_market(self, order: Order) -> float:
        mark = self._mark_fn(order.symbol)
        if mark <= 0.0:
            raise ValueError(f"non-positive mark for {order.symbol}: {mark}")
        slip_sign = 1 if order.side is Side.BUY else -1
        return mark * (1.0 + slip_sign * self._slippage_bps / _BPS)

    async def submit_order(self, order: Order) -> VenueOrderAck:
        # Idempotency: a repeated client_order_id is a no-op that returns the
        # previously issued ack shape (paper venue only tracks the id set).
        if order.client_order_id in self._acked_ids:
            return VenueOrderAck(
                client_order_id=order.client_order_id,
                venue_order_id=f"paper-{order.client_order_id[:8]}",
                accepted_at=utcnow(),
            )
        self._acked_ids.add(order.client_order_id)

        fill_price = self._mark_to_market(order)
        fee = abs(order.quantity) * fill_price * self._fee_bps / _BPS
        signed_qty = order.quantity if order.side is Side.BUY else -order.quantity

        pos = self._positions.setdefault(order.symbol, Position(symbol=order.symbol))
        pos.quantity += signed_qty

        self._fills.append(
            Fill(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                fee=fee,
                event_time=utcnow(),
            )
        )
        self._venue_seq += 1
        return VenueOrderAck(
            client_order_id=order.client_order_id,
            venue_order_id=f"paper-{self._venue_seq:010d}",
            accepted_at=utcnow(),
        )

    async def cancel_order(self, *, symbol: str, client_order_id: str) -> None:
        # Paper venue fills instantly, so cancel is always a no-op.
        _ = symbol, client_order_id

    async def cancel_all(self, *, symbol: str | None = None) -> int:
        _ = symbol
        return 0

    async def get_positions(self, *, symbol: str | None = None) -> Sequence[VenuePosition]:
        out: list[VenuePosition] = []
        for sym, pos in self._positions.items():
            if symbol is not None and sym != symbol:
                continue
            if pos.quantity == 0.0:
                continue
            mark = self._mark_fn(sym)
            entry = self._last_entry_price(sym)
            out.append(
                VenuePosition(
                    symbol=sym,
                    quantity=pos.quantity,
                    entry_price=entry,
                    unrealized_pnl=pos.quantity * (mark - entry),
                )
            )
        return out

    def _last_entry_price(self, symbol: str) -> float:
        for fill in reversed(self._fills):
            if fill.symbol == symbol:
                return fill.price
        return 0.0
