"""Simulated execution venue.

Orders submitted during bar N fill at bar N+1's OPEN, adjusted by a per-side
slippage bps and a taker fee bps. This is the industry-standard, look-ahead-
safe convention for bar-cadence backtests. The same interface will be
implemented by `BybitExecutionVenue` (Phase 4) so the strategy code path is
identical across replay / paper / live.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from trade.mre.types import Fill, Order, Side

_BPS = 10_000.0


class SimulatedVenue:
    def __init__(self, *, fee_bps: float, slippage_bps: float) -> None:
        if fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        self._fee_bps = fee_bps
        self._slippage_bps = slippage_bps
        self._pending: list[Order] = []

    @property
    def pending(self) -> tuple[Order, ...]:
        return tuple(self._pending)

    def submit(self, orders: Iterable[Order]) -> None:
        for order in orders:
            self._pending.append(order)

    def process_open(
        self,
        *,
        symbol: str,
        bar_open_price: float,
        bar_open_time: datetime,
    ) -> list[Fill]:
        """Fill all pending orders for `symbol` at the given open price."""
        if bar_open_price <= 0:
            raise ValueError("bar_open_price must be positive")
        remaining: list[Order] = []
        fills: list[Fill] = []
        for order in self._pending:
            if order.symbol != symbol:
                remaining.append(order)
                continue
            slippage_sign = 1 if order.side is Side.BUY else -1
            fill_price = bar_open_price * (1.0 + slippage_sign * self._slippage_bps / _BPS)
            fee = order.quantity * fill_price * self._fee_bps / _BPS
            fills.append(
                Fill(
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    price=fill_price,
                    fee=fee,
                    event_time=bar_open_time,
                )
            )
        self._pending = remaining
        return fills

    def cancel_all(self) -> int:
        n = len(self._pending)
        self._pending.clear()
        return n
