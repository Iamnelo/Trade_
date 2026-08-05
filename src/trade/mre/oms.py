"""Order Manager: position tracking, equity bookkeeping, target -> delta orders.

Bookkeeping model: spot-like. Cash decreases on BUY (or increases on short
SELL) by signed_qty * fill_price; fee always debits cash. Equity is
`cash + sum(qty * mark_price)`. This is exact for max_gross_notional <= 1x
equity and slightly over-constrains capital vs. real perpetual margining
above 1x — acceptable for V1, which does not run leverage above 1x.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from trade.mre.types import (
    EquityPoint,
    Fill,
    Order,
    Position,
    Side,
    TargetPosition,
    freeze_mapping,
)
from trade.mre.types import PortfolioSnapshot as _PortfolioSnapshot

_MIN_DELTA_QTY = 1e-12


class OrderManager:
    def __init__(self, *, initial_equity: float) -> None:
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        self._initial_equity = initial_equity
        self._cash = initial_equity
        self._positions: dict[str, Position] = {}
        self._equity_history: list[EquityPoint] = []
        self._fill_history: list[Fill] = []
        self._order_seq = 0

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def initial_equity(self) -> float:
        return self._initial_equity

    def positions(self) -> dict[str, Position]:
        return {
            s: Position(symbol=p.symbol, quantity=p.quantity) for s, p in self._positions.items()
        }

    def position_qty(self, symbol: str) -> float:
        pos = self._positions.get(symbol)
        return pos.quantity if pos is not None else 0.0

    def equity_history(self) -> tuple[EquityPoint, ...]:
        return tuple(self._equity_history)

    def fill_history(self) -> tuple[Fill, ...]:
        return tuple(self._fill_history)

    def equity(self, marks: Mapping[str, float]) -> float:
        e = self._cash
        for symbol, pos in self._positions.items():
            if pos.quantity == 0.0:
                continue
            mark = marks.get(symbol)
            if mark is None:
                continue
            e += pos.quantity * mark
        return e

    def snapshot(self, marks: Mapping[str, float]) -> _PortfolioSnapshot:
        return _PortfolioSnapshot(
            cash=self._cash,
            positions=freeze_mapping({s: p.quantity for s, p in self._positions.items()}),
            marks=freeze_mapping(marks),
        )

    def apply_fill(self, fill: Fill) -> None:
        side_sign = 1.0 if fill.side is Side.BUY else -1.0
        signed_qty = fill.quantity * side_sign
        pos = self._positions.setdefault(fill.symbol, Position(symbol=fill.symbol))
        pos.quantity += signed_qty
        self._cash -= signed_qty * fill.price
        self._cash -= fill.fee
        self._fill_history.append(fill)

    def record_equity(self, timestamp: datetime, marks: Mapping[str, float]) -> None:
        self._equity_history.append(EquityPoint(timestamp=timestamp, equity=self.equity(marks)))

    def compute_delta_orders(
        self,
        targets: Sequence[TargetPosition],
        *,
        submit_time: datetime,
    ) -> list[Order]:
        orders: list[Order] = []
        for target in targets:
            current = self._positions.get(target.symbol, Position(symbol=target.symbol)).quantity
            delta = target.target_qty - current
            if abs(delta) < _MIN_DELTA_QTY:
                continue
            side = Side.BUY if delta > 0 else Side.SELL
            self._order_seq += 1
            orders.append(
                Order(
                    client_order_id=f"{target.symbol}|{submit_time.isoformat()}|{self._order_seq}",
                    symbol=target.symbol,
                    side=side,
                    quantity=abs(delta),
                    submit_time=submit_time,
                )
            )
        return orders
