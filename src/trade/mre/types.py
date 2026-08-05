"""Value types for the market replay engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """Declarative signed position target. The OMS diffs against the current
    position to produce a delta order — idempotent by construction, so a
    strategy re-emitting the same target in consecutive bars produces no
    duplicate orders.
    """

    symbol: str
    target_qty: float


@dataclass(frozen=True, slots=True)
class Order:
    client_order_id: str
    symbol: str
    side: Side
    quantity: float
    submit_time: datetime


@dataclass(frozen=True, slots=True)
class Fill:
    client_order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    fee: float
    event_time: datetime  # bar-open time the fill executed at


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: float = 0.0  # signed; +N long, -N short


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: datetime
    equity: float


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_equity: float
    fee_bps: float = 5.5  # Bybit USDT-perp taker default
    slippage_bps: float = 5.0
    max_gross_notional_fraction: float = 1.0  # portfolio gross <= this * equity


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    cash: float
    positions: Mapping[str, float]  # symbol -> signed qty
    marks: Mapping[str, float]  # symbol -> latest mark price

    @property
    def equity(self) -> float:
        e = self.cash
        for symbol, qty in self.positions.items():
            if qty == 0.0:
                continue
            mark = self.marks.get(symbol)
            if mark is None:
                continue
            e += qty * mark
        return e

    @property
    def gross_notional(self) -> float:
        g = 0.0
        for symbol, qty in self.positions.items():
            mark = self.marks.get(symbol)
            if mark is None:
                continue
            g += abs(qty * mark)
        return g


def freeze_mapping(m: Mapping[str, float]) -> Mapping[str, float]:
    """Return an immutable view of a str->float mapping (for PortfolioSnapshot)."""
    return MappingProxyType(dict(m))


@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity_curve: tuple[EquityPoint, ...]
    fills: tuple[Fill, ...]
    initial_equity: float
    final_equity: float
    config: BacktestConfig
    strategy_name: str
    dataset_manifest_ids: tuple[str, ...] = field(default_factory=tuple)
    halted_reasons_seen: tuple[str, ...] = field(default_factory=tuple)
