"""Trade-level metrics computed from a stream of fills.

Bar-level metrics (Sharpe, drawdown, ulcer) live in `performance.py` and
consume an equity curve. This module answers the complementary question:
"of the individual round-trips this strategy took, how many won, by how
much, and how did wins compare to losses?"

A trade is delineated by position-side flips: consecutive fills that keep
the position on the same side count as one trade; a flip closes the
current trade at the flip price and opens a new one on the opposite
side. This matches how a discretionary trader thinks about wins and
losses and produces intuitive expectancy / profit-factor numbers.

Fees on both entry and exit fills are charged against the round-trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from trade.mre.types import Fill, Side


@dataclass(frozen=True, slots=True)
class TradeMetrics:
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_win_pnl: float
    avg_loss_pnl: float
    largest_win_pnl: float
    largest_loss_pnl: float
    expectancy_per_trade: float  # mean PnL per trade in equity units
    profit_factor: float  # sum(winning pnl) / abs(sum(losing pnl)); inf if no losers
    total_pnl: float

    @property
    def expected_return_per_trade_pct(self) -> float:
        """Alias useful when the caller has divided expectancy by initial equity."""
        return self.expectancy_per_trade


def _signed_qty(side: Side, qty: float) -> float:
    return qty if side is Side.BUY else -qty


def compute_trade_metrics(fills: Sequence[Fill]) -> TradeMetrics:
    """Round-trip trade PnLs from a fill stream.

    Position accumulates by side; a flip through zero closes the open
    round-trip at the flip price and opens a fresh one on the new side.
    A trade closed by the final fill (position returned to zero) is
    recorded. A round-trip left open at the end is dropped — we cannot
    know its P&L without a mark price and this module is deliberately
    price-source-free.
    """
    if not fills:
        return _empty_metrics()

    trades_pnl: list[float] = []
    position_qty = 0.0
    avg_entry_price = 0.0
    open_fees = 0.0

    for f in fills:
        signed = _signed_qty(f.side, f.quantity)
        new_qty = position_qty + signed

        # Case 1: adding to an existing side (or opening from zero).
        same_side_or_open = position_qty == 0.0 or (position_qty > 0) == (signed > 0)
        if same_side_or_open:
            if position_qty == 0.0:
                avg_entry_price = f.price
                open_fees = f.fee
            else:
                total_qty_abs = abs(position_qty) + abs(signed)
                avg_entry_price = (
                    avg_entry_price * abs(position_qty) + f.price * abs(signed)
                ) / total_qty_abs
                open_fees += f.fee
            position_qty = new_qty
            continue

        # Case 2: closing all/part of the position (opposite side).
        closing_qty = min(abs(position_qty), abs(signed))
        entry_notional = closing_qty * avg_entry_price
        exit_notional = closing_qty * f.price
        # Fee share proportional to what's being closed.
        fee_share_open = open_fees * (closing_qty / abs(position_qty)) if position_qty else 0.0
        fee_share_close = f.fee * (closing_qty / abs(signed))
        gross_pnl = (
            exit_notional - entry_notional if position_qty > 0 else entry_notional - exit_notional
        )
        trades_pnl.append(gross_pnl - fee_share_open - fee_share_close)
        open_fees -= fee_share_open

        # Anything left on the fill flips to the opposite side and opens a new trade.
        remaining_close_qty = abs(signed) - closing_qty
        position_qty = position_qty + signed if remaining_close_qty > 0 else 0.0
        if remaining_close_qty > 0:
            avg_entry_price = f.price
            open_fees = f.fee - fee_share_close
        else:
            avg_entry_price = 0.0
            open_fees = 0.0

    return _summarise(trades_pnl)


def _summarise(trades_pnl: list[float]) -> TradeMetrics:
    if not trades_pnl:
        return _empty_metrics()
    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p < 0]
    n = len(trades_pnl)
    n_wins = len(wins)
    n_losses = len(losses)
    sum_wins = sum(wins)
    sum_losses = sum(losses)  # negative or zero
    return TradeMetrics(
        n_trades=n,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=n_wins / n,
        avg_win_pnl=sum_wins / n_wins if n_wins else 0.0,
        avg_loss_pnl=sum_losses / n_losses if n_losses else 0.0,
        largest_win_pnl=max(wins) if wins else 0.0,
        largest_loss_pnl=min(losses) if losses else 0.0,
        expectancy_per_trade=sum(trades_pnl) / n,
        profit_factor=(sum_wins / -sum_losses) if sum_losses < 0 else float("inf"),
        total_pnl=sum(trades_pnl),
    )


def _empty_metrics() -> TradeMetrics:
    return TradeMetrics(
        n_trades=0,
        n_wins=0,
        n_losses=0,
        win_rate=0.0,
        avg_win_pnl=0.0,
        avg_loss_pnl=0.0,
        largest_win_pnl=0.0,
        largest_loss_pnl=0.0,
        expectancy_per_trade=0.0,
        profit_factor=0.0,
        total_pnl=0.0,
    )
