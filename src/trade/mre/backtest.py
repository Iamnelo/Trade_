"""Backtest runner: wires the MRE components into a deterministic loop.

Timing convention:

  for each bar in the source (event_time order):
    1. Clock advances to bar.event_time (bar OPEN).
       Any pending orders decided on previous bars fill at THIS bar's open,
       adjusted for slippage; fees debit cash.
    2. Clock advances to bar.event_time + step (bar CLOSE).
       Equity is marked to bar.close and appended to the equity curve.
       The risk manager updates its HWMs and re-evaluates halts.
    3. Strategy sees `bar` (fully closed by now) and returns TargetPositions.
       The risk manager gates the targets; the OMS diffs against current
       holdings and produces delta orders; the venue queues them for the
       NEXT bar's open. No look-ahead by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

from trade.data.backfill.common import interval_to_timedelta
from trade.mre.oms import OrderManager
from trade.mre.risk import RiskManager
from trade.mre.source import MarketReplaySource
from trade.mre.strategy import Strategy
from trade.mre.types import BacktestConfig, BacktestResult
from trade.mre.venue import SimulatedVenue


def run_backtest(
    *,
    source: MarketReplaySource,
    strategy: Strategy,
    config: BacktestConfig,
    risk: RiskManager | None = None,
    dataset_manifest_ids: Sequence[str] = (),
) -> BacktestResult:
    step = interval_to_timedelta(source.interval)
    venue = SimulatedVenue(fee_bps=config.fee_bps, slippage_bps=config.slippage_bps)
    oms = OrderManager(initial_equity=config.initial_equity)
    rm = risk or RiskManager(initial_equity=config.initial_equity)

    # Tracks the last observed close per symbol so equity marks-to-market are
    # correct across interleaved multi-symbol bars.
    latest_close: dict[str, float] = {}

    for bar in source.iter_bars():
        # 1. Bar OPEN: fill pending orders at this bar's open price.
        source.clock.advance_to(bar.event_time)
        for fill in venue.process_open(
            symbol=bar.symbol,
            bar_open_price=bar.open,
            bar_open_time=bar.event_time,
        ):
            oms.apply_fill(fill)

        # 2. Bar CLOSE: mark to bar.close, record equity, update RM.
        source.clock.advance_to(bar.event_time + step)
        latest_close[bar.symbol] = bar.close
        equity_at_close = oms.equity(latest_close)
        oms.record_equity(bar.event_time + step, latest_close)
        rm.update(bar.event_time + step, equity_at_close)

        # 3. Strategy decides. Risk manager gates. Delta orders queue for next bar.
        snapshot = oms.snapshot(latest_close)
        raw_targets = strategy.on_bar(bar, source, snapshot)
        allowed = rm.gate_targets(raw_targets)
        new_orders = oms.compute_delta_orders(allowed, submit_time=bar.event_time + step)
        venue.submit(new_orders)

    equity_history = oms.equity_history()
    final_equity = equity_history[-1].equity if equity_history else config.initial_equity

    return BacktestResult(
        equity_curve=equity_history,
        fills=oms.fill_history(),
        initial_equity=config.initial_equity,
        final_equity=final_equity,
        config=config,
        strategy_name=strategy.name,
        dataset_manifest_ids=tuple(dataset_manifest_ids),
        halted_reasons_seen=rm.reasons_ever_triggered,
    )
