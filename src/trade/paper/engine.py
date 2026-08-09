"""Phase 4c paper-trading engine.

Ties the frozen winners, live data, simulated execution, risk/kill-switch, the
tamper-evident journal, and Telegram notifications into one loop. It reuses the
exact backtest machinery (`SimulatedVenue` + `OrderManager` + `RiskManager`),
so a paper run is consistent with the WFO/forward-test that gate it.

TWO INDEPENDENT SAFETY GUARANTEES
---------------------------------
1. HARD MASTER SWITCH — `config.execution_enabled` (default False). When off,
   the engine consumes bars, predicts, journals every decision, and notifies,
   but NEVER submits a simulated order: no position is ever opened or closed.
2. PAPER-ONLY BY CONSTRUCTION — the engine constructs and uses only the
   in-memory `SimulatedVenue`. There is no code path to a real or testnet
   venue, asserted at construction. Even when the master switch is armed,
   execution is simulated.

Per-bar processing mirrors `run_backtest`:
  1. bar OPEN  — fill orders queued on the previous bar (paper venue).
  2. bar CLOSE — mark equity, update risk-manager drawdown gates.
  3. decide    — strategy emits a target; risk + kill-switch gate it; if (and
                 only if) the master switch is armed, queue the delta order for
                 the next bar's open.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from trade.data.backfill.common import interval_to_timedelta
from trade.data.schemas import KlineRecord
from trade.mre.clock import SimClock
from trade.mre.oms import OrderManager
from trade.mre.risk import RiskManager
from trade.mre.source import MarketReplaySource
from trade.mre.types import Fill, TargetPosition
from trade.mre.venue import SimulatedVenue
from trade.paper.config import PaperTradingConfig
from trade.paper.journal import PaperJournal
from trade.paper.notifier import Notifier, NullNotifier
from trade.paper.predictor import PaperSymbolBundle, SymbolDecision
from trade.runtime.kill_switch import KillSwitchController, check_data_staleness
from trade.utils.clock import utcnow

_FEE_BPS = 5.5
_SLIPPAGE_BPS = 5.0


@dataclass(frozen=True, slots=True)
class EngineState:
    armed: bool
    halted: bool
    halted_reasons: tuple[str, ...]
    equity: float
    cash: float
    positions: dict[str, float]
    initial_equity: float
    last_sha: str
    n_decisions: int
    n_fills: int

    @property
    def total_return_pct(self) -> float:
        return (self.equity / self.initial_equity - 1.0) * 100.0


class PaperTradingEngine:
    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        bundles: Sequence[PaperSymbolBundle],
        journal: PaperJournal,
        notifier: Notifier | None = None,
    ) -> None:
        if not bundles:
            raise ValueError("at least one bundle is required")
        intervals = {b.interval for b in bundles}
        if len(intervals) != 1:
            raise ValueError(f"all winners must share one interval, got {sorted(intervals)}")
        self._config = config
        self._bundles = {b.symbol: b for b in bundles}
        self._journal = journal
        self._notifier = notifier or NullNotifier()
        self._interval = next(iter(intervals))
        self._step = interval_to_timedelta(self._interval)

        # Reused backtest machinery — exact parity with the gated evaluation.
        self._venue = SimulatedVenue(fee_bps=_FEE_BPS, slippage_bps=_SLIPPAGE_BPS)
        self._oms = OrderManager(initial_equity=config.initial_equity)
        self._risk = RiskManager(
            initial_equity=config.initial_equity,
            daily_pct=config.daily_drawdown_pct,
            weekly_pct=config.weekly_drawdown_pct,
            monthly_pct=config.monthly_drawdown_pct,
        )
        self._kill = KillSwitchController()

        self._assert_paper_only()

        self._buffers: dict[str, list[KlineRecord]] = {s: [] for s in self._bundles}
        self._latest_close: dict[str, float] = {}
        self._n_decisions = 0
        self._n_fills = 0
        self._last_bar_time: datetime | None = None

    # -- safety ----------------------------------------------------------

    def _assert_paper_only(self) -> None:
        # Structural guarantee: only ever the in-memory simulated venue.
        if type(self._venue) is not SimulatedVenue:  # pragma: no cover - defensive
            raise RuntimeError(
                "paper engine may only use SimulatedVenue; refusing to run against "
                f"{type(self._venue).__name__}"
            )

    @property
    def armed(self) -> bool:
        return self._config.execution_enabled

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        mode = "ARMED (simulated execution ON)" if self.armed else "DISABLED (observe-only)"
        payload = {
            "mode": mode,
            "execution_enabled": self.armed,
            "symbols": sorted(self._bundles),
            "initial_equity": self._config.initial_equity,
            "winners": {s: b.reproducibility_hash[:12] for s, b in self._bundles.items()},
        }
        self._journal.record("engine_start", payload)
        self._notify(
            f"📟 Paper trading START — {mode}\n"
            f"symbols={sorted(self._bundles)} equity={self._config.initial_equity:.0f}"
        )

    def stop(self) -> None:
        st = self.state()
        self._journal.record(
            "engine_stop",
            {"equity": st.equity, "total_return_pct": st.total_return_pct, "n_fills": st.n_fills},
        )
        self._notify(
            f"🛑 Paper trading STOP — equity={st.equity:.2f} "
            f"({st.total_return_pct:+.2f}%) fills={st.n_fills}"
        )

    async def run(self, feed: AsyncIterable[list[KlineRecord]]) -> None:
        self.start()
        try:
            async for batch in feed:
                self.process_batch(batch)
        finally:
            self.stop()

    # -- core loop -------------------------------------------------------

    def process_batch(self, batch: Sequence[KlineRecord]) -> None:
        """Group a batch by event_time and process each group in order."""
        relevant = [b for b in batch if b.symbol in self._bundles and b.interval == self._interval]
        relevant.sort(key=lambda b: (b.event_time, b.symbol))
        i = 0
        while i < len(relevant):
            j = i
            while j < len(relevant) and relevant[j].event_time == relevant[i].event_time:
                j += 1
            self.process_bar_group(relevant[i:j])
            i = j

    def process_bar_group(self, group: Sequence[KlineRecord]) -> None:
        if not group:
            return
        event_time = group[0].event_time
        for bar in group:
            self._buffers[bar.symbol].append(bar)
        self._last_bar_time = event_time

        # 1. bar OPEN: fill orders queued on the previous group at this open.
        for bar in group:
            for fill in self._venue.process_open(
                symbol=bar.symbol, bar_open_price=bar.open, bar_open_time=bar.event_time
            ):
                self._settle_fill(fill)

        # 2. bar CLOSE: mark, record equity, update drawdown gates.
        close_time = event_time + self._step
        for bar in group:
            self._latest_close[bar.symbol] = bar.close
        equity = self._oms.equity(self._latest_close)
        self._oms.record_equity(close_time, self._latest_close)
        self._risk.update(close_time, equity)
        self._sync_drawdown_halt()

        # 3. decide per symbol; gate; (only if armed) queue orders.
        source = self._build_source(close_time)
        for bar in group:
            bundle = self._bundles[bar.symbol]
            snapshot = self._oms.snapshot(self._latest_close)
            decision = bundle.decide(source=source, bar=bar, snapshot=snapshot)
            self._handle_decision(bundle, decision, submit_time=close_time)

    # -- helpers ---------------------------------------------------------

    def _build_source(self, now: datetime) -> MarketReplaySource:
        all_bars: list[KlineRecord] = []
        for buf in self._buffers.values():
            all_bars.extend(buf)
        return MarketReplaySource(bars=all_bars, clock=SimClock(now), interval=self._interval)

    def _sync_drawdown_halt(self) -> None:
        for reason in self._risk.halted_reasons:
            if reason not in self._kill.active_reasons:
                trip = self._kill.trip(reason=reason, detail={"source": "risk_manager"})
                self._journal.record("halt", {"reason": reason, "detail": trip.detail})
                self._notify(f"⛔ HALT: {reason} — new positions blocked, flatten-only")
        # Clear kill-switch drawdown reasons the risk manager has released.
        for reason in ("daily_dd", "weekly_dd", "monthly_dd"):
            if reason in self._kill.active_reasons and reason not in self._risk.halted_reasons:
                self._kill.clear(reason)
                self._journal.record("halt_cleared", {"reason": reason})

    def _handle_decision(
        self,
        bundle: PaperSymbolBundle,
        decision: SymbolDecision,
        *,
        submit_time: datetime,
    ) -> None:
        self._n_decisions += 1
        halted = self._kill.is_halted or self._risk.is_halted
        record: dict[str, Any] = {
            "symbol": decision.symbol,
            "event_time": decision.event_time.isoformat(),
            "close": decision.close,
            "status": decision.status,
            "direction": decision.direction,
            "chosen_class": decision.chosen_class,
            "confidence": decision.confidence,
            "meets_threshold": decision.meets_threshold,
            "target_qty": decision.target_qty,
            "probs": decision.probs,
            "threshold": bundle.threshold,
            "execution_enabled": self.armed,
            "halted": halted,
        }

        if decision.status == "warmup":
            record["action"] = "WARMUP"
            self._journal.record("decision", record)
            return

        # Gate the target through the risk manager (flatten-only when halted).
        raw_target = TargetPosition(symbol=decision.symbol, target_qty=decision.target_qty or 0.0)
        allowed = self._risk.gate_targets([raw_target])
        if halted:
            allowed = [t for t in allowed if t.target_qty == 0.0]

        if not self.armed:
            record["action"] = "OBSERVE_ONLY"
            record["note"] = "execution disabled by master switch"
            self._journal.record("decision", record)
            self._notify(self._decision_line(decision, action="OBSERVE_ONLY", halted=halted))
            return

        orders = self._oms.compute_delta_orders(allowed, submit_time=submit_time)
        if orders:
            self._venue.submit(orders)
            record["action"] = "ORDER_QUEUED"
            record["orders"] = [
                {"side": o.side.value, "qty": o.quantity, "cid": o.client_order_id} for o in orders
            ]
        else:
            record["action"] = "HOLD"
        self._journal.record("decision", record)
        self._notify(self._decision_line(decision, action=record["action"], halted=halted))

    def _settle_fill(self, fill: Fill) -> None:
        prev_qty = self._oms.position_qty(fill.symbol)
        self._oms.apply_fill(fill)
        new_qty = self._oms.position_qty(fill.symbol)
        self._n_fills += 1

        if abs(prev_qty) < 1e-12 and abs(new_qty) > 1e-12:
            kind = "OPEN"
        elif abs(prev_qty) > 1e-12 and abs(new_qty) < 1e-12:
            kind = "EXIT"
        elif prev_qty * new_qty < 0:
            kind = "REVERSE"
        else:
            kind = "SCALE"

        equity = self._oms.equity(self._latest_close)
        self._journal.record(
            "fill",
            {
                "fill_kind": kind,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.price,
                "fee": fill.fee,
                "prev_qty": prev_qty,
                "new_qty": new_qty,
                "equity": equity,
            },
        )
        icon = {"OPEN": "🟢", "EXIT": "🔴", "REVERSE": "🔁", "SCALE": "⚖️"}[kind]
        ret_pct = (equity / self._config.initial_equity - 1.0) * 100
        self._notify(
            f"{icon} {kind} {fill.symbol} {fill.side.value} "
            f"{fill.quantity:.6f} @ {fill.price:.2f}\n"
            f"pos={new_qty:+.6f} equity={equity:.2f} ({ret_pct:+.2f}%)"
        )

    def check_staleness(self, *, now: datetime | None = None) -> None:
        """Trip the kill switch if confirmed bars have stopped arriving."""
        if self._last_bar_time is None:
            return
        fired = check_data_staleness(
            last_bar_time=self._last_bar_time,
            now=now or utcnow(),
            threshold_seconds=self._config.data_staleness_kill_seconds,
        )
        if fired is not None and fired[0] not in self._kill.active_reasons:
            trip = self._kill.trip(reason=fired[0], detail=fired[1])
            self._journal.record("halt", {"reason": fired[0], "detail": trip.detail})
            self._notify(f"⛔ HALT: data staleness — {fired[1]['seconds']:.0f}s since last bar")

    def _decision_line(self, d: SymbolDecision, *, action: str, halted: bool) -> str:
        conf = f"{d.confidence:.3f}" if d.confidence is not None else "n/a"
        flag = " [HALTED]" if halted else ""
        return (
            f"🧠 {d.symbol} {d.close:.2f} → {d.direction.upper()} "
            f"(p={conf}, θ={self._bundles[d.symbol].threshold}) {action}{flag}"
        )

    def _notify(self, text: str) -> None:
        # Notifier implementations never raise; still guard defensively so a
        # bug in a custom notifier cannot break the trading loop.
        with contextlib.suppress(Exception):
            self._notifier.notify(text)

    # -- introspection ---------------------------------------------------

    def state(self) -> EngineState:
        halted_reasons = tuple(
            sorted(set(self._kill.active_reasons) | set(self._risk.halted_reasons))
        )
        return EngineState(
            armed=self.armed,
            halted=self._kill.is_halted or self._risk.is_halted,
            halted_reasons=halted_reasons,
            equity=self._oms.equity(self._latest_close),
            cash=self._oms.cash,
            positions={s: self._oms.position_qty(s) for s in self._bundles},
            initial_equity=self._config.initial_equity,
            last_sha=self._journal.last_sha,
            n_decisions=self._n_decisions,
            n_fills=self._n_fills,
        )


def load_bundles(config: PaperTradingConfig, *, repo_root: Path) -> list[PaperSymbolBundle]:
    """Build one bundle per configured symbol from the freeze manifest."""
    manifest = json.loads(config.manifest_path.read_text())
    by_symbol = {w["symbol"]: w for w in manifest["winners"]}
    bundles: list[PaperSymbolBundle] = []
    for symbol in config.symbols:
        if symbol not in by_symbol:
            raise ValueError(f"symbol {symbol} not in freeze manifest {config.manifest_path}")
        bundles.append(PaperSymbolBundle(entry=by_symbol[symbol], repo_root=repo_root))
    return bundles
