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
from trade.mre.types import Fill, Order, Side, TargetPosition
from trade.mre.venue import SimulatedVenue
from trade.paper.config import PaperTradingConfig
from trade.paper.journal import PaperJournal
from trade.paper.notifier import Notifier, NullNotifier
from trade.paper.predictor import PaperSymbolBundle, SymbolDecision
from trade.paper.report import build_report, load_events, render_markdown, render_telegram
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
        reports_dir: Path | None = None,
        state_path: Path | None = None,
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

        self._reports_dir = reports_dir or (config.journal_dir / "reports")
        self._state_path = state_path or (config.journal_dir / "state.json")

        self._buffers: dict[str, list[KlineRecord]] = {s: [] for s in self._bundles}
        self._latest_close: dict[str, float] = {}
        self._n_decisions = 0
        self._n_fills = 0
        self._last_bar_time: datetime | None = None
        # Signed cost basis per symbol for per-trade realized P&L: (qty, avg_price).
        self._cost_basis: dict[str, tuple[float, float]] = {}
        self._realized_pnl = 0.0
        # Close-time of the previously processed group; a change of UTC day / ISO
        # week versus the incoming group triggers a report for the completed one.
        self._prev_close_time: datetime | None = None

        self._restore_state()

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

        # 4. period-boundary reports + durable state snapshot.
        self._maybe_emit_reports(close_time)
        self._prev_close_time = close_time
        self._snapshot_state()

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
            "reason": decision.reason,
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
        signed = fill.quantity if fill.side is Side.BUY else -fill.quantity
        realized = self._update_cost_basis(fill.symbol, signed_delta=signed, price=fill.price)
        realized -= fill.fee  # net of the fee paid on this fill
        self._realized_pnl += realized

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
                "realized_pnl": realized,
                "cumulative_realized_pnl": self._realized_pnl,
                "equity": equity,
            },
        )
        icon = {"OPEN": "🟢", "EXIT": "🔴", "REVERSE": "🔁", "SCALE": "⚖️"}[kind]
        ret_pct = (equity / self._config.initial_equity - 1.0) * 100
        pnl_line = ""
        if kind in ("EXIT", "REVERSE"):
            pnl_line = f" pnl={realized:+.2f}"
        self._notify(
            f"{icon} {kind} {fill.symbol} {fill.side.value} "
            f"{fill.quantity:.6f} @ {fill.price:.2f}{pnl_line}\n"
            f"pos={new_qty:+.6f} equity={equity:.2f} ({ret_pct:+.2f}%)"
        )

    def _update_cost_basis(self, symbol: str, *, signed_delta: float, price: float) -> float:
        """Update the signed cost basis and return realized P&L (price-based).

        Standard average-cost accounting: adding in the same direction updates
        the average entry; reducing/closing realizes P&L against it; a reverse
        closes the old side and opens the new at `price`.
        """
        qty, avg = self._cost_basis.get(symbol, (0.0, 0.0))
        new_qty = qty + signed_delta
        realized = 0.0
        if abs(qty) < 1e-12 or (qty > 0) == (signed_delta > 0):
            # Opening or scaling in the same direction.
            total = abs(qty) + abs(signed_delta)
            avg = (abs(qty) * avg + abs(signed_delta) * price) / total if total else price
        else:
            # Reducing, closing, or reversing.
            closed = min(abs(qty), abs(signed_delta))
            realized = closed * (price - avg) * (1.0 if qty > 0 else -1.0)
            if abs(signed_delta) > abs(qty):
                avg = price  # reversed onto the new side
            elif abs(new_qty) < 1e-12:
                avg = 0.0
        self._cost_basis[symbol] = (new_qty, avg)
        return realized

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

    # -- reports ---------------------------------------------------------

    def _maybe_emit_reports(self, close_time: datetime) -> None:
        prev = self._prev_close_time
        if prev is None:
            return
        if close_time.date() != prev.date():
            self._emit_report("daily", as_of=prev)
        if close_time.isocalendar()[:2] != prev.isocalendar()[:2]:
            self._emit_report("weekly", as_of=prev)

    def _emit_report(self, period: str, *, as_of: datetime) -> None:
        report = build_report(
            journal_dir=self._config.journal_dir,
            period=period,
            as_of=as_of,
            initial_equity=self._config.initial_equity,
        )
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        (self._reports_dir / f"{period}_{report.window_label}.md").write_text(
            render_markdown(report), encoding="utf-8"
        )
        self._journal.record("report", {"period": period, "window": report.window_label})
        self._notify(render_telegram(report))

    def emit_report_now(self, period: str, *, as_of: datetime | None = None) -> str:
        """Generate a report on demand and return its markdown (also written)."""
        at = as_of or (self._prev_close_time or utcnow())
        report = build_report(
            journal_dir=self._config.journal_dir,
            period=period,
            as_of=at,
            initial_equity=self._config.initial_equity,
        )
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        md = render_markdown(report)
        (self._reports_dir / f"{period}_{report.window_label}.md").write_text(md, encoding="utf-8")
        return md

    # -- durable state ---------------------------------------------------

    def _snapshot_state(self) -> None:
        keep = {s: b.max_lookback + 5 for s, b in self._bundles.items()}
        buffers = {
            s: [self._bar_to_dict(bar) for bar in self._buffers[s][-keep[s] :]]
            for s in self._bundles
        }
        state = {
            "last_bar_time": self._last_bar_time.isoformat() if self._last_bar_time else None,
            "prev_close_time": (
                self._prev_close_time.isoformat() if self._prev_close_time else None
            ),
            "latest_close": self._latest_close,
            "pending_orders": [self._order_to_dict(o) for o in self._venue.pending],
            "buffers": buffers,
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(state, default=str), encoding="utf-8")

    def _restore_state(self) -> None:
        # Rebuild trading state from the journal (the tamper-evident source of
        # truth), so a restart resumes exactly where it left off.
        for ev in load_events(self._config.journal_dir):
            kind = ev.get("kind")
            if kind == "fill":
                fill = self._fill_from_record(ev)
                signed = fill.quantity if fill.side is Side.BUY else -fill.quantity
                realized = self._update_cost_basis(
                    fill.symbol, signed_delta=signed, price=fill.price
                )
                self._realized_pnl += realized - fill.fee
                self._oms.apply_fill(fill)
                self._n_fills += 1
            elif kind == "decision":
                self._n_decisions += 1

        if not self._state_path.exists():
            return
        state = json.loads(self._state_path.read_text(encoding="utf-8"))
        self._latest_close = {k: float(v) for k, v in state.get("latest_close", {}).items()}
        if state.get("last_bar_time"):
            self._last_bar_time = datetime.fromisoformat(state["last_bar_time"])
        if state.get("prev_close_time"):
            self._prev_close_time = datetime.fromisoformat(state["prev_close_time"])
        for sym, bars in state.get("buffers", {}).items():
            if sym in self._buffers:
                self._buffers[sym] = [self._bar_from_dict(b) for b in bars]
        pending = [self._order_from_dict(o) for o in state.get("pending_orders", [])]
        if pending:
            self._venue.submit(pending)

    @staticmethod
    def _bar_to_dict(bar: KlineRecord) -> dict[str, Any]:
        return {
            "symbol": bar.symbol,
            "interval": bar.interval,
            "event_time": bar.event_time.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "turnover": bar.turnover,
        }

    def _bar_from_dict(self, d: dict[str, Any]) -> KlineRecord:
        return KlineRecord(
            source="restored",
            category="linear",
            symbol=d["symbol"],
            interval=d["interval"],
            event_time=datetime.fromisoformat(d["event_time"]),
            ingest_time=datetime.fromisoformat(d["event_time"]),
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            volume=float(d["volume"]),
            turnover=float(d["turnover"]),
        )

    @staticmethod
    def _order_to_dict(order: Order) -> dict[str, Any]:
        return {
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "submit_time": order.submit_time.isoformat(),
        }

    @staticmethod
    def _order_from_dict(d: dict[str, Any]) -> Order:
        return Order(
            client_order_id=d["client_order_id"],
            symbol=d["symbol"],
            side=Side(d["side"]),
            quantity=float(d["quantity"]),
            submit_time=datetime.fromisoformat(d["submit_time"]),
        )

    @staticmethod
    def _fill_from_record(ev: dict[str, Any]) -> Fill:
        ts = datetime.fromisoformat(ev["timestamp"])
        return Fill(
            client_order_id=str(ev.get("client_order_id", "")),
            symbol=str(ev["symbol"]),
            side=Side(str(ev["side"])),
            quantity=float(ev["quantity"]),
            price=float(ev["price"]),
            fee=float(ev.get("fee", 0.0)),
            event_time=ts,
        )

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
