"""Daily / weekly performance reports for the paper-trading system.

Everything is reconstructed from the journal's `decisions.jsonl` — the single
source of truth — so a report can be regenerated at any time and never
disagrees with what actually happened. No model or trading state is required.

A report covers a window (one UTC day, or one ISO week) and also carries
since-inception cumulative figures. It is rendered to markdown for the reports
directory and summarised in one Telegram-friendly block.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PaperReport:
    period: str  # "daily" | "weekly"
    window_label: str  # e.g. "2026-08-08" or "2026-W32"
    generated_at: str
    n_decisions: int
    n_fills: int
    n_opens: int
    n_exits: int
    n_wins: int
    n_losses: int
    win_rate: float | None
    realized_pnl_window: float
    avg_confidence: float | None
    per_symbol_trades: dict[str, int]
    halts: list[str]
    start_equity: float
    end_equity: float
    window_return_pct: float
    max_drawdown_pct: float
    cumulative_return_pct: float
    cumulative_realized_pnl: float
    open_positions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "window_label": self.window_label,
            "generated_at": self.generated_at,
            "n_decisions": self.n_decisions,
            "n_fills": self.n_fills,
            "n_opens": self.n_opens,
            "n_exits": self.n_exits,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "win_rate": self.win_rate,
            "realized_pnl_window": self.realized_pnl_window,
            "avg_confidence": self.avg_confidence,
            "per_symbol_trades": self.per_symbol_trades,
            "halts": self.halts,
            "start_equity": self.start_equity,
            "end_equity": self.end_equity,
            "window_return_pct": self.window_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "cumulative_return_pct": self.cumulative_return_pct,
            "cumulative_realized_pnl": self.cumulative_realized_pnl,
            "open_positions": self.open_positions,
        }


def load_events(journal_dir: Path) -> list[dict[str, Any]]:
    path = Path(journal_dir) / "decisions.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _window_key(period: str, when: datetime | date) -> str:
    if isinstance(when, datetime):
        when = when.date()
    if period == "weekly":
        iso = when.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return when.isoformat()


def _event_time(ev: dict[str, Any]) -> datetime | None:
    ts = ev.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _in_window(ev: dict[str, Any], period: str, label: str) -> bool:
    et = _event_time(ev)
    if et is None:
        return False
    return _window_key(period, et) == label


def _max_drawdown_pct(equities: Sequence[float]) -> float:
    peak = None
    max_dd = 0.0
    for e in equities:
        if peak is None or e > peak:
            peak = e
        if peak and peak > 0:
            dd = (1.0 - e / peak) * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


def build_report(
    *,
    journal_dir: Path,
    period: str,
    as_of: datetime,
    initial_equity: float,
) -> PaperReport:
    if period not in ("daily", "weekly"):
        raise ValueError(f"period must be 'daily' or 'weekly', got {period!r}")
    events = load_events(journal_dir)
    label = _window_key(period, as_of)
    window = [e for e in events if _in_window(e, period, label)]

    decisions = [e for e in window if e.get("kind") == "decision"]
    fills = [e for e in window if e.get("kind") == "fill"]
    opens = [f for f in fills if f.get("fill_kind") == "OPEN"]
    exits = [f for f in fills if f.get("fill_kind") in ("EXIT", "REVERSE")]
    wins = [f for f in exits if float(f.get("realized_pnl", 0.0)) > 0]
    losses = [f for f in exits if float(f.get("realized_pnl", 0.0)) <= 0]
    win_rate = (len(wins) / len(exits)) if exits else None
    realized_window = sum(float(f.get("realized_pnl", 0.0)) for f in exits)

    confs = [
        float(d["confidence"])
        for d in decisions
        if d.get("confidence") is not None and d.get("status") == "decided"
    ]
    avg_conf = (sum(confs) / len(confs)) if confs else None

    per_symbol: dict[str, int] = {}
    for f in fills:
        sym = str(f.get("symbol", "?"))
        per_symbol[sym] = per_symbol.get(sym, 0) + 1

    halts = [e.get("reason", "?") for e in window if e.get("kind") == "halt"]

    # Equity: fills carry equity; frame the window with the last equity seen
    # before it and the last within it.
    all_equity_events = [e for e in events if "equity" in e and _event_time(e) is not None]
    all_equity_events.sort(key=_event_time)  # type: ignore[arg-type]
    window_equities = [
        float(e["equity"]) for e in all_equity_events if _in_window(e, period, label)
    ]

    # Only prior points that occurred BEFORE the window matter for start equity.
    prior_before = [
        float(e["equity"])
        for e in all_equity_events
        if _event_time(e) is not None and _window_key(period, _event_time(e)) < label  # type: ignore[arg-type]
    ]
    start_equity = prior_before[-1] if prior_before else initial_equity
    end_equity = window_equities[-1] if window_equities else start_equity
    cumulative_equity = (
        float(all_equity_events[-1]["equity"]) if all_equity_events else initial_equity
    )

    window_return = (end_equity / start_equity - 1.0) * 100.0 if start_equity else 0.0
    cumulative_return = (
        (cumulative_equity / initial_equity - 1.0) * 100.0 if initial_equity else 0.0
    )
    cumulative_realized = (
        float(all_equity_events[-1].get("cumulative_realized_pnl", 0.0))
        if all_equity_events
        else 0.0
    )
    dd = _max_drawdown_pct([start_equity, *window_equities])

    return PaperReport(
        period=period,
        window_label=label,
        generated_at=as_of.isoformat(),
        n_decisions=len(decisions),
        n_fills=len(fills),
        n_opens=len(opens),
        n_exits=len(exits),
        n_wins=len(wins),
        n_losses=len(losses),
        win_rate=win_rate,
        realized_pnl_window=realized_window,
        avg_confidence=avg_conf,
        per_symbol_trades=per_symbol,
        halts=halts,
        start_equity=start_equity,
        end_equity=end_equity,
        window_return_pct=window_return,
        max_drawdown_pct=dd,
        cumulative_return_pct=cumulative_return,
        cumulative_realized_pnl=cumulative_realized,
    )


def render_markdown(report: PaperReport) -> str:
    wr = "n/a" if report.win_rate is None else f"{report.win_rate * 100:.1f}%"
    ac = "n/a" if report.avg_confidence is None else f"{report.avg_confidence:.3f}"
    lines = [
        f"# Paper-trading {report.period} report — {report.window_label}",
        "",
        f"_Generated {report.generated_at} — SIMULATED, no real orders._",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Window return | {report.window_return_pct:+.2f}% |",
        f"| Window realized P&L | {report.realized_pnl_window:+.2f} |",
        f"| Max drawdown (window) | {report.max_drawdown_pct:.2f}% |",
        f"| Decisions | {report.n_decisions} |",
        f"| Fills (opens/exits) | {report.n_fills} ({report.n_opens}/{report.n_exits}) |",
        f"| Wins / losses | {report.n_wins} / {report.n_losses} |",
        f"| Win rate | {wr} |",
        f"| Avg confidence | {ac} |",
        f"| Start → end equity | {report.start_equity:.2f} → {report.end_equity:.2f} |",
        f"| Cumulative return | {report.cumulative_return_pct:+.2f}% |",
        f"| Cumulative realized P&L | {report.cumulative_realized_pnl:+.2f} |",
        "",
    ]
    if report.per_symbol_trades:
        by_symbol = ", ".join(f"{s}={n}" for s, n in sorted(report.per_symbol_trades.items()))
        lines.append(f"Fills by symbol: {by_symbol}")
    if report.halts:
        lines.append(f"Halts: {', '.join(report.halts)}")
    return "\n".join(lines) + "\n"


def render_telegram(report: PaperReport) -> str:
    wr = "n/a" if report.win_rate is None else f"{report.win_rate * 100:.0f}%"
    return (
        f"📊 {report.period.upper()} {report.window_label} (SIM)\n"
        f"ret={report.window_return_pct:+.2f}% pnl={report.realized_pnl_window:+.2f} "
        f"dd={report.max_drawdown_pct:.2f}%\n"
        f"trades={report.n_exits} win={wr} equity={report.end_equity:.2f} "
        f"(cum {report.cumulative_return_pct:+.2f}%)"
    )
