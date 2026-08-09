"""Realized P&L cost basis, daily/weekly reports, and restart durability."""

from __future__ import annotations

from pathlib import Path

import typer.testing

from tests.test_paper_engine import _engine, _freeze_synth
from trade.cli.app import app
from trade.paper.config import PaperTradingConfig
from trade.paper.engine import PaperTradingEngine
from trade.paper.journal import PaperJournal
from trade.paper.notifier import NullNotifier
from trade.paper.predictor import PaperSymbolBundle
from trade.paper.report import build_report, load_events, render_markdown


def test_cost_basis_realized_pnl_long_round_trip(tmp_path: Path) -> None:
    entry, _ = _freeze_synth(tmp_path)
    engine = _engine(tmp_path, execution_enabled=True, entry=entry)
    # Long 2 @ 100, then close 2 @ 110 => realized +20 (fees zero here).
    r_open = engine._update_cost_basis("BTCUSDT", signed_delta=2.0, price=100.0)
    assert r_open == 0.0
    r_close = engine._update_cost_basis("BTCUSDT", signed_delta=-2.0, price=110.0)
    assert round(r_close, 6) == 20.0


def test_cost_basis_realized_pnl_short_and_reverse(tmp_path: Path) -> None:
    entry, _ = _freeze_synth(tmp_path)
    engine = _engine(tmp_path, execution_enabled=True, entry=entry)
    # Short 1 @ 100, then buy 3 @ 90: closes short (+10) and opens long 2 @ 90.
    engine._update_cost_basis("BTCUSDT", signed_delta=-1.0, price=100.0)
    realized = engine._update_cost_basis("BTCUSDT", signed_delta=3.0, price=90.0)
    assert round(realized, 6) == 10.0
    qty, avg = engine._cost_basis["BTCUSDT"]
    assert round(qty, 6) == 2.0
    assert avg == 90.0


def test_daily_report_from_journal(tmp_path: Path) -> None:
    # Two UTC days of bars => processing the second day emits a daily report
    # for the first, and a report can also be built on demand.
    entry, bars = _freeze_synth(tmp_path, threshold=0.34)
    engine = _engine(tmp_path, execution_enabled=True, entry=entry)
    for bar in bars:
        engine.process_batch([bar])

    events = load_events(tmp_path / "journal")
    assert any(e.get("kind") == "report" for e in events)

    as_of = bars[100].event_time
    rpt = build_report(
        journal_dir=tmp_path / "journal", period="daily", as_of=as_of, initial_equity=10_000.0
    )
    assert rpt.period == "daily"
    md = render_markdown(rpt)
    assert "Paper-trading daily report" in md
    assert "SIMULATED" in md


def test_restart_resumes_positions_and_equity(tmp_path: Path) -> None:
    entry, bars = _freeze_synth(tmp_path, threshold=0.34)
    journal_dir = tmp_path / "journal"

    # First engine processes the first half.
    e1 = _engine(tmp_path, execution_enabled=True, entry=entry)
    for bar in bars[:120]:
        e1.process_batch([bar])
    st1 = e1.state()

    # A fresh engine over the SAME journal/state must resume identically.
    config = PaperTradingConfig(
        execution_enabled=True, symbols=("BTCUSDT",), journal_dir=journal_dir
    )
    bundle = PaperSymbolBundle(entry=entry, repo_root=Path("/"))
    e2 = PaperTradingEngine(
        config=config, bundles=[bundle], journal=PaperJournal(journal_dir), notifier=NullNotifier()
    )
    st2 = e2.state()
    assert round(st2.equity, 6) == round(st1.equity, 6)
    assert st2.positions == st1.positions
    assert st2.n_fills == st1.n_fills


def test_report_cli_smoke(tmp_path: Path) -> None:
    journal = PaperJournal(tmp_path / "j")
    journal.record(
        "fill",
        {"symbol": "BTCUSDT", "side": "buy", "quantity": 1.0, "price": 100.0, "equity": 10000.0},
    )
    result = typer.testing.CliRunner().invoke(
        app,
        ["paper", "report", "--period", "weekly", "--journal-dir", str(tmp_path / "j")],
    )
    assert result.exit_code == 0, result.output
    assert "weekly report" in result.output
