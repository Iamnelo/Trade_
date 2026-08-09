"""Paper-trading CLI (Phase 4c).

Commands:
  run     — start the paper engine against live Bybit data or a replay CSV.
  status  — summarise the journal (equity, counts) and verify its sha chain.
  verify  — verify the journal's tamper-evident audit chain.

Execution is DISABLED by default. Arming simulated execution requires BOTH the
``--arm-execution`` flag AND ``--confirm "ARM PAPER EXECUTION"`` so it can never
be turned on by a stray flag. Even when armed, execution is simulated only —
the engine has no path to a real or testnet venue.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from trade.data.stream.bybit_ws import BybitKlineStream
from trade.paper.config import PaperTradingConfig, TelegramConfig
from trade.paper.engine import PaperTradingEngine, load_bundles
from trade.paper.feed import ReplayFeed
from trade.paper.journal import PaperJournal
from trade.paper.notifier import build_notifier
from trade.research.runner import load_klines_csv

paper_app = typer.Typer(no_args_is_help=True)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARM_PHRASE = "ARM PAPER EXECUTION"


def _resolve_execution(arm_execution: bool, confirm: str | None) -> bool:
    if not arm_execution:
        return False
    if confirm != _ARM_PHRASE:
        raise typer.BadParameter(
            f'to arm simulated execution you must pass --confirm "{_ARM_PHRASE}" '
            "exactly. Refusing to arm."
        )
    return True


def _banner(config: PaperTradingConfig) -> None:
    mode = (
        "ARMED (simulated execution ON)" if config.execution_enabled else "DISABLED (observe-only)"
    )
    typer.echo("=" * 68)
    typer.echo(f"  PAPER TRADING — execution {mode}")
    typer.echo("  Simulated only. No real or testnet orders are ever placed.")
    typer.echo(f"  symbols={list(config.symbols)}  manifest={config.manifest_path.name}")
    typer.echo(f"  journal={config.journal_dir}")
    if config.execution_enabled:
        typer.echo("  Reminder: only arm after the forward-test gates pass + approval.")
    typer.echo("=" * 68)


@paper_app.command("run")
def run(
    replay_csv: list[Path] = typer.Option(
        None, "--replay", help="CSV(s) of daily bars for an offline dry run (no network)."
    ),
    symbols: list[str] = typer.Option(None, "--symbol", help="Symbols; default = manifest."),
    manifest_path: Path = typer.Option(
        _REPO_ROOT / "artifacts" / "frozen" / "freeze_manifest.json", "--manifest"
    ),
    journal_dir: Path = typer.Option(_REPO_ROOT / "paper_journal", "--journal-dir"),
    initial_equity: float = typer.Option(10_000.0, "--initial-equity"),
    arm_execution: bool = typer.Option(
        False, "--arm-execution", help="Enable SIMULATED execution (still no real orders)."
    ),
    confirm: str = typer.Option(None, "--confirm", help=f'Must equal "{_ARM_PHRASE}" to arm.'),
    ws_url: str = typer.Option("wss://stream.bybit.com/v5/public/linear", "--ws-url"),
    max_reconnects: int = typer.Option(None, "--max-reconnects"),
) -> None:
    execution_enabled = _resolve_execution(arm_execution, confirm)
    manifest = json.loads(manifest_path.read_text())
    manifest_symbols = tuple(w["symbol"] for w in manifest["winners"])
    chosen = tuple(symbols) if symbols else manifest_symbols

    config = PaperTradingConfig(
        execution_enabled=execution_enabled,
        manifest_path=manifest_path,
        symbols=chosen,
        initial_equity=initial_equity,
        journal_dir=journal_dir,
        telegram=TelegramConfig.from_env(),
    )
    _banner(config)

    bundles = load_bundles(config, repo_root=_REPO_ROOT)
    journal = PaperJournal(config.journal_dir)
    notifier = build_notifier(config.telegram)
    engine = PaperTradingEngine(config=config, bundles=bundles, journal=journal, notifier=notifier)

    interval = bundles[0].interval
    if replay_csv:
        bars = []
        for csv_path, bundle in zip(replay_csv, bundles, strict=False):
            bars.extend(load_klines_csv(csv_path, symbol=bundle.symbol, interval=bundle.interval))
        typer.echo(f"replay: {len(bars)} bars across {len(replay_csv)} file(s)")
        asyncio.run(engine.run(ReplayFeed(bars)))
    else:
        stream = BybitKlineStream(
            url=ws_url,
            symbols=list(chosen),
            intervals=[interval],
            max_reconnects=max_reconnects,
        )
        typer.echo(f"connecting live WS {ws_url} topics={stream.topics()}")
        asyncio.run(engine.run(stream))

    st = engine.state()
    typer.echo(
        f"done. equity={st.equity:.2f} ({st.total_return_pct:+.2f}%) "
        f"decisions={st.n_decisions} fills={st.n_fills} halted={st.halted}"
    )


@paper_app.command("status")
def status(
    journal_dir: Path = typer.Option(_REPO_ROOT / "paper_journal", "--journal-dir"),
    tail: int = typer.Option(10, "--tail", help="Show the last N journal lines."),
) -> None:
    decisions = journal_dir / "decisions.jsonl"
    if not decisions.exists():
        typer.echo(f"no journal at {decisions}")
        raise typer.Exit(code=1)
    lines = [line for line in decisions.read_text().splitlines() if line.strip()]
    last_equity = None
    for line in lines:
        rec = json.loads(line)
        if "equity" in rec:
            last_equity = rec["equity"]
    typer.echo(f"journal: {len(lines)} events  last_equity={last_equity}")
    for line in lines[-tail:]:
        rec = json.loads(line)
        typer.echo(f"  {rec.get('timestamp', '')} {rec.get('kind', '')} {rec.get('symbol', '')}")
    PaperJournal(journal_dir).verify()
    typer.echo("audit chain: OK")


@paper_app.command("verify")
def verify(
    journal_dir: Path = typer.Option(_REPO_ROOT / "paper_journal", "--journal-dir"),
) -> None:
    PaperJournal(journal_dir).verify()
    typer.echo("audit chain: OK")
