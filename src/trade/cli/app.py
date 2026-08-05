"""Top-level typer app that wires subcommands together."""

from __future__ import annotations

import typer

from trade.cli.backfill import backfill_app
from trade.cli.backtest import backtest_app
from trade.cli.ingest import ingest_app

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(backfill_app, name="backfill", help="Historical data backfill commands.")
app.add_typer(ingest_app, name="ingest", help="Live streaming ingest commands.")
app.add_typer(backtest_app, name="backtest", help="Backtest and benchmark suite commands.")
