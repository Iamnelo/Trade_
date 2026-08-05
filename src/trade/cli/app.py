"""Top-level typer app that wires subcommands together."""

from __future__ import annotations

import typer

from trade.cli.backfill import backfill_app

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(backfill_app, name="backfill", help="Historical data backfill commands.")
