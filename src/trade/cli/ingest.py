"""Ingest CLI: run the live Bybit WS ingestor with reconciler and DQ metrics."""

from __future__ import annotations

import asyncio
import signal

import typer
from prometheus_client import start_http_server

from trade.config import get_settings
from trade.data.quality.metrics import build_metrics
from trade.data.questdb.writer import QuestDBKlineWriter, default_connection_factory
from trade.data.stream.bybit_ws import BybitKlineStream
from trade.data.stream.ingest import run_ingest
from trade.data.stream.reconciler import ReconcileTarget, default_reconcile_period
from trade.exchanges.bybit_public import BybitPublicClient
from trade.logging_setup import configure_logging, get_logger

ingest_app = typer.Typer(no_args_is_help=True)


@ingest_app.command("bybit")
def bybit(
    symbols: list[str] = typer.Option(["BTCUSDT", "ETHUSDT"], help="Symbols to subscribe to"),
    interval: str = typer.Option("60", help="Bybit interval (60 = hourly)"),
    category: str = typer.Option("linear"),
    ws_url: str = typer.Option(
        "wss://stream.bybit.com/v5/public/linear",
        help="Bybit v5 public WebSocket URL",
    ),
    questdb_host: str = typer.Option("localhost"),
    questdb_port: int = typer.Option(8812),
    questdb_user: str = typer.Option("admin"),
    questdb_password: str = typer.Option("quest"),
    questdb_db: str = typer.Option("qdb"),
    prometheus_port: int = typer.Option(9464, help="Port to expose /metrics on"),
    ensure_schema: bool = typer.Option(True, help="Apply QuestDB schema.sql on startup"),
) -> None:
    """Run the live Bybit ingest loop until interrupted (Ctrl-C)."""
    configure_logging()
    log = get_logger("cli.ingest.bybit")
    settings = get_settings()

    metrics = build_metrics()
    start_http_server(prometheus_port, registry=metrics.registry)
    log.info("prometheus_listening", port=prometheus_port)

    connection_factory = default_connection_factory(
        host=questdb_host,
        port=questdb_port,
        user=questdb_user,
        password=questdb_password,
        dbname=questdb_db,
    )
    writer = QuestDBKlineWriter(connection_factory)
    if ensure_schema:
        writer.ensure_schema()
        log.info("questdb_schema_applied")

    stream = BybitKlineStream(
        url=ws_url,
        category=category,
        symbols=symbols,
        intervals=[interval],
        on_reconnect=lambda: metrics.observe_reconnect(source="bybit"),
    )
    targets = [ReconcileTarget(symbol=s, interval=interval, category=category) for s in symbols]

    async def _run() -> None:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        async with BybitPublicClient(base_url=settings.bybit_base_url, category=category) as client:
            # A separate connection is used for the reconciler's reads to avoid
            # racing the writer connection on the same cursor.
            reader = QuestDBKlineWriter(connection_factory)
            await run_ingest(
                stream=stream,
                writer=writer,
                reader=reader,
                client=client,
                targets=targets,
                metrics=metrics,
                reconcile_period_seconds=default_reconcile_period(interval).total_seconds(),
                stop_event=stop_event,
            )

    asyncio.run(_run())
