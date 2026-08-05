"""Live ingest orchestrator: wire the Bybit WS stream into the QuestDB writer,
publish DQ metrics for each bar, and run the REST reconciler in parallel.

Two concurrent tasks share a single event loop:

- `stream_task`: consumes `BybitKlineStream`, writes to `QuestDBKlineWriter`,
  updates staleness/last-event metrics.
- `reconcile_task`: periodically calls `reconcile_once` to close any gaps.

Both tasks respect `stop_event` for a graceful shutdown.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from trade.data.quality.metrics import DQMetrics
from trade.data.questdb.writer import QuestDBKlineWriter
from trade.data.stream.bybit_ws import BybitKlineStream
from trade.data.stream.reconciler import ReconcileTarget, reconcile_once
from trade.exchanges.bybit_public import BybitPublicClient
from trade.logging_setup import get_logger
from trade.utils.clock import utcnow


async def _run_stream(
    *,
    stream: BybitKlineStream,
    writer: QuestDBKlineWriter,
    metrics: DQMetrics,
    stop_event: asyncio.Event,
) -> None:
    log = get_logger("ingest.stream")
    async for batch in stream:
        if stop_event.is_set():
            return
        writer.write(batch)
        now = utcnow()
        for r in batch:
            metrics.observe_message(source=r.source, symbol=r.symbol, interval=r.interval)
            metrics.observe_event_time(
                source=r.source,
                symbol=r.symbol,
                interval=r.interval,
                epoch_seconds=r.event_time.timestamp(),
            )
            metrics.observe_staleness(
                source=r.source,
                symbol=r.symbol,
                interval=r.interval,
                seconds=(now - r.event_time).total_seconds(),
            )
        log.info("stream_batch", written=len(batch))


async def _run_reconciler(
    *,
    client: BybitPublicClient,
    writer: QuestDBKlineWriter,
    reader: Any,
    targets: Sequence[ReconcileTarget],
    metrics: DQMetrics,
    period_seconds: float,
    stop_event: asyncio.Event,
) -> None:
    log = get_logger("ingest.reconciler")
    while not stop_event.is_set():
        try:
            filled = await reconcile_once(
                client=client,
                reader=reader,
                sink=writer,
                targets=targets,
                metrics=metrics,
            )
            if filled:
                log.info("reconciled", bars_written=filled)
        except Exception as exc:  # never let the reconciler die silently
            log.warning("reconcile_failed", error=str(exc))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=period_seconds)
        except TimeoutError:
            continue


async def run_ingest(
    *,
    stream: BybitKlineStream,
    writer: QuestDBKlineWriter,
    reader: Any,
    client: BybitPublicClient,
    targets: Sequence[ReconcileTarget],
    metrics: DQMetrics,
    reconcile_period_seconds: float = 60.0,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the stream + reconciler until `stop_event` is set."""
    stop = stop_event or asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(_run_stream(stream=stream, writer=writer, metrics=metrics, stop_event=stop))
        tg.create_task(
            _run_reconciler(
                client=client,
                writer=writer,
                reader=reader,
                targets=targets,
                metrics=metrics,
                period_seconds=reconcile_period_seconds,
                stop_event=stop,
            )
        )
