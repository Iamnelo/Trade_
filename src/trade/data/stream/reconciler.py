"""REST reconciler: fill gaps in QuestDB by re-fetching bars from Bybit REST.

Runs periodically. For each configured (symbol, interval), it:

1. Reads the newest `latest_event_times` from QuestDB.
2. Uses those to find missing bars over a bounded look-back window ending at
   the most recent CLOSED bar boundary.
3. Fetches only the missing bars via the REST client.
4. Writes them back to QuestDB (DEDUP UPSERT handles overlap safely).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from trade.data.backfill.common import interval_to_timedelta
from trade.data.quality.checks import find_missing_bars
from trade.data.quality.metrics import DQMetrics
from trade.data.schemas import KlineRecord
from trade.exchanges.bybit_public import BybitPublicClient


@dataclass(frozen=True, slots=True)
class ReconcileTarget:
    symbol: str
    interval: str
    category: str = "linear"


class QuestDBReader(Protocol):
    """Minimal read surface the reconciler needs. Any object that quacks like this works."""

    def latest_event_times(
        self, *, source: str, symbol: str, interval: str, limit: int
    ) -> list[datetime]: ...


class QuestDBSink(Protocol):
    def write(self, records: Sequence[KlineRecord]) -> int: ...


def _floor_to_interval(now: datetime, interval: str) -> datetime:
    """Truncate `now` back to the most recent bar boundary at `interval`."""
    step = interval_to_timedelta(interval)
    step_seconds = int(step.total_seconds())
    epoch = int(now.replace(tzinfo=UTC).timestamp())
    floored = epoch - (epoch % step_seconds)
    return datetime.fromtimestamp(floored, tz=UTC)


async def reconcile_once(
    *,
    client: BybitPublicClient,
    reader: QuestDBReader,
    sink: QuestDBSink,
    targets: Sequence[ReconcileTarget],
    metrics: DQMetrics,
    lookback_bars: int = 240,
    now: datetime | None = None,
) -> int:
    """Run a single reconciliation pass across all targets. Returns bars written."""
    now_ts = now or datetime.now(UTC)
    total_written = 0
    for target in targets:
        step = interval_to_timedelta(target.interval)
        window_end = _floor_to_interval(now_ts, target.interval)
        window_start = window_end - step * lookback_bars

        try:
            observed = reader.latest_event_times(
                source="bybit",
                symbol=target.symbol,
                interval=target.interval,
                limit=lookback_bars,
            )
            missing = find_missing_bars(
                observed,
                interval=target.interval,
                window_start=window_start,
                window_end=window_end,
            )
            if not missing:
                continue

            fetched = await client.fetch_klines(
                target.symbol,
                target.interval,
                start=min(missing),
                end=max(missing) + step,
                limit=min(1000, len(missing) + 5),
            )
            missing_set = set(missing)
            ingest_time = datetime.now(UTC)
            records = [
                KlineRecord(
                    source="bybit",
                    category=target.category,
                    symbol=target.symbol,
                    interval=target.interval,
                    event_time=k.open_time,
                    ingest_time=ingest_time,
                    open=k.open,
                    high=k.high,
                    low=k.low,
                    close=k.close,
                    volume=k.volume,
                    turnover=k.turnover,
                )
                for k in fetched
                if k.open_time in missing_set
            ]
            written = sink.write(records)
            total_written += written
            for _ in records:
                metrics.observe_reconciler_fill(
                    source="bybit", symbol=target.symbol, interval=target.interval
                )
        except Exception:
            metrics.observe_reconciler_error(source="bybit")
            raise
    return total_written


def default_lookback_seconds(interval: str, lookback_bars: int = 240) -> float:
    return interval_to_timedelta(interval).total_seconds() * lookback_bars


def default_reconcile_period(interval: str) -> timedelta:
    """Reconcile roughly at 1/12 of the bar interval, clamped to [30s, 600s]."""
    step = interval_to_timedelta(interval)
    seconds = step.total_seconds() / 12
    return timedelta(seconds=max(30.0, min(seconds, 600.0)))
