"""Market Replay Source: PIT-correct historical kline stream.

Holds a sorted view of `KlineRecord` events and a `SimClock`. Exposes:

- `iter_bars()`: yields bars in event_time order (ties broken by symbol for
  determinism). The runner drives clock advances externally so a strategy
  can consult `history(...)` at bar-close time.
- `history(symbol, interval, lookback)`: returns the last `lookback` bars
  for that stream whose event_time + interval <= sim_clock.now (i.e., bars
  whose close has already occurred by the current sim time). This is the
  ONLY read path — there is no method that returns "everything, filter later".

All checks encode PIT-correctness at the API surface; a leaky query is not
just discouraged but impossible.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import datetime

from trade.data.backfill.common import interval_to_timedelta
from trade.data.manifest import DatasetManifest
from trade.data.parquet import parquet_bytes_to_dataframe
from trade.data.schemas import KlineRecord
from trade.data.storage import Store
from trade.mre.clock import SimClock


class MarketReplaySource:
    def __init__(
        self,
        *,
        bars: Sequence[KlineRecord],
        clock: SimClock,
        interval: str,
    ) -> None:
        for b in bars:
            if b.interval != interval:
                raise ValueError(
                    f"MarketReplaySource: expected interval={interval!r} "
                    f"but got a bar with interval={b.interval!r} for {b.symbol}"
                )
        # Sort by event_time; break ties on symbol so replay is deterministic
        # even if the underlying storage returns rows in a different order.
        self._bars: list[KlineRecord] = sorted(bars, key=lambda b: (b.event_time, b.symbol))
        self._clock = clock
        self._interval = interval
        self._step = interval_to_timedelta(interval)
        self._by_key: dict[tuple[str, str], list[KlineRecord]] = defaultdict(list)
        for b in self._bars:
            self._by_key[(b.symbol, b.interval)].append(b)
        self._symbols = tuple(sorted({b.symbol for b in self._bars}))

    @property
    def clock(self) -> SimClock:
        return self._clock

    @property
    def interval(self) -> str:
        return self._interval

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def __len__(self) -> int:
        return len(self._bars)

    def iter_bars(self) -> Iterator[KlineRecord]:
        yield from self._bars

    def history(
        self,
        symbol: str,
        interval: str,
        *,
        lookback: int,
    ) -> list[KlineRecord]:
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        stream = self._by_key.get((symbol, interval), [])
        now = self._clock.now
        step = interval_to_timedelta(interval)
        # A bar is available iff it has closed by `now` (event_time + step <= now).
        available = [b for b in stream if b.event_time + step <= now]
        return available[-lookback:]

    def latest(self, symbol: str, interval: str) -> KlineRecord | None:
        h = self.history(symbol, interval, lookback=1)
        return h[-1] if h else None

    @classmethod
    def from_manifest(
        cls,
        *,
        manifest: DatasetManifest,
        store: Store,
        clock: SimClock,
        interval: str,
        symbol_filter: str | None = None,
    ) -> MarketReplaySource:
        """Load klines from every partition in a manifest into memory.

        Suitable for research and backtest scale (V1 targets a few years of
        hourly bars = <1M rows). For larger corpora, add a chunked variant.
        """
        bars: list[KlineRecord] = []
        for partition in manifest.partitions:
            raw = store.get(partition.key)
            df = parquet_bytes_to_dataframe(raw)
            for row in df.iter_rows(named=True):
                if row["interval"] != interval:
                    continue
                if symbol_filter is not None and row["symbol"] != symbol_filter:
                    continue
                bars.append(
                    KlineRecord(
                        source=str(row["source"]),
                        category=str(row["category"]),
                        symbol=str(row["symbol"]),
                        interval=str(row["interval"]),
                        event_time=_as_utc(row["event_time"]),
                        ingest_time=_as_utc(row["ingest_time"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        turnover=float(row["turnover"]),
                    )
                )
        return cls(bars=bars, clock=clock, interval=interval)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"parquet returned a naive datetime: {value!r}")
    return value
