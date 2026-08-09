"""Bar feeds for the paper engine.

The engine consumes an async iterator of confirmed-`KlineRecord` batches. In
production that is the `BybitKlineStream` (WebSocket, confirmed bars only). For
tests and dry runs, `ReplayFeed` yields historical bars grouped by event_time
so a run is deterministic and needs no network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from itertools import groupby

from trade.data.schemas import KlineRecord


class ReplayFeed:
    """Async-iterable that yields committed bars grouped by event_time.

    Bars sharing an event_time (e.g. same-day BTC + ETH closes) are delivered
    in one batch, mirroring how the engine groups a live tick.
    """

    def __init__(self, bars: Sequence[KlineRecord]) -> None:
        self._bars = sorted(bars, key=lambda b: (b.event_time, b.symbol))

    async def __aiter__(self) -> AsyncIterator[list[KlineRecord]]:
        for _event_time, group in groupby(self._bars, key=lambda b: b.event_time):
            yield list(group)
