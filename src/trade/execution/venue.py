"""ExecutionVenue Protocol.

The signal engine + OMS + risk manager depend ONLY on this Protocol. Three
implementations satisfy it:

- `PaperExecutionVenue` (in `paper_venue.py`): in-memory, instant fills at
  mark ± slippage. Used for paper trading and CLI smoke tests.
- `BybitExecutionVenue` (in `trade.exchanges.bybit_venue`): signed Bybit
  v5 REST. Used for live trading.
- The Market Replay Engine's `SimulatedVenue` is intentionally NOT part of
  this Protocol — the MRE has a bar-close event-loop shape that differs
  from the async request/response shape of live venues. The `run_backtest`
  loop drives the SimulatedVenue directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from trade.execution.types import VenueOrderAck, VenuePosition
from trade.mre.types import Order


class ExecutionVenue(Protocol):
    async def submit_order(self, order: Order) -> VenueOrderAck: ...

    async def cancel_order(self, *, symbol: str, client_order_id: str) -> None: ...

    async def cancel_all(self, *, symbol: str | None = None) -> int: ...

    async def get_positions(self, *, symbol: str | None = None) -> Sequence[VenuePosition]: ...
