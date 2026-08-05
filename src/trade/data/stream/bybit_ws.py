"""Bybit v5 public WebSocket client for kline streams.

Subscribes to `kline.{interval}.{symbol}` topics and emits `KlineRecord` for
every CONFIRMED bar (Bybit publishes both in-progress and closed bars; only
the closed ones are yielded). Sends a `{"op":"ping"}` heartbeat every 20s.
Reconnects with exponential backoff and re-subscribes to the same topics.

The `connector` argument is injected so tests can supply a fake websocket
factory. The default uses the real `websockets` library.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from typing import Any, Protocol

import websockets

from trade.data.schemas import KlineRecord
from trade.utils.clock import from_epoch_ms, utcnow

_HEARTBEAT_INTERVAL_SECONDS: float = 20.0
_RECONNECT_INITIAL_SECONDS: float = 1.0
_RECONNECT_MAX_SECONDS: float = 30.0


class WSClient(Protocol):
    async def send(self, data: str) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...
    async def __aenter__(self) -> WSClient: ...
    async def __aexit__(self, *args: object) -> None: ...


Connector = Callable[[str], Awaitable[WSClient]]


async def _default_connector(url: str) -> WSClient:
    return await websockets.connect(url)  # type: ignore[return-value]


def _kline_topic(interval: str, symbol: str) -> str:
    return f"kline.{interval}.{symbol}"


def _parse_kline_message(
    payload: dict[str, Any],
    *,
    category: str,
) -> list[KlineRecord]:
    """Convert a Bybit kline WS envelope into KlineRecord list (confirmed bars only)."""
    topic = str(payload.get("topic", ""))
    parts = topic.split(".")
    if len(parts) != 3 or parts[0] != "kline":
        return []
    interval, symbol = parts[1], parts[2]

    ingest_time = utcnow()
    out: list[KlineRecord] = []
    for bar in payload.get("data") or []:
        if not bar.get("confirm", False):
            continue
        out.append(
            KlineRecord(
                source="bybit",
                category=category,
                symbol=symbol,
                interval=interval,
                event_time=from_epoch_ms(int(bar["start"])),
                ingest_time=ingest_time,
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar["volume"]),
                turnover=float(bar["turnover"]),
            )
        )
    return out


class BybitKlineStream:
    """Async-iterable stream of confirmed kline records over WebSocket."""

    def __init__(
        self,
        *,
        url: str = "wss://stream.bybit.com/v5/public/linear",
        category: str = "linear",
        symbols: Sequence[str],
        intervals: Sequence[str],
        connector: Connector | None = None,
        heartbeat_seconds: float = _HEARTBEAT_INTERVAL_SECONDS,
        max_reconnects: int | None = None,
        on_reconnect: Callable[[], None] | None = None,
    ) -> None:
        if not symbols:
            raise ValueError("at least one symbol is required")
        if not intervals:
            raise ValueError("at least one interval is required")
        self._url = url
        self._category = category
        self._symbols = tuple(symbols)
        self._intervals = tuple(intervals)
        self._connector = connector or _default_connector
        self._heartbeat_seconds = heartbeat_seconds
        self._max_reconnects = max_reconnects
        self._on_reconnect = on_reconnect
        self._topics = [_kline_topic(i, s) for s in self._symbols for i in self._intervals]

    def topics(self) -> list[str]:
        return list(self._topics)

    async def __aiter__(self) -> AsyncIterator[list[KlineRecord]]:
        attempt = 0
        backoff = _RECONNECT_INITIAL_SECONDS
        while True:
            with contextlib.suppress(Exception):
                async for batch in self._session():
                    yield batch
                # Session ended without exception (server closed cleanly) — reconnect.

            attempt += 1
            if self._max_reconnects is not None and attempt > self._max_reconnects:
                return
            if self._on_reconnect is not None:
                self._on_reconnect()
            await asyncio.sleep(min(backoff, _RECONNECT_MAX_SECONDS))
            backoff = min(backoff * 2, _RECONNECT_MAX_SECONDS)

    async def _session(self) -> AsyncIterator[list[KlineRecord]]:
        ws = await self._connector(self._url)
        try:
            await ws.send(json.dumps({"op": "subscribe", "args": list(self._topics)}))
            heartbeat_task = asyncio.create_task(self._heartbeat(ws))
            try:
                while True:
                    raw = await ws.recv()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    payload = json.loads(raw)
                    # Ignore ack / subscribe / pong control frames.
                    if not isinstance(payload, dict) or "topic" not in payload:
                        continue
                    records = _parse_kline_message(payload, category=self._category)
                    if records:
                        yield records
            finally:
                heartbeat_task.cancel()
                # Swallow the cancellation to keep tests deterministic.
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await heartbeat_task
        finally:
            await ws.close()

    async def _heartbeat(self, ws: WSClient) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            await ws.send(json.dumps({"op": "ping"}))


def confirmed_records(payloads: Iterable[dict[str, Any]], *, category: str) -> list[KlineRecord]:
    """Testing helper: parse a sequence of raw WS envelopes."""
    out: list[KlineRecord] = []
    for p in payloads:
        out.extend(_parse_kline_message(p, category=category))
    return out
