"""Tests for the Bybit v5 public WebSocket kline stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest

from trade.data.stream.bybit_ws import BybitKlineStream, confirmed_records


class _FakeWSClient:
    def __init__(self, messages: list[str]) -> None:
        self._messages = list(messages)
        self.sent: list[str] = []
        self.closed = False

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        if not self._messages:
            # Simulate the server closing the socket cleanly after messages drain.
            await asyncio.sleep(0)
            raise ConnectionError("stream ended")
        return self._messages.pop(0)

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> _FakeWSClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


def _kline_envelope(
    interval: str, symbol: str, start_ms: int, close: float, *, confirm: bool
) -> dict[str, Any]:
    return {
        "topic": f"kline.{interval}.{symbol}",
        "type": "snapshot",
        "data": [
            {
                "start": start_ms,
                "end": start_ms + 3_600_000 - 1,
                "interval": interval,
                "open": "100.0",
                "close": str(close),
                "high": str(close + 1),
                "low": str(close - 1),
                "volume": "10.0",
                "turnover": str(close * 10),
                "confirm": confirm,
                "timestamp": start_ms + 3_600_000,
            }
        ],
        "ts": start_ms + 3_600_000,
    }


def test_confirmed_records_parses_confirmed_only() -> None:
    payloads = [
        _kline_envelope("60", "BTCUSDT", 1704067200000, 100.5, confirm=False),
        _kline_envelope("60", "BTCUSDT", 1704070800000, 101.0, confirm=True),
    ]
    got = confirmed_records(payloads, category="linear")
    assert len(got) == 1
    r = got[0]
    assert r.source == "bybit"
    assert r.category == "linear"
    assert r.symbol == "BTCUSDT"
    assert r.interval == "60"
    assert r.close == 101.0
    assert r.event_time == datetime(2024, 1, 1, 1, tzinfo=UTC)


def test_stream_subscribes_and_yields_confirmed_bars() -> None:
    messages = [
        json.dumps({"success": True, "op": "subscribe"}),  # ack — ignored
        json.dumps(_kline_envelope("60", "BTCUSDT", 1704067200000, 100.5, confirm=False)),
        json.dumps(_kline_envelope("60", "BTCUSDT", 1704070800000, 101.0, confirm=True)),
    ]
    ws = _FakeWSClient(messages)

    async def _connector(_url: str) -> _FakeWSClient:
        return ws

    reconnects: list[bool] = []

    stream = BybitKlineStream(
        symbols=["BTCUSDT"],
        intervals=["60"],
        connector=_connector,
        max_reconnects=0,
        heartbeat_seconds=1000.0,
        on_reconnect=lambda: reconnects.append(True),
    )

    async def _drive() -> list[Any]:
        collected = []
        async for batch in stream:
            collected.extend(batch)
        return collected

    got = asyncio.run(_drive())
    assert len(got) == 1
    assert got[0].close == 101.0

    subscribe_frame = json.loads(ws.sent[0])
    assert subscribe_frame == {"op": "subscribe", "args": ["kline.60.BTCUSDT"]}
    assert ws.closed
    # A single session ended cleanly; because max_reconnects=0, no retries fired.
    assert reconnects == []


def test_stream_reconnects_and_calls_hook() -> None:
    factory_calls = 0

    def _make_ws() -> _FakeWSClient:
        return _FakeWSClient([json.dumps({"success": True, "op": "subscribe"})])

    async def _connector(_url: str) -> _FakeWSClient:
        nonlocal factory_calls
        factory_calls += 1
        return _make_ws()

    hook_calls: list[bool] = []
    stream = BybitKlineStream(
        symbols=["BTCUSDT"],
        intervals=["60"],
        connector=_connector,
        max_reconnects=1,
        heartbeat_seconds=1000.0,
        on_reconnect=lambda: hook_calls.append(True),
    )

    async def _drive() -> AsyncIterator[Any]:
        async for _ in stream:
            yield None

    async def _run() -> None:
        async for _ in _drive():
            pass

    asyncio.run(_run())
    # The reconnect loop must have called the connector at least twice
    # (initial + one reconnect) and invoked the hook once.
    assert factory_calls >= 2
    assert hook_calls == [True]


def test_stream_rejects_empty_symbols_or_intervals() -> None:
    with pytest.raises(ValueError, match="symbol"):
        BybitKlineStream(symbols=[], intervals=["60"])
    with pytest.raises(ValueError, match="interval"):
        BybitKlineStream(symbols=["BTCUSDT"], intervals=[])
