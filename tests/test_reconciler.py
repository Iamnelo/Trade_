"""Tests for the REST reconciler."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import respx

from trade.data.quality.metrics import build_metrics
from trade.data.stream.reconciler import (
    ReconcileTarget,
    default_reconcile_period,
    reconcile_once,
)
from trade.exchanges.bybit_public import BybitPublicClient


class _FakeReader:
    def __init__(self, times: list[datetime]) -> None:
        self._times = times

    def latest_event_times(
        self, *, source: str, symbol: str, interval: str, limit: int
    ) -> list[datetime]:
        return list(self._times)


class _FakeSink:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def write(self, records: list[Any]) -> int:
        self.records.extend(records)
        return len(records)


def _kline_page(missing_hour: int) -> dict[str, Any]:
    ts_ms = int(datetime(2024, 1, 1, missing_hour, tzinfo=UTC).timestamp() * 1000)
    rows = [
        [str(ts_ms), "100.0", "101.0", "99.0", "100.5", "10.0", "1005.0"],
    ]
    return {"retCode": 0, "retMsg": "OK", "result": {"list": rows}}


async def test_reconcile_once_fills_missing_bar() -> None:
    # Observed: bars at 00:00 and 02:00 in [00:00, 03:00). Missing: 01:00.
    observed = [
        datetime(2024, 1, 1, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 2, tzinfo=UTC),
    ]
    reader = _FakeReader(observed)
    sink = _FakeSink()
    metrics = build_metrics()

    with respx.mock(base_url="https://bybit.example.com") as router:
        router.get("/v5/market/kline").respond(json=_kline_page(missing_hour=1))

        async with BybitPublicClient(base_url="https://bybit.example.com") as client:
            written = await reconcile_once(
                client=client,
                reader=reader,
                sink=sink,
                targets=[ReconcileTarget(symbol="BTCUSDT", interval="60")],
                metrics=metrics,
                lookback_bars=3,
                now=datetime(2024, 1, 1, 3, tzinfo=UTC),
            )

    assert written == 1
    assert len(sink.records) == 1
    filled = sink.records[0]
    assert filled.event_time == datetime(2024, 1, 1, 1, tzinfo=UTC)
    assert filled.source == "bybit"


async def test_reconcile_once_noop_when_complete() -> None:
    observed = [
        datetime(2024, 1, 1, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 2, tzinfo=UTC),
    ]
    reader = _FakeReader(observed)
    sink = _FakeSink()
    metrics = build_metrics()

    with respx.mock(base_url="https://bybit.example.com", assert_all_called=False) as router:
        route = router.get("/v5/market/kline").respond(
            json={"retCode": 0, "retMsg": "OK", "result": {"list": []}}
        )
        async with BybitPublicClient(base_url="https://bybit.example.com") as client:
            written = await reconcile_once(
                client=client,
                reader=reader,
                sink=sink,
                targets=[ReconcileTarget(symbol="BTCUSDT", interval="60")],
                metrics=metrics,
                lookback_bars=3,
                now=datetime(2024, 1, 1, 3, tzinfo=UTC),
            )

    assert written == 0
    assert sink.records == []
    assert not route.called  # nothing missing = no REST call


async def test_reconcile_once_bumps_error_metric_on_exception() -> None:
    class _BadReader:
        def latest_event_times(
            self, *, source: str, symbol: str, interval: str, limit: int
        ) -> list[datetime]:
            raise RuntimeError("db down")

    metrics = build_metrics()
    async with BybitPublicClient(base_url="https://bybit.example.com") as client:
        try:
            await reconcile_once(
                client=client,
                reader=_BadReader(),
                sink=_FakeSink(),
                targets=[ReconcileTarget(symbol="BTCUSDT", interval="60")],
                metrics=metrics,
                lookback_bars=3,
                now=datetime(2024, 1, 1, 3, tzinfo=UTC),
            )
        except RuntimeError:
            pass
        else:  # pragma: no cover — exception must propagate
            raise AssertionError("Expected RuntimeError to propagate")

    value = metrics.registry.get_sample_value("trade_reconciler_errors_total", {"source": "bybit"})
    assert value == 1.0


def test_default_reconcile_period_clamped() -> None:
    # 60-minute bars: 60min/12 = 5min = 300s (within clamp).
    assert default_reconcile_period("60").total_seconds() == 300.0
    # 1-minute bars: 1min/12 = 5s → clamped to 30s.
    assert default_reconcile_period("1").total_seconds() == 30.0
    # Daily bars: 1440min/12 = 120min = 7200s → clamped to 600s.
    assert default_reconcile_period("D").total_seconds() == 600.0


def _unused(_x: httpx.Response) -> None:
    """Referenced only to keep httpx import used across CI."""
