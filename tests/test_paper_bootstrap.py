"""Historical bootstrap: seed_history + closed-candle fetch.

Verifies the live paper bot obtains genuine historical feature context so the
FIRST live bar decides instead of WARMUP — without lowering any warmup
requirement, changing the timeframe, or touching model/strategy/risk/fills.
No network: the REST client is injected with a fake.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tests.test_paper_engine import _bars, _engine, _freeze_synth
from trade.exchanges.base import Kline
from trade.paper.bootstrap import _current_interval_boundary, fetch_seed_bars


def _last_decision(journal_dir: Path) -> dict:
    lines = [
        json.loads(line)
        for line in (journal_dir / "decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    decisions = [line for line in lines if line["kind"] == "decision"]
    return decisions[-1]


def test_seed_history_enables_first_bar_nonwarmup(tmp_path: Path) -> None:
    entry, bars = _freeze_synth(tmp_path)  # synth winner max_lookback = 21
    engine = _engine(tmp_path, execution_enabled=False, entry=entry)

    seeded = engine.seed_history(bars[:60])
    assert seeded["BTCUSDT"] == 60

    # The very next (live) bar must produce a real decision, not WARMUP.
    engine.process_batch([bars[60]])
    dec = _last_decision(tmp_path / "journal")
    assert dec["status"] == "decided"
    assert dec["action"] != "WARMUP"


def test_seed_history_is_idempotent_and_restart_safe(tmp_path: Path) -> None:
    entry, bars = _freeze_synth(tmp_path)
    engine = _engine(tmp_path, execution_enabled=False, entry=entry)
    first = engine.seed_history(bars[:60])
    assert first["BTCUSDT"] == 60
    # Buffer already >= max_lookback -> a second seed adds nothing.
    second = engine.seed_history(bars[:60])
    assert second["BTCUSDT"] == 0


def test_seed_history_dedupes_and_sorts_prefer_existing(tmp_path: Path) -> None:
    # Seed a partial buffer (below max_lookback so the merge path runs), then
    # re-seed with overlapping event_times; existing bars must win on ties and
    # the union stays sorted with no duplicates.
    entry, bars = _freeze_synth(tmp_path)
    engine = _engine(tmp_path, execution_enabled=False, entry=entry)
    engine.seed_history(bars[:10])  # 10 < max_lookback (21)

    # A bar with the SAME event_time as bars[5] but a different close.
    b5 = bars[5]
    conflicting = _bars(1)[0]
    conflicting = type(b5)(
        source=b5.source,
        category=b5.category,
        symbol="BTCUSDT",
        interval="D",
        event_time=b5.event_time,
        ingest_time=b5.ingest_time,
        open=b5.open,
        high=b5.high,
        low=b5.low,
        close=b5.close + 999.0,  # would corrupt the buffer if it won
        volume=b5.volume,
        turnover=b5.turnover,
    )
    engine.seed_history([conflicting, *bars[10:15]])

    buf = engine._buffers["BTCUSDT"]
    times = [b.event_time for b in buf]
    assert times == sorted(times)  # sorted
    assert len(times) == len(set(times))  # no duplicates
    assert len(buf) == 15  # union of 0..15
    kept = next(b for b in buf if b.event_time == b5.event_time)
    assert kept.close == b5.close  # existing (live/earlier) bar won the tie


def test_current_interval_boundary_excludes_in_progress() -> None:
    now = datetime(2026, 8, 12, 14, 30, tzinfo=UTC)  # mid-day
    boundary = _current_interval_boundary(now, "D")
    assert boundary == datetime(2026, 8, 12, 0, 0, tzinfo=UTC)
    # fetch uses this as an EXCLUSIVE end, so today's still-open candle (opens at
    # 00:00) is dropped; the last CLOSED candle is 2026-08-11.


class _FakeClient:
    """Stand-in BybitPublicClient returning synthetic klines incl. one unclosed."""

    def __init__(self, closed_days: int) -> None:
        self.closed_days = closed_days
        self.calls: list[tuple[datetime, datetime]] = []

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[Kline]:
        self.calls.append((start, end))
        # Emit closed daily candles in [start, end) PLUS one at `end` (the
        # in-progress candle) that page_klines must drop.
        out: list[Kline] = []
        day = start
        while day <= end:
            out.append(
                Kline(
                    symbol=symbol,
                    interval=interval,
                    open_time=day,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1.0,
                    turnover=100.0,
                )
            )
            day = day + timedelta(days=1)
        return out


def test_fetch_seed_bars_returns_only_closed_candles() -> None:
    now = datetime(2026, 8, 12, 14, 0, tzinfo=UTC)
    fake = _FakeClient(closed_days=5)
    bars = asyncio.run(
        fetch_seed_bars(
            symbols=["BTCUSDT"],
            interval="D",
            n_bars=5,
            base_url="https://example.invalid",
            now=now,
            client=fake,
        )
    )
    # end is 2026-08-12T00:00; the in-progress candle at that instant is excluded.
    assert all(b.event_time < datetime(2026, 8, 12, tzinfo=UTC) for b in bars)
    assert bars[-1].event_time == datetime(2026, 8, 11, tzinfo=UTC)
