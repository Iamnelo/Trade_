"""Pure data-quality checks over kline sequences.

These functions do not talk to Prometheus, storage, or exchanges. They accept
records and return typed results so both the live-ingest path and the backfill
path can reuse them. Callers publish the results into `DQMetrics`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from trade.data.backfill.common import interval_to_timedelta
from trade.data.schemas import KlineRecord


@dataclass(frozen=True, slots=True)
class GapReport:
    expected_bars: int
    observed_bars: int
    missing_bars: int
    missing_event_times: tuple[datetime, ...]

    @property
    def gap_percent(self) -> float:
        if self.expected_bars == 0:
            return 0.0
        return 100.0 * self.missing_bars / self.expected_bars


def gap_report(
    records: Sequence[KlineRecord],
    *,
    interval: str,
    window_start: datetime,
    window_end: datetime,
) -> GapReport:
    """Detect missing bars in a closed-open time window [window_start, window_end).

    The expected grid is the set of bar open-times aligned to `window_start`
    that fall inside the window. Records outside the window are ignored.
    """
    if window_start.tzinfo is None or window_end.tzinfo is None:
        raise ValueError("window bounds must be UTC-aware datetimes")
    if window_start >= window_end:
        raise ValueError("window_start must be strictly before window_end")

    step = interval_to_timedelta(interval)
    expected: list[datetime] = []
    cursor = window_start
    while cursor < window_end:
        expected.append(cursor)
        cursor += step

    observed = {r.event_time for r in records if window_start <= r.event_time < window_end}
    missing = tuple(t for t in expected if t not in observed)
    return GapReport(
        expected_bars=len(expected),
        observed_bars=len(expected) - len(missing),
        missing_bars=len(missing),
        missing_event_times=missing,
    )


def staleness_seconds(last_event_time: datetime, now: datetime) -> float:
    """Wall-clock seconds elapsed since the most recent observed bar's event_time."""
    if last_event_time.tzinfo is None or now.tzinfo is None:
        raise ValueError("timestamps must be UTC-aware")
    return max(0.0, (now - last_event_time).total_seconds())


@dataclass(frozen=True, slots=True)
class PriceSanityResult:
    total_bars: int
    violations: tuple[KlineRecord, ...]

    @property
    def violation_count(self) -> int:
        return len(self.violations)


def price_sanity(
    records: Sequence[KlineRecord],
    *,
    window: int = 20,
    band_pct: float = 0.20,
) -> PriceSanityResult:
    """Flag bars whose close deviates from the rolling-median close by > band_pct.

    Uses a trailing window ending at the bar itself (not centered) so this is
    replay-safe. The first `window - 1` bars have no reference and are skipped.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if not 0.0 < band_pct <= 1.0:
        raise ValueError("band_pct must be within (0.0, 1.0]")

    sorted_records = sorted(records, key=lambda r: r.event_time)
    violations: list[KlineRecord] = []
    for i in range(window - 1, len(sorted_records)):
        window_closes = sorted([r.close for r in sorted_records[i - window + 1 : i + 1]])
        median = window_closes[len(window_closes) // 2]
        if median <= 0:
            continue
        rel = abs(sorted_records[i].close - median) / median
        if rel > band_pct:
            violations.append(sorted_records[i])
    return PriceSanityResult(total_bars=len(sorted_records), violations=tuple(violations))


def find_missing_bars(
    observed_event_times: Iterable[datetime],
    *,
    interval: str,
    window_start: datetime,
    window_end: datetime,
) -> list[datetime]:
    """Small helper: expected bar times in window minus observed ones. Used by the reconciler."""
    step = interval_to_timedelta(interval)
    if window_start.tzinfo is None or window_end.tzinfo is None:
        raise ValueError("window bounds must be UTC-aware")
    observed = {t for t in observed_event_times if window_start <= t < window_end}
    missing: list[datetime] = []
    cursor = window_start
    while cursor < window_end:
        if cursor not in observed:
            missing.append(cursor)
        cursor += step
    return missing


def within_expected_interval(
    a: datetime, b: datetime, *, interval: str, tolerance: timedelta = timedelta(seconds=1)
) -> bool:
    """True if the gap between consecutive bars matches the interval within tolerance."""
    step = interval_to_timedelta(interval)
    delta = abs((b - a) - step)
    return delta <= tolerance
