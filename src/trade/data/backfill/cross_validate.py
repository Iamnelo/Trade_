"""Cross-source validation between backfilled datasets.

Given two sequences of `KlineRecord` (typically Bybit vs. Binance for the same
symbol+interval), report per-bar close-price deltas in basis points and
coverage mismatches. Absolute price differs across venues; only the relative
delta is meaningful.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from trade.data.schemas import KlineRecord


@dataclass(frozen=True, slots=True)
class CrossSourceDelta:
    event_time: datetime
    a_close: float
    b_close: float
    close_delta_bps: float


@dataclass(frozen=True, slots=True)
class CrossSourceReport:
    common_bars: int
    a_only_bars: int
    b_only_bars: int
    deltas: tuple[CrossSourceDelta, ...]

    @property
    def max_abs_delta_bps(self) -> float:
        return max((abs(d.close_delta_bps) for d in self.deltas), default=0.0)

    @property
    def median_abs_delta_bps(self) -> float:
        if not self.deltas:
            return 0.0
        vals = sorted(abs(d.close_delta_bps) for d in self.deltas)
        return vals[len(vals) // 2]


def _index(records: Iterable[KlineRecord]) -> dict[datetime, KlineRecord]:
    return {r.event_time: r for r in records}


def compare_klines(
    a_records: Iterable[KlineRecord],
    b_records: Iterable[KlineRecord],
) -> CrossSourceReport:
    a_by_time = _index(a_records)
    b_by_time = _index(b_records)
    common_times = sorted(a_by_time.keys() & b_by_time.keys())
    a_only = len(a_by_time.keys() - b_by_time.keys())
    b_only = len(b_by_time.keys() - a_by_time.keys())

    deltas: list[CrossSourceDelta] = []
    for ts in common_times:
        a = a_by_time[ts]
        b = b_by_time[ts]
        bps = 0.0 if a.close == 0 else (b.close - a.close) / a.close * 10_000
        deltas.append(
            CrossSourceDelta(
                event_time=ts,
                a_close=a.close,
                b_close=b.close,
                close_delta_bps=bps,
            )
        )

    return CrossSourceReport(
        common_bars=len(common_times),
        a_only_bars=a_only,
        b_only_bars=b_only,
        deltas=tuple(deltas),
    )
