"""Funding-rate features for the ModelDrivenStrategy.

Funding data is event-driven (Bybit settles every 8 hours) and does not
fit the pure kline-history Feature protocol. These classes bridge the
gap: each takes a pre-loaded `list[FundingRecord]` at construction and
exposes the standard `Feature` interface. At inference time the feature
answers "what was the most recent settled funding rate as of this bar's
close?" via a binary search — strictly PIT-safe.

Three families, all sharing the same lookup convention:

- `FundingRate` (v="1"): value of the latest settlement whose
  `event_time <= bar.event_time`. Availability delay 0 because Bybit
  publishes the final rate at settlement.
- `FundingZScoreN` (v="N"): z-score of the latest funding rate against
  the trailing N settlements.
- `FundingRegime` (v="short_long"): log-ratio of trailing short-window
  mean vs trailing long-window mean. Positive = funding is running hot
  vs its longer baseline; negative = cool.

Loading funding CSVs happens in the ablation runner, not here — the
feature classes are indifferent to source (Bybit / Binance / custom).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from datetime import datetime, timedelta

from trade.data.schemas import FundingRecord, KlineRecord
from trade.features.types import FeatureSpec


class _FundingBackedFeature:
    """Shared bookkeeping: bucket records by symbol + keep per-symbol sorted times."""

    def __init__(self, funding_records: Sequence[FundingRecord]) -> None:
        by_symbol: dict[str, list[FundingRecord]] = {}
        for r in sorted(funding_records, key=lambda x: x.event_time):
            by_symbol.setdefault(r.symbol, []).append(r)
        self._by_symbol = by_symbol
        self._times_by_symbol = {
            sym: [r.event_time for r in stream] for sym, stream in by_symbol.items()
        }

    def _latest_index(self, symbol: str, as_of: datetime) -> int:
        """Index of the latest funding record with event_time <= as_of, or -1."""
        times = self._times_by_symbol.get(symbol)
        if not times:
            return -1
        # bisect_right returns first index STRICTLY greater than as_of.
        return bisect.bisect_right(times, as_of) - 1


class FundingRate(_FundingBackedFeature):
    spec = FeatureSpec(
        name="funding_rate",
        version="1",
        inputs=("funding_rate",),
        lookback_bars=1,
        availability_delay=timedelta(0),
    )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if not history:
            return None
        last = history[-1]
        idx = self._latest_index(last.symbol, last.event_time)
        if idx < 0:
            return None
        return self._by_symbol[last.symbol][idx].funding_rate


class FundingZScoreN(_FundingBackedFeature):
    def __init__(self, funding_records: Sequence[FundingRecord], *, window: int = 21) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        super().__init__(funding_records)
        self._window = window
        self.spec = FeatureSpec(
            name="funding_zscore",
            version=str(window),
            inputs=("funding_rate",),
            lookback_bars=1,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if not history:
            return None
        last = history[-1]
        idx = self._latest_index(last.symbol, last.event_time)
        if idx < 0 or idx + 1 < self._window:
            return None
        stream = self._by_symbol[last.symbol]
        window_slice = stream[idx - self._window + 1 : idx + 1]
        values = [r.funding_rate for r in window_slice]
        mean = sum(values) / self._window
        var = sum((v - mean) ** 2 for v in values) / self._window
        sd = math.sqrt(var)
        if sd == 0.0:
            return 0.0
        return (stream[idx].funding_rate - mean) / sd


class FundingRegime(_FundingBackedFeature):
    def __init__(
        self,
        funding_records: Sequence[FundingRecord],
        *,
        short_window: int = 9,  # ~3 days at 8h cadence
        long_window: int = 63,  # ~3 weeks at 8h cadence
    ) -> None:
        if short_window < 2 or long_window < 2:
            raise ValueError("short_window and long_window must be >= 2")
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        super().__init__(funding_records)
        self._short = short_window
        self._long = long_window
        self.spec = FeatureSpec(
            name="funding_regime",
            version=f"{short_window}_{long_window}",
            inputs=("funding_rate",),
            lookback_bars=1,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        """Standardised short-vs-long funding gap.

        `(short_mean - long_mean) / long_std`. Signed correctly (positive =
        short-run funding is hotter than the long-run baseline, negative =
        cooler); handles negative funding regimes cleanly because it works
        with raw differences, not ratios of absolute values.
        """
        if not history:
            return None
        last = history[-1]
        idx = self._latest_index(last.symbol, last.event_time)
        if idx < 0 or idx + 1 < self._long:
            return None
        stream = self._by_symbol[last.symbol]
        long_slice = stream[idx - self._long + 1 : idx + 1]
        short_slice = stream[idx - self._short + 1 : idx + 1]
        long_rates = [r.funding_rate for r in long_slice]
        long_mean = sum(long_rates) / self._long
        short_mean = sum(r.funding_rate for r in short_slice) / self._short
        long_var = sum((r - long_mean) ** 2 for r in long_rates) / self._long
        long_std = math.sqrt(long_var)
        if long_std == 0.0:
            # Long history is perfectly flat — regime "change" is undefined;
            # any non-zero gap between short and long is by definition anomalous
            # but z-scoring blows up. Return the raw gap sign as a fallback.
            if short_mean == long_mean:
                return 0.0
            return 1.0 if short_mean > long_mean else -1.0
        return (short_mean - long_mean) / long_std
