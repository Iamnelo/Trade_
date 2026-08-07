"""Cyclic hour-of-day encoding (sin/cos).

Emitted as two separate features because the framework produces one
scalar per feature id. Both must be included together in an experiment
for the model to reconstruct the circular structure:

    hour_sin = sin(2π * hour / 24)
    hour_cos = cos(2π * hour / 24)

Reading `event_time.hour` never needs history beyond the current bar,
so `lookback_bars=1` — the contract test still verifies determinism and
lookback-respect trivially.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timedelta

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


class HourOfDaySin:
    spec = FeatureSpec(
        name="time_of_day",
        version="sin",
        inputs=("event_time",),
        lookback_bars=1,
        availability_delay=timedelta(0),
    )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if not history:
            return None
        hour = history[-1].event_time.hour
        return math.sin(2 * math.pi * hour / 24.0)


class HourOfDayCos:
    spec = FeatureSpec(
        name="time_of_day",
        version="cos",
        inputs=("event_time",),
        lookback_bars=1,
        availability_delay=timedelta(0),
    )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if not history:
            return None
        hour = history[-1].event_time.hour
        return math.cos(2 * math.pi * hour / 24.0)
