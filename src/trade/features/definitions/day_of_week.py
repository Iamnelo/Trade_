"""Cyclic day-of-week encoding (sin/cos).

Uses `event_time.weekday()` (Mon=0 ... Sun=6):

    dow_sin = sin(2π * weekday / 7)
    dow_cos = cos(2π * weekday / 7)

Include both features together to give the model the full cycle. Crypto
markets are 24/7 so both weekday and weekend structure can leak into
returns via funding cadence and macro news schedules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timedelta

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


class DayOfWeekSin:
    spec = FeatureSpec(
        name="day_of_week",
        version="sin",
        inputs=("event_time",),
        lookback_bars=1,
        availability_delay=timedelta(0),
    )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if not history:
            return None
        dow = history[-1].event_time.weekday()
        return math.sin(2 * math.pi * dow / 7.0)


class DayOfWeekCos:
    spec = FeatureSpec(
        name="day_of_week",
        version="cos",
        inputs=("event_time",),
        lookback_bars=1,
        availability_delay=timedelta(0),
    )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if not history:
            return None
        dow = history[-1].event_time.weekday()
        return math.cos(2 * math.pi * dow / 7.0)
