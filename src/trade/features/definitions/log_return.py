"""Log-return feature: ln(close_t / close_{t-window})."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timedelta

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


class LogReturnN:
    def __init__(self, *, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self.spec = FeatureSpec(
            name="log_return",
            version=str(window),
            inputs=("close",),
            lookback_bars=window + 1,
            availability_delay=timedelta(0),
        )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if len(history) < self.spec.lookback_bars:
            return None
        window = history[-self.spec.lookback_bars :]
        old_close = window[0].close
        new_close = window[-1].close
        if old_close <= 0 or new_close <= 0:
            return None
        return math.log(new_close / old_close)
