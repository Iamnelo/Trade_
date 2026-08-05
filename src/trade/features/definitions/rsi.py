"""RSI-14 reference feature.

The 14-period Relative Strength Index computed on close prices, in the
classic Wilder simple-average form (not the exponential smoothing form).
Kept as the reference feature for Phase 2.5: the contract test framework
in `trade.features.contract` is exercised against this feature to prove
the framework itself works.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from itertools import pairwise

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


class RSI14:
    spec = FeatureSpec(
        name="rsi_close",
        version="14",
        inputs=("close",),
        # 14 close-to-close diffs need 15 bars.
        lookback_bars=15,
        # Available at bar close — the platform's convention for MRE/live parity.
        availability_delay=timedelta(0),
        entity="symbol",
        interval="60",
    )

    def compute(self, history: Sequence[KlineRecord]) -> float | None:
        if len(history) < self.spec.lookback_bars:
            return None
        # Only look at the declared lookback tail. Anything else would be
        # a spec violation caught by the contract test.
        window = history[-self.spec.lookback_bars :]
        diffs = [b.close - a.close for a, b in pairwise(window)]
        gains = [d if d > 0 else 0.0 for d in diffs]
        losses = [-d if d < 0 else 0.0 for d in diffs]
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss == 0.0:
            # All bars up or flat — RSI is defined as 100 in the limit.
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
