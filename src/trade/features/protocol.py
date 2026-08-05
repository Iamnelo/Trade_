"""Feature protocol.

A `Feature` is a pure function of a bounded window of past bars. The store
guarantees that `compute(history)` is only ever called with a history whose
last bar is at time t and whose earlier bars are strictly less than t. The
feature must:

- return the value AT time t (the last bar in `history`)
- read no data outside `history`
- be deterministic (same input -> same output)
- respect its declared `lookback_bars` (adding bars older than the tail must
  not change the result)
- return None when history is insufficient to produce a value

Every concrete feature MUST be paired with a contract test — see
`trade.features.contract`. CI blocks a feature merging without one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec


class Feature(Protocol):
    @property
    def spec(self) -> FeatureSpec: ...

    def compute(self, history: Sequence[KlineRecord]) -> float | None: ...
