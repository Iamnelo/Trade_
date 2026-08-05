"""Feature-store value types.

Every feature is described by a `FeatureSpec` (name + version + declared
inputs, lookback, and availability delay). Every materialisation produces
`MaterializedFeature` rows carrying an explicit `availability_time` — the
moment the value became usable in decisions — so the store's PIT join is
strictly monotone in wall-clock terms, not merely in event time.

Training frames are the ONLY shape a model receives features in. There is
no "latest features" convenience: the store yields a `TrainingFrame` only
via `point_in_time_join`, which selects at most one feature value per label
row (the newest with `availability_time <= label.event_time`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    version: str
    inputs: tuple[str, ...]  # bar column names read from source data
    lookback_bars: int
    availability_delay: timedelta
    entity: str = "symbol"
    interval: str = "60"

    @property
    def full_id(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class MaterializedFeature:
    feature_id: str
    entity_id: str
    event_time: datetime  # source bar's event_time
    availability_time: datetime  # event_time + spec.availability_delay
    value: float


@dataclass(frozen=True, slots=True)
class LabelRow:
    entity_id: str
    event_time: datetime
    label: float


@dataclass(frozen=True, slots=True)
class TrainingFrame:
    entity_ids: tuple[str, ...]
    event_times: tuple[datetime, ...]
    labels: tuple[float, ...]
    features: Mapping[str, tuple[float | None, ...]] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.event_times)

    @property
    def feature_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.features.keys()))
