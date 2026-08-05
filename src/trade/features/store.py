"""FeatureStore: PIT-only training frame construction.

Two implementations:

- `InMemoryFeatureStore`: simple dict-backed store, useful for research
  scripts and tests.
- `DuckDBFeatureStore` (see `duckdb_store.py`): queries parquet on disk via
  DuckDB and uses window functions for PIT joins; the production choice.

There is DELIBERATELY no `get_latest_features` / `load_now` / `snapshot`
method. The ONLY way to obtain features for a label sequence is
`point_in_time_join(labels=..., feature_ids=...)` which returns a
`TrainingFrame` containing, per label at time t, the newest feature value
with `availability_time <= t`. Adding a bypass is a spec violation.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from trade.data.schemas import KlineRecord
from trade.features.protocol import Feature
from trade.features.types import (
    FeatureSpec,
    LabelRow,
    MaterializedFeature,
    TrainingFrame,
)


class FeatureStore(Protocol):
    def materialize(
        self,
        *,
        feature: Feature,
        entity_id: str,
        bars: Sequence[KlineRecord],
    ) -> list[MaterializedFeature]: ...

    def put_materialized(
        self,
        *,
        materialized: Sequence[MaterializedFeature],
        feature_spec: FeatureSpec,
    ) -> None: ...

    def point_in_time_join(
        self,
        *,
        labels: Sequence[LabelRow],
        feature_ids: Sequence[str],
    ) -> TrainingFrame: ...


def materialize_feature(
    *,
    feature: Feature,
    entity_id: str,
    bars: Sequence[KlineRecord],
) -> list[MaterializedFeature]:
    """Roll a feature across bars, emitting one row per bar with a defined value.

    A bar that produces `None` (insufficient history) is skipped rather than
    written as null. That keeps the materialisation compact and makes the
    PIT-join "last available value" semantics unambiguous.
    """
    sorted_bars = sorted(bars, key=lambda b: b.event_time)
    out: list[MaterializedFeature] = []
    for i in range(len(sorted_bars)):
        window = sorted_bars[: i + 1]
        value = feature.compute(window)
        if value is None:
            continue
        current = sorted_bars[i]
        out.append(
            MaterializedFeature(
                feature_id=feature.spec.full_id,
                entity_id=entity_id,
                event_time=current.event_time,
                availability_time=current.event_time + feature.spec.availability_delay,
                value=value,
            )
        )
    return out


class InMemoryFeatureStore:
    def __init__(self) -> None:
        # (feature_id, entity_id) -> list of MaterializedFeature sorted by availability_time
        self._table: dict[tuple[str, str], list[MaterializedFeature]] = defaultdict(list)
        # feature_id -> FeatureSpec (kept for round-tripping / introspection)
        self._specs: dict[str, FeatureSpec] = {}

    def materialize(
        self,
        *,
        feature: Feature,
        entity_id: str,
        bars: Sequence[KlineRecord],
    ) -> list[MaterializedFeature]:
        rows = materialize_feature(feature=feature, entity_id=entity_id, bars=bars)
        self.put_materialized(materialized=rows, feature_spec=feature.spec)
        return rows

    def put_materialized(
        self,
        *,
        materialized: Sequence[MaterializedFeature],
        feature_spec: FeatureSpec,
    ) -> None:
        self._specs[feature_spec.full_id] = feature_spec
        for row in materialized:
            key = (row.feature_id, row.entity_id)
            self._table[key].append(row)
        # Keep each stream sorted by availability_time so PIT lookup can binary-search.
        for key in {(r.feature_id, r.entity_id) for r in materialized}:
            self._table[key].sort(key=lambda r: r.availability_time)

    def point_in_time_join(
        self,
        *,
        labels: Sequence[LabelRow],
        feature_ids: Sequence[str],
    ) -> TrainingFrame:
        columns: dict[str, list[float | None]] = {fid: [] for fid in feature_ids}
        for label in labels:
            for fid in feature_ids:
                stream = self._table.get((fid, label.entity_id), [])
                if not stream:
                    columns[fid].append(None)
                    continue
                # Newest availability_time <= label.event_time.
                # bisect on a parallel list of availability_time keys.
                keys = [r.availability_time for r in stream]
                idx = bisect_right(keys, label.event_time)
                if idx == 0:
                    columns[fid].append(None)
                else:
                    columns[fid].append(stream[idx - 1].value)
        return TrainingFrame(
            entity_ids=tuple(lbl.entity_id for lbl in labels),
            event_times=tuple(lbl.event_time for lbl in labels),
            labels=tuple(lbl.label for lbl in labels),
            features={fid: tuple(vals) for fid, vals in columns.items()},
        )
