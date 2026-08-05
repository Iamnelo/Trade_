"""DuckDB-backed FeatureStore (production offline tier).

Same contract as `InMemoryFeatureStore` — the only way to get features for
training is `point_in_time_join`. DuckDB is used for the PIT join because
its window functions and parquet reader make the query both compact and
fast at V1 scale (millions of rows).

The store keeps materialised features as one in-memory Arrow table per
(feature_id, entity_id) for tests; a parquet-backed variant slots in
without changing the query.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import duckdb

from trade.data.schemas import KlineRecord
from trade.features.protocol import Feature
from trade.features.store import materialize_feature
from trade.features.types import (
    FeatureSpec,
    LabelRow,
    MaterializedFeature,
    TrainingFrame,
)


class DuckDBFeatureStore:
    def __init__(self) -> None:
        self._con: duckdb.DuckDBPyConnection = duckdb.connect(":memory:")
        self._con.execute(
            """
            CREATE TABLE features (
                feature_id VARCHAR,
                entity_id VARCHAR,
                event_time TIMESTAMPTZ,
                availability_time TIMESTAMPTZ,
                value DOUBLE
            )
            """
        )
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
        if not materialized:
            return
        # Bulk insert via a list of tuples.
        rows: list[tuple[Any, ...]] = [
            (r.feature_id, r.entity_id, r.event_time, r.availability_time, r.value)
            for r in materialized
        ]
        self._con.executemany(
            "INSERT INTO features VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def point_in_time_join(
        self,
        *,
        labels: Sequence[LabelRow],
        feature_ids: Sequence[str],
    ) -> TrainingFrame:
        if not labels:
            return TrainingFrame(
                entity_ids=(),
                event_times=(),
                labels=(),
                features=dict.fromkeys(feature_ids, ()),
            )
        # Register the label list as an ephemeral relation.
        label_rows = [
            (i, label.entity_id, label.event_time, label.label) for i, label in enumerate(labels)
        ]
        self._con.execute(
            """
            CREATE OR REPLACE TEMP TABLE labels (
                row_ix INTEGER,
                entity_id VARCHAR,
                event_time TIMESTAMPTZ,
                label DOUBLE
            )
            """
        )
        self._con.executemany("INSERT INTO labels VALUES (?, ?, ?, ?)", label_rows)

        columns: dict[str, list[float | None]] = {fid: [None] * len(labels) for fid in feature_ids}
        for fid in feature_ids:
            query = """
                WITH ranked AS (
                    SELECT
                        l.row_ix,
                        f.value,
                        row_number() OVER (
                            PARTITION BY l.row_ix
                            ORDER BY f.availability_time DESC
                        ) AS rn
                    FROM labels l
                    LEFT JOIN features f
                      ON f.feature_id = ?
                     AND f.entity_id = l.entity_id
                     AND f.availability_time <= l.event_time
                )
                SELECT row_ix, value
                FROM ranked
                WHERE rn = 1
                ORDER BY row_ix
            """
            cursor = self._con.execute(query, [fid])
            for row_ix, value in cursor.fetchall():
                columns[fid][row_ix] = None if value is None else float(value)

        return TrainingFrame(
            entity_ids=tuple(lbl.entity_id for lbl in labels),
            event_times=tuple(lbl.event_time for lbl in labels),
            labels=tuple(lbl.label for lbl in labels),
            features={fid: tuple(vals) for fid, vals in columns.items()},
        )
