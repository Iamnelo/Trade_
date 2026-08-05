"""FeatureSetManifest: content-hashed record of a feature materialisation.

Mirrors DatasetManifest so downstream consumers (training runs, MLflow
ExperimentRecord, the reproducibility-hash pipeline) can treat feature and
dataset manifests uniformly. Adds:

- `derived_from`: dataset manifest IDs the feature was computed from
- `feature_spec_sha256`: hash of the feature's canonicalised spec
- `code_git_sha`: repo commit at materialisation time
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_spec_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_hex(canonical.encode())


@dataclass(frozen=True, slots=True)
class FeaturePartition:
    key: str
    sha256: str
    rows: int
    bytes: int
    event_time_min: datetime
    event_time_max: datetime


@dataclass
class FeatureSetManifest:
    feature_id: str
    entity_id: str
    feature_spec_sha256: str
    derived_from: list[str] = field(default_factory=list)
    code_git_sha: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    partitions: list[FeaturePartition] = field(default_factory=list)

    def add_partition(
        self,
        *,
        key: str,
        data: bytes,
        rows: int,
        event_time_min: datetime,
        event_time_max: datetime,
    ) -> None:
        self.partitions.append(
            FeaturePartition(
                key=key,
                sha256=sha256_hex(data),
                rows=rows,
                bytes=len(data),
                event_time_min=event_time_min,
                event_time_max=event_time_max,
            )
        )

    @property
    def total_rows(self) -> int:
        return sum(p.rows for p in self.partitions)

    @property
    def coverage_start(self) -> datetime | None:
        if not self.partitions:
            return None
        return min(p.event_time_min for p in self.partitions)

    @property
    def coverage_end(self) -> datetime | None:
        if not self.partitions:
            return None
        return max(p.event_time_max for p in self.partitions)

    def to_json(self) -> str:
        return json.dumps(
            {
                "feature_id": self.feature_id,
                "entity_id": self.entity_id,
                "feature_spec_sha256": self.feature_spec_sha256,
                "derived_from": sorted(self.derived_from),
                "code_git_sha": self.code_git_sha,
                "created_at": self.created_at.isoformat(),
                "partitions": [
                    {
                        "key": p.key,
                        "sha256": p.sha256,
                        "rows": p.rows,
                        "bytes": p.bytes,
                        "event_time_min": p.event_time_min.isoformat(),
                        "event_time_max": p.event_time_max.isoformat(),
                    }
                    for p in self.partitions
                ],
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, s: str) -> FeatureSetManifest:
        payload = json.loads(s)
        return cls(
            feature_id=payload["feature_id"],
            entity_id=payload["entity_id"],
            feature_spec_sha256=payload["feature_spec_sha256"],
            derived_from=list(payload.get("derived_from", [])),
            code_git_sha=payload.get("code_git_sha", ""),
            created_at=datetime.fromisoformat(payload["created_at"]),
            partitions=[
                FeaturePartition(
                    key=p["key"],
                    sha256=p["sha256"],
                    rows=p["rows"],
                    bytes=p["bytes"],
                    event_time_min=datetime.fromisoformat(p["event_time_min"]),
                    event_time_max=datetime.fromisoformat(p["event_time_max"]),
                )
                for p in payload.get("partitions", [])
            ],
        )


def compute_feature_spec_sha256(
    *,
    name: str,
    version: str,
    inputs: tuple[str, ...],
    lookback_bars: int,
    availability_delay_seconds: float,
    entity: str,
    interval: str,
) -> str:
    return canonical_spec_sha256(
        {
            "name": name,
            "version": version,
            "inputs": sorted(inputs),
            "lookback_bars": lookback_bars,
            "availability_delay_seconds": availability_delay_seconds,
            "entity": entity,
            "interval": interval,
        }
    )
