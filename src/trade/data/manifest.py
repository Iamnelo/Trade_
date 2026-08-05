"""Dataset manifests: content-hashed record of which partitions belong to a
dataset snapshot.

Committed alongside training runs so a model can always be retrained against
the exact bytes it originally saw.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class Partition:
    key: str
    sha256: str
    rows: int
    bytes: int
    event_time_min: datetime
    event_time_max: datetime


@dataclass
class DatasetManifest:
    dataset: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    partitions: list[Partition] = field(default_factory=list)

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
            Partition(
                key=key,
                sha256=sha256_hex(data),
                rows=rows,
                bytes=len(data),
                event_time_min=event_time_min,
                event_time_max=event_time_max,
            )
        )

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

    @property
    def total_rows(self) -> int:
        return sum(p.rows for p in self.partitions)

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "dataset": self.dataset,
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
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> DatasetManifest:
        payload = json.loads(s)
        return cls(
            dataset=payload["dataset"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            partitions=[
                Partition(
                    key=p["key"],
                    sha256=p["sha256"],
                    rows=p["rows"],
                    bytes=p["bytes"],
                    event_time_min=datetime.fromisoformat(p["event_time_min"]),
                    event_time_max=datetime.fromisoformat(p["event_time_max"]),
                )
                for p in payload["partitions"]
            ],
        )
