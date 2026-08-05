"""UTC-first time utilities.

Every timestamp inside the platform is UTC. Naive datetimes anywhere in
trading, feature, or logging code are a defect.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_epoch_ms(ts: datetime) -> int:
    if ts.tzinfo is None:
        raise ValueError("Naive datetime is not permitted; use a UTC-aware datetime")
    return int(ts.timestamp() * 1000)


def from_epoch_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)
