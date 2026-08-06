"""Append-only, sha256-chained audit log.

Every decision the platform makes — signal generated, order submitted, fill
received, kill switch tripped, reconciliation divergence, health-score
update — is written as an `AuditEvent` whose `sha` is
`sha256(canonical_json({sequence, kind, timestamp, payload, prev_sha}))`.
Each event's `prev_sha` equals the previous event's `sha`, so the whole
log is a cryptographic chain: any tampering with a middle event
invalidates every hash after it.

`AuditLog.verify_chain` re-derives every hash from the raw payload and
raises `AuditChainError` on the first mismatch.

Storage is pluggable via `AuditStore` (see `store.py` for the in-memory
and file-JSONL implementations). The log itself is stateless beyond the
current sequence and last sha, so it survives restarts by asking the
store for both on construction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from trade.utils.clock import utcnow


class AuditChainError(RuntimeError):
    """Raised when the sha chain fails to verify."""


@dataclass(frozen=True, slots=True)
class AuditEvent:
    sequence: int
    kind: str
    timestamp: datetime
    payload: dict[str, Any]
    prev_sha: str
    sha: str


def _canonical_hash(
    *,
    sequence: int,
    kind: str,
    timestamp: datetime,
    payload: dict[str, Any],
    prev_sha: str,
) -> str:
    canonical = json.dumps(
        {
            "sequence": sequence,
            "kind": kind,
            "timestamp": timestamp.isoformat(),
            "payload": payload,
            "prev_sha": prev_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditStore(Protocol):
    def append(self, event: AuditEvent) -> None: ...
    def iter_events(self) -> Iterator[AuditEvent]: ...
    def next_sequence(self) -> int: ...
    def last_sha(self) -> str: ...


class AuditLog:
    def __init__(self, store: AuditStore) -> None:
        self._store = store
        self._sequence = store.next_sequence()
        self._last_sha = store.last_sha()

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def last_sha(self) -> str:
        return self._last_sha

    def append(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        now: datetime | None = None,
    ) -> AuditEvent:
        if not kind:
            raise ValueError("kind must be non-empty")
        timestamp = now or utcnow()
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be UTC-aware")
        sha = _canonical_hash(
            sequence=self._sequence,
            kind=kind,
            timestamp=timestamp,
            payload=payload,
            prev_sha=self._last_sha,
        )
        event = AuditEvent(
            sequence=self._sequence,
            kind=kind,
            timestamp=timestamp,
            payload=payload,
            prev_sha=self._last_sha,
            sha=sha,
        )
        self._store.append(event)
        self._last_sha = sha
        self._sequence += 1
        return event

    def verify_chain(self) -> None:
        expected_prev = ""
        for expected_seq, event in enumerate(self._store.iter_events()):
            if event.sequence != expected_seq:
                raise AuditChainError(f"sequence gap at {event.sequence}: expected {expected_seq}")
            if event.prev_sha != expected_prev:
                raise AuditChainError(
                    f"prev_sha mismatch at sequence {event.sequence}: "
                    f"stored={event.prev_sha!r} expected={expected_prev!r}"
                )
            recomputed = _canonical_hash(
                sequence=event.sequence,
                kind=event.kind,
                timestamp=event.timestamp,
                payload=event.payload,
                prev_sha=event.prev_sha,
            )
            if recomputed != event.sha:
                raise AuditChainError(
                    f"sha mismatch at sequence {event.sequence}: "
                    f"stored={event.sha!r} recomputed={recomputed!r}"
                )
            expected_prev = event.sha
