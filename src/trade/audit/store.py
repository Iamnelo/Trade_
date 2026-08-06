"""Audit-log storage backends: in-memory and append-only JSONL file."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from trade.audit.log import AuditEvent


def _event_to_json(event: AuditEvent) -> str:
    return json.dumps(
        {
            "sequence": event.sequence,
            "kind": event.kind,
            "timestamp": event.timestamp.isoformat(),
            "payload": event.payload,
            "prev_sha": event.prev_sha,
            "sha": event.sha,
        },
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _event_from_json(line: str) -> AuditEvent:
    raw = json.loads(line)
    return AuditEvent(
        sequence=int(raw["sequence"]),
        kind=str(raw["kind"]),
        timestamp=datetime.fromisoformat(raw["timestamp"]),
        payload=dict(raw["payload"]),
        prev_sha=str(raw["prev_sha"]),
        sha=str(raw["sha"]),
    )


class InMemoryAuditStore:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self._events.append(event)

    def iter_events(self) -> Iterator[AuditEvent]:
        yield from self._events

    def next_sequence(self) -> int:
        return len(self._events)

    def last_sha(self) -> str:
        return self._events[-1].sha if self._events else ""


class FileAuditStore:
    """Append-only JSONL — one event per line, never rewritten."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Precompute sequence + last_sha by streaming any existing file.
        self._count = 0
        self._last_sha = ""
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    event = _event_from_json(line)
                    self._count = event.sequence + 1
                    self._last_sha = event.sha

    def append(self, event: AuditEvent) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(_event_to_json(event))
            fh.write("\n")
        self._count = event.sequence + 1
        self._last_sha = event.sha

    def iter_events(self) -> Iterator[AuditEvent]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                yield _event_from_json(line)

    def next_sequence(self) -> int:
        return self._count

    def last_sha(self) -> str:
        return self._last_sha
