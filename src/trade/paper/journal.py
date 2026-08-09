"""Tamper-evident journal for the paper-trading system.

Every decision, order, fill, exit, halt, and lifecycle event is written twice:

1. To a sha256-chained `AuditLog` (append-only) — the authoritative,
   tamper-evident record. `verify()` re-derives the whole chain.
2. To a human-readable decisions JSONL, one event per line, for quick review.

Both live under the configured journal directory and survive restarts (the
audit store recovers its sequence + last sha from the existing file).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from trade.audit.log import AuditLog
from trade.audit.store import FileAuditStore
from trade.utils.clock import utcnow


class PaperJournal:
    def __init__(self, journal_dir: Path) -> None:
        self._dir = Path(journal_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._audit = AuditLog(FileAuditStore(self._dir / "audit.jsonl"))
        self._decisions_path = self._dir / "decisions.jsonl"

    @property
    def audit(self) -> AuditLog:
        return self._audit

    @property
    def last_sha(self) -> str:
        return self._audit.last_sha

    def record(self, kind: str, payload: dict[str, Any], *, now: datetime | None = None) -> None:
        ts = now or utcnow()
        self._audit.append(kind=kind, payload=payload, now=ts)
        line = json.dumps(
            {"timestamp": ts.isoformat(), "kind": kind, **payload},
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        with self._decisions_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def verify(self) -> None:
        """Raise AuditChainError if the sha chain has been tampered with."""
        self._audit.verify_chain()
