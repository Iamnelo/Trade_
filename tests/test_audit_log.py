"""Tests for the sha-chained audit log + storage backends."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade.audit.log import AuditChainError, AuditLog
from trade.audit.store import FileAuditStore, InMemoryAuditStore


def _at(hour: int) -> datetime:
    return datetime(2024, 1, 1, hour, tzinfo=UTC)


def test_append_increments_sequence_and_chains_sha() -> None:
    log = AuditLog(InMemoryAuditStore())
    e1 = log.append(kind="signal", payload={"symbol": "BTCUSDT", "class": "up"}, now=_at(0))
    e2 = log.append(kind="order_submitted", payload={"coid": "abc"}, now=_at(1))

    assert e1.sequence == 0
    assert e1.prev_sha == ""
    assert e2.sequence == 1
    assert e2.prev_sha == e1.sha
    assert e1.sha != e2.sha


def test_verify_chain_passes_on_untampered_log() -> None:
    store = InMemoryAuditStore()
    log = AuditLog(store)
    for i in range(5):
        log.append(kind="tick", payload={"i": i}, now=_at(i))
    log.verify_chain()


def test_verify_chain_detects_payload_tampering() -> None:
    store = InMemoryAuditStore()
    log = AuditLog(store)
    log.append(kind="signal", payload={"x": 1}, now=_at(0))
    log.append(kind="fill", payload={"y": 2}, now=_at(1))
    # Tamper: mutate the private events list via replace().
    tampered = replace(store._events[0], payload={"x": 999})
    store._events[0] = tampered
    with pytest.raises(AuditChainError, match="sha mismatch"):
        log.verify_chain()


def test_verify_chain_detects_deleted_event() -> None:
    store = InMemoryAuditStore()
    log = AuditLog(store)
    log.append(kind="a", payload={}, now=_at(0))
    log.append(kind="b", payload={}, now=_at(1))
    log.append(kind="c", payload={}, now=_at(2))
    del store._events[1]
    with pytest.raises(AuditChainError):
        log.verify_chain()


def test_append_rejects_empty_kind() -> None:
    log = AuditLog(InMemoryAuditStore())
    with pytest.raises(ValueError, match="kind"):
        log.append(kind="", payload={}, now=_at(0))


def test_append_rejects_naive_timestamp() -> None:
    log = AuditLog(InMemoryAuditStore())
    with pytest.raises(ValueError, match="UTC-aware"):
        log.append(kind="signal", payload={}, now=datetime(2024, 1, 1))


def test_file_audit_store_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(FileAuditStore(path))
    log.append(kind="signal", payload={"symbol": "BTCUSDT"}, now=_at(0))
    log.append(kind="fill", payload={"qty": 0.5}, now=_at(1))

    # New process opens the store from disk — sequence and last_sha carry over.
    log2 = AuditLog(FileAuditStore(path))
    assert log2.sequence == 2
    assert log2.last_sha == log.last_sha
    log2.append(kind="tick", payload={"n": 1}, now=_at(2))
    log2.verify_chain()


def test_file_audit_store_never_rewrites_existing_lines(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(FileAuditStore(path))
    log.append(kind="a", payload={}, now=_at(0))
    original = path.read_text()

    log.append(kind="b", payload={}, now=_at(1))
    updated = path.read_text()

    # The first line must be byte-identical between the two versions.
    assert updated.startswith(original)
    # And exactly one new line was appended.
    assert len(updated.splitlines()) == len(original.splitlines()) + 1


def test_file_audit_store_json_format_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(FileAuditStore(path))
    event = log.append(kind="signal", payload={"symbol": "BTCUSDT"}, now=_at(0))
    stored = json.loads(path.read_text().splitlines()[0])
    assert stored["sequence"] == 0
    assert stored["kind"] == "signal"
    assert stored["payload"] == {"symbol": "BTCUSDT"}
    assert stored["prev_sha"] == ""
    assert stored["sha"] == event.sha


def test_in_memory_store_iter_returns_all_events() -> None:
    store = InMemoryAuditStore()
    log = AuditLog(store)
    for i in range(3):
        log.append(kind="k", payload={"i": i}, now=_at(i))
    kinds = [e.kind for e in store.iter_events()]
    assert kinds == ["k", "k", "k"]


def test_timestamps_survive_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(FileAuditStore(path))
    log.append(kind="a", payload={}, now=_at(0))
    log.append(kind="b", payload={}, now=_at(1) + timedelta(minutes=30))
    reloaded = list(FileAuditStore(path).iter_events())
    assert reloaded[0].timestamp == _at(0)
    assert reloaded[1].timestamp == _at(1) + timedelta(minutes=30)
