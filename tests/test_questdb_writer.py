"""Tests for QuestDBKlineWriter using mocked psycopg connections."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from trade.data.questdb.writer import QuestDBKlineWriter
from trade.data.schemas import KlineRecord


def _kline(hour: int) -> KlineRecord:
    return KlineRecord(
        source="bybit",
        category="linear",
        symbol="BTCUSDT",
        interval="60",
        event_time=datetime(2024, 1, 1, hour, tzinfo=UTC),
        ingest_time=datetime(2024, 1, 1, hour, 0, 5, tzinfo=UTC),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
        turnover=1005.0,
    )


def _fake_conn() -> tuple[MagicMock, MagicMock]:
    conn = MagicMock()
    cur = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = None
    conn.cursor.return_value.__enter__.return_value = cur
    conn.cursor.return_value.__exit__.return_value = None
    return conn, cur


def test_write_empty_batch_is_a_noop() -> None:
    conn, cur = _fake_conn()
    writer = QuestDBKlineWriter(connection_factory=lambda: conn)
    assert writer.write([]) == 0
    cur.executemany.assert_not_called()


def test_write_calls_executemany_with_all_rows() -> None:
    conn, cur = _fake_conn()
    writer = QuestDBKlineWriter(connection_factory=lambda: conn)
    records = [_kline(0), _kline(1)]
    written = writer.write(records)
    assert written == 2

    cur.executemany.assert_called_once()
    sql, rows = cur.executemany.call_args.args
    assert sql.startswith("INSERT INTO klines")
    assert len(rows) == 2
    assert rows[0][0] == "bybit"
    assert rows[0][2] == "BTCUSDT"
    conn.commit.assert_called_once()


def test_ensure_schema_executes_each_ddl_statement() -> None:
    conn, cur = _fake_conn()
    writer = QuestDBKlineWriter(connection_factory=lambda: conn)
    writer.ensure_schema()

    # schema.sql has one CREATE TABLE — one execute call, one commit.
    assert cur.execute.call_count == 1
    assert "klines" in cur.execute.call_args.args[0]
    conn.commit.assert_called_once()


def test_latest_event_times_query_and_result_shape() -> None:
    conn, cur = _fake_conn()
    events = [
        (datetime(2024, 1, 1, 5, tzinfo=UTC),),
        (datetime(2024, 1, 1, 4, tzinfo=UTC),),
    ]
    cur.fetchall.return_value = events
    writer = QuestDBKlineWriter(connection_factory=lambda: conn)

    got = writer.latest_event_times(source="bybit", symbol="BTCUSDT", interval="60", limit=10)
    assert got == [e[0] for e in events]
    sql, params = cur.execute.call_args.args
    assert "ORDER BY event_time DESC" in sql
    assert params == ("bybit", "BTCUSDT", "60", 10)
