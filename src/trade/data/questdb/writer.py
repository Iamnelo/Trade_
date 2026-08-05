"""QuestDB writer for kline records.

Talks to QuestDB via the Postgres wire protocol (psycopg 3). Idempotency is
provided by the table's DEDUP UPSERT KEYS clause, so replayed batches from
backfill, live ingest, and reconciliation collapse to the newest ingest_time
per natural key.

The connection factory is injected so unit tests can supply a fake without
touching real QuestDB.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import psycopg

from trade.data.schemas import KlineRecord

_SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"

_INSERT_SQL = (
    "INSERT INTO klines "
    "(source, category, symbol, interval, event_time, ingest_time, "
    " open, high, low, close, volume, turnover) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)


class DBConnection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> DBConnection: ...
    def __exit__(self, *args: object) -> None: ...


ConnectionFactory = Callable[[], DBConnection]


def default_connection_factory(
    *,
    host: str = "localhost",
    port: int = 8812,
    user: str = "admin",
    password: str = "quest",
    dbname: str = "qdb",
) -> ConnectionFactory:
    """Return a factory that opens a fresh psycopg connection on each call."""

    def _connect() -> DBConnection:
        # psycopg.connect is typed as returning a concrete Connection; the
        # abstract DBConnection Protocol above lets tests inject fakes.
        return psycopg.connect(  # type: ignore[return-value]
            host=host, port=port, user=user, password=password, dbname=dbname
        )

    return _connect


class QuestDBKlineWriter:
    def __init__(self, connection_factory: ConnectionFactory) -> None:
        self._connect = connection_factory

    def ensure_schema(self) -> None:
        ddl = _SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        with self._connect() as conn, conn.cursor() as cur:
            # QuestDB accepts one statement per execute; split on ';' and skip empties.
            for statement in (s.strip() for s in ddl.split(";")):
                if statement:
                    cur.execute(statement)
            conn.commit()

    def write(self, records: Sequence[KlineRecord]) -> int:
        if not records:
            return 0
        rows = [
            (
                r.source,
                r.category,
                r.symbol,
                r.interval,
                r.event_time,
                r.ingest_time,
                r.open,
                r.high,
                r.low,
                r.close,
                r.volume,
                r.turnover,
            )
            for r in records
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(_INSERT_SQL, rows)
            conn.commit()
        return len(rows)

    def latest_event_times(
        self,
        *,
        source: str,
        symbol: str,
        interval: str,
        limit: int = 1000,
    ) -> list[Any]:
        """Return the newest `limit` event_time values for a stream, ordered DESC."""
        sql = (
            "SELECT event_time FROM klines "
            "WHERE source = %s AND symbol = %s AND interval = %s "
            "ORDER BY event_time DESC LIMIT %s"
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (source, symbol, interval, limit))
            return [row[0] for row in cur.fetchall()]
