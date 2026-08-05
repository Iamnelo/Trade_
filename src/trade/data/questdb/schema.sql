-- QuestDB schema for curated kline data.
--
-- QuestDB speaks the Postgres wire protocol on port 8812. Idempotency is
-- provided by DEDUP UPSERT KEYS so replayed batches (backfill + live +
-- reconciliation) collapse to the newest ingest_time per natural key.
--
-- This DDL is applied by trade.data.questdb.writer.ensure_schema(). Keep in
-- sync with src/trade/data/schemas.py::KLINE_COLUMNS.

CREATE TABLE IF NOT EXISTS klines (
    source SYMBOL CAPACITY 8 CACHE,
    category SYMBOL CAPACITY 4 CACHE,
    symbol SYMBOL CAPACITY 32 CACHE,
    interval SYMBOL CAPACITY 16 CACHE,
    event_time TIMESTAMP,
    ingest_time TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    turnover DOUBLE
) TIMESTAMP(event_time) PARTITION BY MONTH WAL
  DEDUP UPSERT KEYS(event_time, source, category, symbol, interval);
