"""Backfill CLI: pull historical klines and write parquet + manifest."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import typer

from trade.config import get_settings
from trade.data.backfill.binance import backfill_binance_klines
from trade.data.backfill.bybit import backfill_bybit_klines
from trade.data.backfill.cross_validate import compare_klines
from trade.data.manifest import DatasetManifest
from trade.data.parquet import klines_to_parquet_bytes
from trade.data.schemas import KlineRecord, partition_key
from trade.data.storage import LocalStore, S3Store, Store
from trade.exchanges.binance_public import BinancePublicClient
from trade.exchanges.bybit_public import BybitPublicClient
from trade.logging_setup import configure_logging, get_logger

backfill_app = typer.Typer(no_args_is_help=True)


def _parse_iso(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _open_store(
    local_dir: Path | None,
    s3_bucket: str | None,
    s3_endpoint: str | None,
) -> Store:
    if s3_bucket is not None:
        return S3Store(bucket=s3_bucket, endpoint_url=s3_endpoint)
    return LocalStore(local_dir or Path("data"))


def _group_by_month(
    records: list[KlineRecord],
) -> dict[tuple[int, int], list[KlineRecord]]:
    buckets: dict[tuple[int, int], list[KlineRecord]] = defaultdict(list)
    for r in records:
        buckets[(r.event_time.year, r.event_time.month)].append(r)
    return dict(buckets)


async def _collect(source: str, iterator: AsyncIterator[list[KlineRecord]]) -> list[KlineRecord]:
    log = get_logger(f"cli.backfill.{source}")
    out: list[KlineRecord] = []
    async for batch in iterator:
        out.extend(batch)
        log.info("batch", size=len(batch), total=len(out))
    return out


def _write_partitions(
    *,
    store: Store,
    records: list[KlineRecord],
    dataset: str,
    source: str,
    category: str,
    symbol: str,
    interval: str,
) -> DatasetManifest:
    log = get_logger(f"cli.backfill.{source}")
    manifest = DatasetManifest(dataset=f"{dataset}_{symbol}_{interval}")
    for (year, month), records_for_month in sorted(_group_by_month(records).items()):
        key = partition_key(
            dataset=dataset,
            source=source,
            category=category,
            symbol=symbol,
            interval=interval,
            year=year,
            month=month,
        )
        data = klines_to_parquet_bytes(records_for_month)
        store.put(key, data)
        event_times = [r.event_time for r in records_for_month]
        manifest.add_partition(
            key=key,
            data=data,
            rows=len(records_for_month),
            event_time_min=min(event_times),
            event_time_max=max(event_times),
        )
        log.info("wrote_partition", key=key, rows=len(records_for_month), bytes=len(data))

    manifest_key = f"manifests/{dataset}/{symbol}_{interval}.json"
    store.put(manifest_key, manifest.to_json().encode("utf-8"))
    log.info("wrote_manifest", key=manifest_key, partitions=len(manifest.partitions))
    return manifest


@backfill_app.command("bybit")
def bybit(
    symbol: str = typer.Option(..., help="Bybit symbol, e.g., BTCUSDT"),
    interval: str = typer.Option("60", help="Bybit interval, e.g., 60 or D"),
    start: str = typer.Option(..., help="ISO-8601 start (inclusive, UTC assumed if naive)"),
    end: str = typer.Option(..., help="ISO-8601 end (exclusive, UTC assumed if naive)"),
    category: str = typer.Option("linear"),
    batch_size: int = typer.Option(1000),
    local_dir: Path | None = typer.Option(None, help="Local storage root"),
    s3_bucket: str | None = typer.Option(None, help="S3-compatible bucket name"),
    s3_endpoint: str | None = typer.Option(None, help="S3 endpoint URL (MinIO in dev)"),
) -> None:
    """Backfill Bybit historical klines into parquet with a content-hashed manifest."""
    configure_logging()
    settings = get_settings()

    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)
    store = _open_store(local_dir, s3_bucket, s3_endpoint)

    async def _run() -> None:
        async with BybitPublicClient(base_url=settings.bybit_base_url, category=category) as client:
            records = await _collect(
                "bybit",
                backfill_bybit_klines(
                    client,
                    symbol=symbol,
                    interval=interval,
                    start=start_dt,
                    end=end_dt,
                    category=category,
                    batch_size=batch_size,
                ),
            )
        _write_partitions(
            store=store,
            records=records,
            dataset="bybit_kline",
            source="bybit",
            category=category,
            symbol=symbol,
            interval=interval,
        )

    asyncio.run(_run())


@backfill_app.command("binance")
def binance(
    symbol: str = typer.Option(..., help="Binance symbol, e.g., BTCUSDT"),
    interval: str = typer.Option("60"),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    category: str = typer.Option("linear"),
    batch_size: int = typer.Option(1500),
    local_dir: Path | None = typer.Option(None),
    s3_bucket: str | None = typer.Option(None),
    s3_endpoint: str | None = typer.Option(None),
) -> None:
    """Backfill Binance USDT-M perp historical klines (cross-validation source)."""
    configure_logging()

    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)
    store = _open_store(local_dir, s3_bucket, s3_endpoint)

    async def _run() -> None:
        async with BinancePublicClient() as client:
            records = await _collect(
                "binance",
                backfill_binance_klines(
                    client,
                    symbol=symbol,
                    interval=interval,
                    start=start_dt,
                    end=end_dt,
                    category=category,
                    batch_size=batch_size,
                ),
            )
        _write_partitions(
            store=store,
            records=records,
            dataset="binance_kline",
            source="binance",
            category=category,
            symbol=symbol,
            interval=interval,
        )

    asyncio.run(_run())


@backfill_app.command("cross-validate")
def cross_validate(
    symbol: str = typer.Option(...),
    interval: str = typer.Option("60"),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    category: str = typer.Option("linear"),
) -> None:
    """Fetch klines from Bybit and Binance in-memory and report cross-source deltas."""
    configure_logging()
    log = get_logger("cli.backfill.cross_validate")
    settings = get_settings()

    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)

    async def _run() -> None:
        async with (
            BybitPublicClient(base_url=settings.bybit_base_url, category=category) as bybit_client,
            BinancePublicClient() as binance_client,
        ):
            bybit_records = await _collect(
                "bybit",
                backfill_bybit_klines(
                    bybit_client,
                    symbol=symbol,
                    interval=interval,
                    start=start_dt,
                    end=end_dt,
                    category=category,
                ),
            )
            binance_records = await _collect(
                "binance",
                backfill_binance_klines(
                    binance_client,
                    symbol=symbol,
                    interval=interval,
                    start=start_dt,
                    end=end_dt,
                    category=category,
                ),
            )
        report = compare_klines(bybit_records, binance_records)
        log.info(
            "cross_validate_report",
            common_bars=report.common_bars,
            bybit_only_bars=report.a_only_bars,
            binance_only_bars=report.b_only_bars,
            max_abs_delta_bps=report.max_abs_delta_bps,
            median_abs_delta_bps=report.median_abs_delta_bps,
        )

    asyncio.run(_run())
