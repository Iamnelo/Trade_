"""Prometheus metrics for data-quality observability.

Uses a custom `CollectorRegistry` so tests can spin up an isolated instance
without leaking across pytest cases. Production code calls `get_metrics()`
which returns the module-level singleton bound to the default registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge


@dataclass(frozen=True, slots=True)
class DQMetrics:
    """Bundle of the DQ gauges/counters. Labels: source, symbol, interval."""

    registry: CollectorRegistry
    stream_messages_total: Counter
    stream_reconnects_total: Counter
    last_event_time_seconds: Gauge
    staleness_seconds: Gauge
    gap_percent: Gauge
    price_sanity_violations_total: Counter
    cross_source_max_abs_delta_bps: Gauge
    reconciler_gaps_filled_total: Counter
    reconciler_errors_total: Counter

    def observe_message(self, *, source: str, symbol: str, interval: str) -> None:
        self.stream_messages_total.labels(source=source, symbol=symbol, interval=interval).inc()

    def observe_reconnect(self, *, source: str) -> None:
        self.stream_reconnects_total.labels(source=source).inc()

    def observe_event_time(
        self, *, source: str, symbol: str, interval: str, epoch_seconds: float
    ) -> None:
        self.last_event_time_seconds.labels(source=source, symbol=symbol, interval=interval).set(
            epoch_seconds
        )

    def observe_staleness(self, *, source: str, symbol: str, interval: str, seconds: float) -> None:
        self.staleness_seconds.labels(source=source, symbol=symbol, interval=interval).set(seconds)

    def observe_gap_percent(self, *, source: str, symbol: str, interval: str, pct: float) -> None:
        self.gap_percent.labels(source=source, symbol=symbol, interval=interval).set(pct)

    def observe_price_sanity_violation(self, *, source: str, symbol: str, interval: str) -> None:
        self.price_sanity_violations_total.labels(
            source=source, symbol=symbol, interval=interval
        ).inc()

    def observe_cross_source_delta(
        self, *, symbol: str, interval: str, max_abs_delta_bps: float
    ) -> None:
        self.cross_source_max_abs_delta_bps.labels(symbol=symbol, interval=interval).set(
            max_abs_delta_bps
        )

    def observe_reconciler_fill(self, *, source: str, symbol: str, interval: str) -> None:
        self.reconciler_gaps_filled_total.labels(
            source=source, symbol=symbol, interval=interval
        ).inc()

    def observe_reconciler_error(self, *, source: str) -> None:
        self.reconciler_errors_total.labels(source=source).inc()


def build_metrics(registry: CollectorRegistry | None = None) -> DQMetrics:
    r = registry or CollectorRegistry()
    return DQMetrics(
        registry=r,
        stream_messages_total=Counter(
            "trade_stream_messages_total",
            "Total messages received from the stream ingestor.",
            ("source", "symbol", "interval"),
            registry=r,
        ),
        stream_reconnects_total=Counter(
            "trade_stream_reconnects_total",
            "Total WebSocket reconnects.",
            ("source",),
            registry=r,
        ),
        last_event_time_seconds=Gauge(
            "trade_last_event_time_seconds",
            "Epoch seconds of the most recently seen bar for a stream.",
            ("source", "symbol", "interval"),
            registry=r,
        ),
        staleness_seconds=Gauge(
            "trade_staleness_seconds",
            "Wall-clock seconds since the last observed bar's event_time.",
            ("source", "symbol", "interval"),
            registry=r,
        ),
        gap_percent=Gauge(
            "trade_gap_percent",
            "Percentage of expected bars missing in the last evaluation window.",
            ("source", "symbol", "interval"),
            registry=r,
        ),
        price_sanity_violations_total=Counter(
            "trade_price_sanity_violations_total",
            "Bars flagged by the rolling-median price-sanity band.",
            ("source", "symbol", "interval"),
            registry=r,
        ),
        cross_source_max_abs_delta_bps=Gauge(
            "trade_cross_source_max_abs_delta_bps",
            "Latest max absolute cross-source close delta in basis points.",
            ("symbol", "interval"),
            registry=r,
        ),
        reconciler_gaps_filled_total=Counter(
            "trade_reconciler_gaps_filled_total",
            "Historical bars re-fetched by the reconciler to close a live-stream gap.",
            ("source", "symbol", "interval"),
            registry=r,
        ),
        reconciler_errors_total=Counter(
            "trade_reconciler_errors_total",
            "Reconciler iterations that raised an error.",
            ("source",),
            registry=r,
        ),
    )
