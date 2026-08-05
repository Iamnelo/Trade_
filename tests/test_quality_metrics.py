"""Tests for the DQ Prometheus metrics registry."""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from trade.data.quality.metrics import build_metrics


def _sample_value(registry: CollectorRegistry, metric_name: str, labels: dict[str, str]) -> float:
    value = registry.get_sample_value(metric_name, labels=labels)
    if value is None:
        raise AssertionError(
            f"No sample for {metric_name} with {labels!r}. Samples: {list(registry.collect())}"
        )
    return value


def test_isolated_registry_avoids_leaks_across_calls() -> None:
    m1 = build_metrics()
    m2 = build_metrics()
    # Distinct registries = safe to build many independent metrics in the same test run.
    assert m1.registry is not m2.registry


def test_observe_message_increments_counter() -> None:
    m = build_metrics()
    m.observe_message(source="bybit", symbol="BTCUSDT", interval="60")
    m.observe_message(source="bybit", symbol="BTCUSDT", interval="60")
    value = _sample_value(
        m.registry,
        "trade_stream_messages_total",
        {"source": "bybit", "symbol": "BTCUSDT", "interval": "60"},
    )
    assert value == 2.0


def test_observe_staleness_sets_gauge() -> None:
    m = build_metrics()
    m.observe_staleness(source="bybit", symbol="ETHUSDT", interval="60", seconds=42.5)
    value = _sample_value(
        m.registry,
        "trade_staleness_seconds",
        {"source": "bybit", "symbol": "ETHUSDT", "interval": "60"},
    )
    assert value == 42.5


def test_observe_cross_source_delta_sets_gauge() -> None:
    m = build_metrics()
    m.observe_cross_source_delta(symbol="BTCUSDT", interval="60", max_abs_delta_bps=3.5)
    value = _sample_value(
        m.registry,
        "trade_cross_source_max_abs_delta_bps",
        {"symbol": "BTCUSDT", "interval": "60"},
    )
    assert value == 3.5


def test_reconciler_fill_counter() -> None:
    m = build_metrics()
    m.observe_reconciler_fill(source="bybit", symbol="BTCUSDT", interval="60")
    value = _sample_value(
        m.registry,
        "trade_reconciler_gaps_filled_total",
        {"source": "bybit", "symbol": "BTCUSDT", "interval": "60"},
    )
    assert value == 1.0
