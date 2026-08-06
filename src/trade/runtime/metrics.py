"""Prometheus metrics for runtime subsystems (kill switch, reconciler, health).

Uses an isolated `CollectorRegistry` so tests can spin up an independent
instance without leaking counters across cases — same pattern as
`trade.data.quality.metrics.build_metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    registry: CollectorRegistry
    kill_switch_trips_total: Counter
    kill_switch_active: Gauge  # per-reason 0/1
    reconciler_iterations_total: Counter
    reconciler_divergences_total: Counter
    reconciler_last_divergence_qty: Gauge
    audit_events_total: Counter
    health_score: Gauge
    health_sub_score: Gauge  # per-sub-score

    def observe_kill_switch_trip(self, *, reason: str) -> None:
        self.kill_switch_trips_total.labels(reason=reason).inc()
        self.kill_switch_active.labels(reason=reason).set(1)

    def observe_kill_switch_clear(self, *, reason: str) -> None:
        self.kill_switch_active.labels(reason=reason).set(0)

    def observe_reconciler_iteration(self) -> None:
        self.reconciler_iterations_total.inc()

    def observe_reconciler_divergence(self, *, symbol: str, delta_qty: float) -> None:
        self.reconciler_divergences_total.labels(symbol=symbol).inc()
        self.reconciler_last_divergence_qty.labels(symbol=symbol).set(delta_qty)

    def observe_audit_event(self, *, kind: str) -> None:
        self.audit_events_total.labels(kind=kind).inc()

    def observe_health(self, *, composite: float, sub_scores: dict[str, float]) -> None:
        self.health_score.set(composite)
        for name, value in sub_scores.items():
            self.health_sub_score.labels(name=name).set(value)


def build_runtime_metrics(registry: CollectorRegistry | None = None) -> RuntimeMetrics:
    r = registry or CollectorRegistry()
    return RuntimeMetrics(
        registry=r,
        kill_switch_trips_total=Counter(
            "trade_kill_switch_trips_total",
            "Kill-switch trips grouped by reason.",
            ("reason",),
            registry=r,
        ),
        kill_switch_active=Gauge(
            "trade_kill_switch_active",
            "Whether the given kill-switch reason is currently active (1) or cleared (0).",
            ("reason",),
            registry=r,
        ),
        reconciler_iterations_total=Counter(
            "trade_reconciler_iterations_total",
            "Total reconciliation passes attempted.",
            registry=r,
        ),
        reconciler_divergences_total=Counter(
            "trade_reconciler_divergences_total",
            "Reconciliation passes that produced a divergence, per symbol.",
            ("symbol",),
            registry=r,
        ),
        reconciler_last_divergence_qty=Gauge(
            "trade_reconciler_last_divergence_qty",
            "Last observed venue - local quantity delta, per symbol.",
            ("symbol",),
            registry=r,
        ),
        audit_events_total=Counter(
            "trade_audit_events_total",
            "Audit-log events appended, grouped by kind.",
            ("kind",),
            registry=r,
        ),
        health_score=Gauge(
            "trade_ai_health_score",
            "Composite AI Health Score (0-100).",
            registry=r,
        ),
        health_sub_score=Gauge(
            "trade_ai_health_sub_score",
            "Individual sub-score contributing to the composite (0-100).",
            ("name",),
            registry=r,
        ),
    )
