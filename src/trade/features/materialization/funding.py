"""Funding-rate feature materialisation.

Funding data does not fit the Feature Protocol's KlineRecord history shape
(funding is event-driven, not bar-aligned), so instead of a `Feature`
subclass we materialise directly into the FeatureStore via
`put_materialized`. From the training pipeline's point of view the result is
indistinguishable from a kline-derived feature: `point_in_time_join` returns
whatever the newest available funding value was at each label event_time.

Two materialisers are provided:

- `materialize_funding_rate`: one row per funding settlement, value = the
  settled rate. availability_time = settlement time (Bybit publishes the
  final rate at settlement).
- `materialize_funding_rate_mean`: rolling mean over the last N
  settlements; one row per settlement once enough history exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from trade.data.schemas import FundingRecord
from trade.features.types import FeatureSpec, MaterializedFeature

FUNDING_RATE_SPEC = FeatureSpec(
    name="funding_rate",
    version="1",
    inputs=("funding_rate",),
    lookback_bars=1,
    availability_delay=timedelta(0),
    entity="symbol",
    interval="funding",  # not a kline interval — funding is 8-hourly settlements
)


def funding_rate_mean_spec(window: int) -> FeatureSpec:
    if window < 2:
        raise ValueError("window must be >= 2")
    return FeatureSpec(
        name="funding_rate_mean",
        version=str(window),
        inputs=("funding_rate",),
        lookback_bars=window,
        availability_delay=timedelta(0),
        entity="symbol",
        interval="funding",
    )


def materialize_funding_rate(
    funding_records: Sequence[FundingRecord],
) -> list[MaterializedFeature]:
    """One MaterializedFeature per settlement."""
    sorted_rows = sorted(funding_records, key=lambda r: r.event_time)
    return [
        MaterializedFeature(
            feature_id=FUNDING_RATE_SPEC.full_id,
            entity_id=r.symbol,
            event_time=r.event_time,
            availability_time=r.event_time + FUNDING_RATE_SPEC.availability_delay,
            value=r.funding_rate,
        )
        for r in sorted_rows
    ]


def materialize_funding_rate_mean(
    funding_records: Sequence[FundingRecord],
    *,
    window: int,
) -> list[MaterializedFeature]:
    """Rolling mean of the last `window` settlements, per symbol."""
    spec = funding_rate_mean_spec(window)
    by_symbol: dict[str, list[FundingRecord]] = {}
    for rec in sorted(funding_records, key=lambda r: r.event_time):
        by_symbol.setdefault(rec.symbol, []).append(rec)

    out: list[MaterializedFeature] = []
    for symbol, stream in by_symbol.items():
        for i in range(window - 1, len(stream)):
            window_slice = stream[i - window + 1 : i + 1]
            mean = sum(r.funding_rate for r in window_slice) / window
            current = stream[i]
            out.append(
                MaterializedFeature(
                    feature_id=spec.full_id,
                    entity_id=symbol,
                    event_time=current.event_time,
                    availability_time=current.event_time + spec.availability_delay,
                    value=mean,
                )
            )
    return out
