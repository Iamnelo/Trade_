"""Multi-symbol feature Protocol + materialisation + contract helpers.

Cross-asset features (e.g., BTC-ETH spread) need aligned histories from
multiple symbols, not the single-symbol `Sequence[KlineRecord]` that the
regular `Feature` Protocol carries. Instead of overloading `Feature`, this
module introduces a parallel `MultiSymbolFeature` Protocol with a
`compute(histories: Mapping[symbol, Sequence[KlineRecord]])` signature.

Materialisation walks the `primary_symbol`'s bars. For each primary bar at
time t, aligned histories for every required symbol are built with all bars
whose `event_time <= t`. The feature must be a pure function of those
histories AND respect its declared `lookback_bars` on every symbol — the
contract helpers below verify both.

Feature-store consumption is unchanged: multi-symbol features produce the
same `MaterializedFeature` rows and go through the SAME `point_in_time_join`
API. Cross-asset semantics live entirely at the feature level.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Protocol

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec, MaterializedFeature


class MultiSymbolFeature(Protocol):
    @property
    def spec(self) -> FeatureSpec: ...

    @property
    def primary_symbol(self) -> str: ...

    @property
    def required_symbols(self) -> tuple[str, ...]: ...

    def compute(
        self,
        histories: Mapping[str, Sequence[KlineRecord]],
    ) -> float | None: ...


def _bars_up_to(bars: Sequence[KlineRecord], t: datetime) -> list[KlineRecord]:
    """Return bars with event_time <= t (bisect-based, O(log n) lookup)."""
    times = [b.event_time for b in bars]
    idx = bisect_right(times, t)
    return list(bars[:idx])


def materialize_multi_symbol_feature(
    feature: MultiSymbolFeature,
    *,
    bars_by_symbol: Mapping[str, Sequence[KlineRecord]],
    entity_id: str,
) -> list[MaterializedFeature]:
    primary = feature.primary_symbol
    required = tuple(feature.required_symbols)
    if primary not in bars_by_symbol:
        raise ValueError(f"missing bars for primary symbol {primary!r}")
    for sym in required:
        if sym not in bars_by_symbol:
            raise ValueError(f"missing bars for required symbol {sym!r}")

    sorted_by_symbol: dict[str, list[KlineRecord]] = {
        sym: sorted(bars_by_symbol[sym], key=lambda b: b.event_time) for sym in required
    }
    primary_bars = sorted_by_symbol[primary]

    out: list[MaterializedFeature] = []
    for i in range(len(primary_bars)):
        primary_now = primary_bars[i]
        histories: dict[str, Sequence[KlineRecord]] = {primary: primary_bars[: i + 1]}
        skip = False
        for sym in required:
            if sym == primary:
                continue
            aligned = _bars_up_to(sorted_by_symbol[sym], primary_now.event_time)
            if not aligned:
                skip = True
                break
            histories[sym] = aligned
        if skip:
            continue
        value = feature.compute(histories)
        if value is None:
            continue
        out.append(
            MaterializedFeature(
                feature_id=feature.spec.full_id,
                entity_id=entity_id,
                event_time=primary_now.event_time,
                availability_time=primary_now.event_time + feature.spec.availability_delay,
                value=value,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Contract helpers — every MultiSymbolFeature module MUST have a test that
# calls assert_multi_symbol_feature_contract. CI enforces this.
# ---------------------------------------------------------------------------


def assert_multi_symbol_feature_deterministic(
    feature: MultiSymbolFeature,
    histories: Mapping[str, Sequence[KlineRecord]],
) -> None:
    v1 = feature.compute(histories)
    v2 = feature.compute(histories)
    assert v1 == v2, f"{feature.spec.full_id}: non-deterministic — got {v1!r} then {v2!r}"


def assert_multi_symbol_feature_respects_lookback(
    feature: MultiSymbolFeature,
    histories: Mapping[str, Sequence[KlineRecord]],
) -> None:
    """Perturbing bars beyond the declared lookback tail in ANY symbol must
    not change the value. Catches cross-symbol leakage the same way the
    single-symbol contract catches within-symbol leakage.
    """
    lookback = feature.spec.lookback_bars
    for sym in feature.required_symbols:
        if len(histories[sym]) < lookback:
            raise ValueError(f"contract test needs >= lookback_bars ({lookback}) for {sym}")

    tail_only = {sym: list(histories[sym][-lookback:]) for sym in feature.required_symbols}
    v_tail = feature.compute(tail_only)

    # Perturb bars beyond the lookback tail in each symbol, one at a time.
    for target_sym in feature.required_symbols:
        if len(histories[target_sym]) <= lookback:
            continue
        perturbed = {s: list(h) for s, h in histories.items()}
        target = perturbed[target_sym]
        for i in range(len(target) - lookback):
            b = target[i]
            target[i] = replace(
                b,
                open=b.open * 1000.0 + 1.0,
                high=b.high * 1000.0 + 1.0,
                low=b.low * 1000.0 + 1.0,
                close=b.close * 1000.0 + 1.0,
                volume=b.volume * 1000.0 + 1.0,
                turnover=b.turnover * 1000.0 + 1.0,
            )
        v_perturbed = feature.compute(perturbed)
        assert v_tail == v_perturbed, (
            f"{feature.spec.full_id}: {target_sym} history perturbation beyond "
            f"lookback={lookback} changed value: tail={v_tail!r} perturbed={v_perturbed!r}"
        )


def assert_multi_symbol_feature_handles_insufficient_history(
    feature: MultiSymbolFeature,
    histories: Mapping[str, Sequence[KlineRecord]],
) -> None:
    """With fewer bars than lookback in ANY symbol, compute() must return None."""
    lookback = feature.spec.lookback_bars
    if lookback <= 1:
        return
    for short_sym in feature.required_symbols:
        too_few = {
            s: (list(h[: lookback - 1]) if s == short_sym else list(h))
            for s, h in histories.items()
        }
        v = feature.compute(too_few)
        assert v is None, (
            f"{feature.spec.full_id}: expected None when {short_sym} has "
            f"only {lookback - 1} bars; got {v!r}"
        )


def assert_multi_symbol_feature_contract(
    feature: MultiSymbolFeature,
    histories: Mapping[str, Sequence[KlineRecord]],
) -> None:
    """Run all three required checks. Every multi-symbol feature test MUST call this."""
    assert_multi_symbol_feature_deterministic(feature, histories)
    assert_multi_symbol_feature_respects_lookback(feature, histories)
    assert_multi_symbol_feature_handles_insufficient_history(feature, histories)
