"""Feature contract assertions.

Every concrete feature MUST have a test that runs `assert_feature_contract`
against a small handcrafted bar sequence. CI enforces the check discipline
by grepping for `assert_feature_contract` uses in tests/ (see the CI job
`feature-contracts`). See `tests/test_features_rsi.py` for the reference
usage.

The three properties any pure PIT feature MUST satisfy:

1. **Deterministic** — same bars in => same value out.
2. **Respects declared lookback** — extending the history with older bars
   beyond `lookback_bars` does not change the computed value. This catches
   the classic "I accidentally read the whole history" leak.
3. **Handles insufficient history** — with fewer bars than `lookback_bars`,
   `compute()` returns None rather than raising or returning garbage.

A Hypothesis-based property test is available for extra rigour when
appropriate. It's optional — the three assertions above are the required
floor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from trade.data.schemas import KlineRecord
from trade.features.protocol import Feature


def assert_feature_deterministic(feature: Feature, bars: Sequence[KlineRecord]) -> None:
    v1 = feature.compute(bars)
    v2 = feature.compute(bars)
    assert v1 == v2, f"{feature.spec.full_id}: non-deterministic — got {v1!r} then {v2!r}"


def assert_feature_respects_lookback(feature: Feature, bars: Sequence[KlineRecord]) -> None:
    """Prepending unrelated older bars must not change the value.

    If it does, the feature is silently peeking beyond its declared window.
    """
    lookback = feature.spec.lookback_bars
    if len(bars) < lookback:
        raise ValueError(f"contract test needs >= lookback_bars ({lookback}) bars; got {len(bars)}")
    tail = list(bars[-lookback:])
    v_tail_only = feature.compute(tail)

    # Prepend a wildly different bar and assert the value did not move.
    if len(bars) > lookback:
        v_with_history = feature.compute(bars)
        assert v_tail_only == v_with_history, (
            f"{feature.spec.full_id}: violates declared lookback "
            f"({lookback} bars). Tail-only value={v_tail_only!r}, "
            f"full-history value={v_with_history!r}"
        )

    # Also assert perturbing bars OUTSIDE the lookback tail does not change the value.
    if len(bars) > lookback:
        perturbed = list(bars)
        # Corrupt the first N-lookback bars beyond recognition.
        for i in range(len(bars) - lookback):
            b = perturbed[i]
            perturbed[i] = replace(
                b,
                open=b.open * 1000.0 + 1.0,
                high=b.high * 1000.0 + 1.0,
                low=b.low * 1000.0 + 1.0,
                close=b.close * 1000.0 + 1.0,
                volume=b.volume * 1000.0 + 1.0,
                turnover=b.turnover * 1000.0 + 1.0,
            )
        v_perturbed = feature.compute(perturbed)
        assert v_tail_only == v_perturbed, (
            f"{feature.spec.full_id}: value changed when perturbing bars "
            f"beyond declared lookback ({lookback}). "
            f"Tail-only={v_tail_only!r}, perturbed-past={v_perturbed!r}"
        )


def assert_feature_handles_insufficient_history(
    feature: Feature, bars: Sequence[KlineRecord]
) -> None:
    """With fewer bars than the declared lookback, compute() must return None."""
    lookback = feature.spec.lookback_bars
    if lookback <= 1:
        return  # cannot be under-supplied
    too_few = list(bars[: lookback - 1])
    v = feature.compute(too_few)
    assert v is None, (
        f"{feature.spec.full_id}: expected None with only "
        f"{len(too_few)} bars (< lookback={lookback}); got {v!r}"
    )


def assert_feature_contract(feature: Feature, bars: Sequence[KlineRecord]) -> None:
    """Run all three required checks. Every feature test MUST call this."""
    assert_feature_deterministic(feature, bars)
    assert_feature_respects_lookback(feature, bars)
    assert_feature_handles_insufficient_history(feature, bars)
