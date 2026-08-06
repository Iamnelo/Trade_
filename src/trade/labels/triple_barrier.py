"""Triple-barrier labels (Lopez de Prado).

For each bar at time t, the label looks forward up to `horizon_bars` bars
and returns:

- +1 if the UPPER barrier is hit first (bar.high >= entry * (1 + up_pct))
- -1 if the LOWER barrier is hit first (bar.low  <= entry * (1 - down_pct))
-  0 on timeout (neither hit within the horizon)

If both barriers trip within the same bar the label is `0` (ambiguous) — the
conservative choice.

Leakage safeguard: the label at time t is a function of bars in (t, t+horizon],
so it is only KNOWN at t + horizon * interval. When walk-forward splitting for
training, set `purge_bars >= horizon` and `embargo_bars >= horizon` on the
splitter so training labels near a split boundary do not overlap the test
window. See `trade.wfo.splitter.purged_embargo_folds`.
"""

from __future__ import annotations

from collections.abc import Sequence

from trade.data.schemas import KlineRecord
from trade.features.types import LabelRow


def triple_barrier_labels(
    bars: Sequence[KlineRecord],
    *,
    horizon_bars: int,
    up_pct: float,
    down_pct: float,
) -> list[LabelRow]:
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    if up_pct <= 0 or down_pct <= 0:
        raise ValueError("up_pct and down_pct must be positive")

    sorted_bars = sorted(bars, key=lambda b: b.event_time)
    n = len(sorted_bars)
    labels: list[LabelRow] = []
    for i in range(n):
        current = sorted_bars[i]
        entry = current.close
        upper = entry * (1.0 + up_pct)
        lower = entry * (1.0 - down_pct)
        outcome = 0
        for j in range(1, horizon_bars + 1):
            k = i + j
            if k >= n:
                break
            fwd = sorted_bars[k]
            up_hit = fwd.high >= upper
            dn_hit = fwd.low <= lower
            if up_hit and dn_hit:
                outcome = 0
                break
            if up_hit:
                outcome = 1
                break
            if dn_hit:
                outcome = -1
                break
        labels.append(
            LabelRow(
                entity_id=current.symbol,
                event_time=current.event_time,
                label=float(outcome),
            )
        )
    return labels


def vol_scaled_barriers(*, atr: float, entry_price: float, k: float = 1.5) -> tuple[float, float]:
    """Convenience: symmetric barriers at `k * ATR / price` in relative terms.

    Returns (up_pct, down_pct). Suitable for a same-day horizon where ATR is
    a reasonable expected move.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if atr < 0:
        raise ValueError("atr must be non-negative")
    pct = k * atr / entry_price
    return pct, pct
