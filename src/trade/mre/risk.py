"""V1 Risk Manager: layered daily / weekly / monthly drawdown gates.

When a period's drawdown from that period's high-water mark exceeds its
threshold, the manager is HALTED for the rest of the period. Halted means:
only flatten targets (target_qty == 0) are allowed through; new-position
and scale-up targets are dropped.

The V1_SPEC kill switches (data staleness, exchange error rate, position
reconciliation drift, model drift) are enforced by the live runtime; in a
backtest they are unreachable by construction (no live systems), so the MRE
implements only the drawdown gates.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from trade.mre.types import TargetPosition


class RiskManager:
    def __init__(
        self,
        *,
        initial_equity: float,
        daily_pct: float = 0.035,
        weekly_pct: float = 0.08,
        monthly_pct: float = 0.12,
    ) -> None:
        if not 0 < daily_pct <= weekly_pct <= monthly_pct < 1:
            raise ValueError(
                "DD limits must satisfy 0 < daily <= weekly <= monthly < 1 "
                f"(got {daily_pct=}, {weekly_pct=}, {monthly_pct=})"
            )
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        self._daily_pct = daily_pct
        self._weekly_pct = weekly_pct
        self._monthly_pct = monthly_pct
        self._daily_hwm = initial_equity
        self._weekly_hwm = initial_equity
        self._monthly_hwm = initial_equity
        self._daily_period_key: tuple[int, int, int] | None = None
        self._weekly_period_key: tuple[int, int] | None = None
        self._monthly_period_key: tuple[int, int] | None = None
        self._halted_reasons: set[str] = set()
        self._reasons_ever: set[str] = set()

    @property
    def is_halted(self) -> bool:
        return bool(self._halted_reasons)

    @property
    def halted_reasons(self) -> tuple[str, ...]:
        return tuple(sorted(self._halted_reasons))

    @property
    def reasons_ever_triggered(self) -> tuple[str, ...]:
        return tuple(sorted(self._reasons_ever))

    def update(self, timestamp: datetime, equity: float) -> None:
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be UTC-aware")
        daily_key = (timestamp.year, timestamp.month, timestamp.day)
        iso_year, iso_week, _ = timestamp.isocalendar()
        weekly_key = (iso_year, iso_week)
        monthly_key = (timestamp.year, timestamp.month)

        if daily_key != self._daily_period_key:
            self._daily_period_key = daily_key
            self._daily_hwm = equity
            self._halted_reasons.discard("daily_dd")
        else:
            self._daily_hwm = max(self._daily_hwm, equity)

        if weekly_key != self._weekly_period_key:
            self._weekly_period_key = weekly_key
            self._weekly_hwm = equity
            self._halted_reasons.discard("weekly_dd")
        else:
            self._weekly_hwm = max(self._weekly_hwm, equity)

        if monthly_key != self._monthly_period_key:
            self._monthly_period_key = monthly_key
            self._monthly_hwm = equity
            self._halted_reasons.discard("monthly_dd")
        else:
            self._monthly_hwm = max(self._monthly_hwm, equity)

        if self._daily_hwm > 0 and 1.0 - equity / self._daily_hwm > self._daily_pct:
            self._halted_reasons.add("daily_dd")
        if self._weekly_hwm > 0 and 1.0 - equity / self._weekly_hwm > self._weekly_pct:
            self._halted_reasons.add("weekly_dd")
        if self._monthly_hwm > 0 and 1.0 - equity / self._monthly_hwm > self._monthly_pct:
            self._halted_reasons.add("monthly_dd")

        self._reasons_ever.update(self._halted_reasons)

    def gate_targets(self, targets: Sequence[TargetPosition]) -> list[TargetPosition]:
        if not self._halted_reasons:
            return list(targets)
        # When halted, only allow reductions to flat.
        return [t for t in targets if t.target_qty == 0.0]
