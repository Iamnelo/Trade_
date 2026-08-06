"""BTC-ETH cross-asset return spread.

At each BTCUSDT bar time t, compute:

    log(close_ETH[t] / close_ETH[t - window]) - log(close_BTC[t] / close_BTC[t - window])

Positive => ETH outperformed BTC over the window. Requires ETHUSDT bars
aligned to the same interval (the store's PIT-join guarantees only bars
with event_time <= t are visible).

This is a multi-symbol feature — see `trade.features.multi_symbol`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import timedelta

from trade.data.schemas import KlineRecord
from trade.features.types import FeatureSpec

_PRIMARY = "BTCUSDT"
_SECONDARY = "ETHUSDT"


class BTCETHReturnSpread:
    def __init__(self, *, window: int = 5, interval: str = "60") -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self.spec = FeatureSpec(
            name="btc_eth_return_spread",
            version=str(window),
            inputs=("close",),
            lookback_bars=window + 1,
            availability_delay=timedelta(0),
            entity="symbol",
            interval=interval,
        )

    @property
    def primary_symbol(self) -> str:
        return _PRIMARY

    @property
    def required_symbols(self) -> tuple[str, ...]:
        return (_PRIMARY, _SECONDARY)

    def compute(
        self,
        histories: Mapping[str, Sequence[KlineRecord]],
    ) -> float | None:
        need = self._window + 1
        btc = histories.get(_PRIMARY)
        eth = histories.get(_SECONDARY)
        if btc is None or eth is None:
            return None
        if len(btc) < need or len(eth) < need:
            return None
        # Only look at the declared lookback tail (contract requirement).
        btc_tail = btc[-need:]
        eth_tail = eth[-need:]
        btc_prev, btc_now = btc_tail[0].close, btc_tail[-1].close
        eth_prev, eth_now = eth_tail[0].close, eth_tail[-1].close
        if btc_prev <= 0 or btc_now <= 0 or eth_prev <= 0 or eth_now <= 0:
            return None
        btc_ret = math.log(btc_now / btc_prev)
        eth_ret = math.log(eth_now / eth_prev)
        return eth_ret - btc_ret
