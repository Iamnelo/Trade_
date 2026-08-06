"""Feature catalog: rebuild concrete Feature instances from `name@version` ids.

A trained model persists a list of `feature_ids` (e.g., "log_return@5",
"macd_hist@12_26_9"). At inference time the strategy needs the same
Feature instances back, which means we need a name-and-version -> class
registry.

Multi-symbol features (`btc_eth_return_spread@N`) are intentionally NOT
in this catalog — the current ModelDrivenStrategy passes single-symbol
histories into each feature's `compute()`, so it cannot consume a
multi-symbol feature. Cross-asset inputs to the model land in a later
phase alongside a `MultiSymbolModelDrivenStrategy`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from trade.features.definitions.atr14 import ATR14
from trade.features.definitions.log_return import LogReturnN
from trade.features.definitions.macd_hist import MACDHistogram
from trade.features.definitions.realized_vol import RealizedVolN
from trade.features.definitions.rsi import RSI14
from trade.features.protocol import Feature


def _build_log_return(version: str) -> Feature:
    return LogReturnN(window=int(version))


def _build_realized_vol(version: str) -> Feature:
    return RealizedVolN(window=int(version))


def _build_atr(version: str) -> Feature:
    return ATR14(period=int(version))


def _build_macd_hist(version: str) -> Feature:
    parts = version.split("_")
    if len(parts) != 3:
        raise ValueError(f"macd_hist version must be 'fast_slow_signal'; got {version!r}")
    fast, slow, signal = (int(p) for p in parts)
    return MACDHistogram(fast=fast, slow=slow, signal=signal)


def _build_rsi_close(version: str) -> Feature:
    if version != "14":
        raise ValueError(f"only rsi_close@14 is registered; got version={version!r}")
    return RSI14()


_BUILDERS: dict[str, Callable[[str], Feature]] = {
    "log_return": _build_log_return,
    "realized_vol": _build_realized_vol,
    "atr": _build_atr,
    "macd_hist": _build_macd_hist,
    "rsi_close": _build_rsi_close,
}


def registered_feature_names() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def build_feature(feature_id: str) -> Feature:
    if "@" not in feature_id:
        raise ValueError(f"feature_id must be 'name@version'; got {feature_id!r}")
    name, version = feature_id.split("@", 1)
    builder = _BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"unknown feature {name!r}; registered: {sorted(_BUILDERS)}")
    feature = builder(version)
    if feature.spec.full_id != feature_id:
        raise ValueError(
            f"catalog builder for {name!r}@{version!r} produced spec.full_id="
            f"{feature.spec.full_id!r}; expected {feature_id!r}"
        )
    return feature


def build_features(feature_ids: Sequence[str]) -> list[Feature]:
    return [build_feature(fid) for fid in feature_ids]
