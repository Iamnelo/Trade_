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
from trade.features.definitions.day_of_week import DayOfWeekCos, DayOfWeekSin
from trade.features.definitions.log_return import LogReturnN
from trade.features.definitions.macd_hist import MACDHistogram
from trade.features.definitions.realized_vol import RealizedVolN
from trade.features.definitions.return_higher_moments import (
    ReturnKurtosisN,
    ReturnSkewN,
)
from trade.features.definitions.rsi import RSI14
from trade.features.definitions.time_of_day import HourOfDayCos, HourOfDaySin
from trade.features.definitions.vol_regime import VolRegime
from trade.features.definitions.volume_zscore import TurnoverZScoreN, VolumeZScoreN
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


def _build_time_of_day(version: str) -> Feature:
    if version == "sin":
        return HourOfDaySin()
    if version == "cos":
        return HourOfDayCos()
    raise ValueError(f"time_of_day version must be 'sin' or 'cos'; got {version!r}")


def _build_day_of_week(version: str) -> Feature:
    if version == "sin":
        return DayOfWeekSin()
    if version == "cos":
        return DayOfWeekCos()
    raise ValueError(f"day_of_week version must be 'sin' or 'cos'; got {version!r}")


def _build_return_skew(version: str) -> Feature:
    return ReturnSkewN(window=int(version))


def _build_return_kurtosis(version: str) -> Feature:
    return ReturnKurtosisN(window=int(version))


def _build_volume_zscore(version: str) -> Feature:
    return VolumeZScoreN(window=int(version))


def _build_turnover_zscore(version: str) -> Feature:
    return TurnoverZScoreN(window=int(version))


def _build_vol_regime(version: str) -> Feature:
    parts = version.split("_")
    if len(parts) != 2:
        raise ValueError(f"vol_regime version must be 'short_long'; got {version!r}")
    short_window, long_window = (int(p) for p in parts)
    return VolRegime(short_window=short_window, long_window=long_window)


_BUILDERS: dict[str, Callable[[str], Feature]] = {
    "log_return": _build_log_return,
    "realized_vol": _build_realized_vol,
    "atr": _build_atr,
    "macd_hist": _build_macd_hist,
    "rsi_close": _build_rsi_close,
    "time_of_day": _build_time_of_day,
    "day_of_week": _build_day_of_week,
    "return_skew": _build_return_skew,
    "return_kurtosis": _build_return_kurtosis,
    "volume_zscore": _build_volume_zscore,
    "turnover_zscore": _build_turnover_zscore,
    "vol_regime": _build_vol_regime,
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
