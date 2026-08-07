"""BinaryModelDrivenStrategy: consumes a LightGBMBinaryClassifierV1.

The 2-class ("directional") training-set drops flat outcomes so the model
only distinguishes up from down. At inference the strategy uses a
symmetric confidence threshold θ ∈ (0.5, 1.0]:

    P(up) >  θ            → LONG
    P(up) <  1 - θ        → SHORT (if allow_short)
    else                  → FLAT

At θ = 0.5 this reduces to argmax; higher θ demands stronger conviction
before opening a position. Everything else (feature online-compute
contract, notional sizing, no-lookahead guarantee) mirrors
`ModelDrivenStrategy`.
"""

from __future__ import annotations

from collections.abc import Sequence

from trade.data.schemas import KlineRecord
from trade.features.protocol import Feature
from trade.model.calibration import IsotonicCalibrator
from trade.model.lightgbm_classifier import LightGBMBinaryClassifierV1
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition


class BinaryModelDrivenStrategy:
    def __init__(
        self,
        *,
        symbol: str,
        interval: str,
        model: LightGBMBinaryClassifierV1,
        features: Sequence[Feature],
        calibrator: IsotonicCalibrator | None = None,
        confidence_threshold: float = 0.55,
        notional_fraction: float = 0.5,
        allow_short: bool = True,
    ) -> None:
        if not 0.0 < notional_fraction <= 1.0:
            raise ValueError("notional_fraction must be in (0.0, 1.0]")
        if not 0.5 < confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be in (0.5, 1.0] for a binary strategy"
            )
        if not features:
            raise ValueError("at least one feature is required")
        expected_fids = tuple(f.spec.full_id for f in features)
        if model.feature_ids and tuple(model.feature_ids) != expected_fids:
            raise ValueError(
                "feature_ids mismatch between model and strategy features "
                f"(model: {model.feature_ids}, strategy: {expected_fids})"
            )
        self._symbol = symbol
        self._interval = interval
        self._model = model
        self._features = tuple(features)
        self._calibrator = calibrator
        self._confidence_threshold = confidence_threshold
        self._notional_fraction = notional_fraction
        self._allow_short = allow_short

    @property
    def name(self) -> str:
        return f"model_lgbm_binary({self._symbol})"

    def on_bar(
        self,
        bar: KlineRecord,
        source: MarketReplaySource,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        if bar.symbol != self._symbol:
            return []
        feature_vector: dict[str, float] = {}
        for feat in self._features:
            history = source.history(bar.symbol, self._interval, lookback=feat.spec.lookback_bars)
            value = feat.compute(history)
            if value is None:
                return []
            feature_vector[feat.spec.full_id] = value

        probs = self._model.predict_proba_single(feature_vector)
        if self._calibrator is not None:
            probs = self._calibrator.transform_single(probs)
        p_up = probs["up"]

        if p_up > self._confidence_threshold:
            qty = portfolio.equity * self._notional_fraction / bar.close
        elif p_up < 1.0 - self._confidence_threshold:
            if not self._allow_short:
                return [TargetPosition(symbol=self._symbol, target_qty=0.0)]
            qty = -portfolio.equity * self._notional_fraction / bar.close
        else:
            return [TargetPosition(symbol=self._symbol, target_qty=0.0)]
        return [TargetPosition(symbol=self._symbol, target_qty=qty)]
