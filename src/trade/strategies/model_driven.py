"""ModelDrivenStrategy: consumes a trained LightGBM classifier + calibrator.

On each bar it computes every feature ONLINE from the MarketReplaySource's
PIT-safe history (same `Feature.compute` code path used offline for the
feature store) and queries the model. Position sizing:

- Argmax class determines direction:
    "up"   -> +notional_fraction * equity / price   (long)
    "down" -> -notional_fraction * equity / price   (short)
    "flat" -> 0
- The strategy only takes a position when the calibrated probability of the
  argmax class exceeds `confidence_threshold`. Otherwise it goes flat.

Zero look-ahead by construction: source.history() returns only bars whose
close has occurred by sim_clock; features are pure functions of that
history; the model's prediction only depends on those features.
"""

from __future__ import annotations

from collections.abc import Sequence

from trade.data.schemas import KlineRecord
from trade.features.protocol import Feature
from trade.model.calibration import IsotonicCalibrator
from trade.model.lightgbm_classifier import LightGBMClassifierV1
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition


class ModelDrivenStrategy:
    def __init__(
        self,
        *,
        symbol: str,
        interval: str,
        model: LightGBMClassifierV1,
        features: Sequence[Feature],
        calibrator: IsotonicCalibrator | None = None,
        confidence_threshold: float = 0.55,
        notional_fraction: float = 0.5,
        allow_short: bool = True,
    ) -> None:
        if not 0.0 < notional_fraction <= 1.0:
            raise ValueError("notional_fraction must be in (0.0, 1.0]")
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0.0, 1.0]")
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
        return f"model_lgbm({self._symbol})"

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
                # Not enough history yet — do nothing (leave any prior target in place).
                return []
            feature_vector[feat.spec.full_id] = value

        probs = self._model.predict_proba_single(feature_vector)
        if self._calibrator is not None:
            probs = self._calibrator.transform_single(probs)

        best_class = max(probs, key=lambda k: probs[k])
        best_prob = probs[best_class]
        if best_prob < self._confidence_threshold:
            return [TargetPosition(symbol=self._symbol, target_qty=0.0)]

        if best_class == "up":
            qty = portfolio.equity * self._notional_fraction / bar.close
        elif best_class == "down":
            if not self._allow_short:
                return [TargetPosition(symbol=self._symbol, target_qty=0.0)]
            qty = -portfolio.equity * self._notional_fraction / bar.close
        else:
            qty = 0.0
        return [TargetPosition(symbol=self._symbol, target_qty=qty)]
