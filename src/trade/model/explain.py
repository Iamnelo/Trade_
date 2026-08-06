"""SHAP-style explanations via LightGBM's built-in ``pred_contrib``.

Wraps `LightGBMClassifierV1.pred_contrib_single` in a
per-class-per-feature dict and a ``top_n_for_class`` convenience for the
signal payload's "why now" explanation. Deliberately avoids the `shap`
Python package (which pulls in numba + llvmlite ~250 MB) since LGBM
already computes the exact tree SHAP values natively.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from trade.model.lightgbm_classifier import LightGBMClassifierV1

_CLASS_NAMES = ("down", "flat", "up")


@dataclass(frozen=True, slots=True)
class ClassContribution:
    class_name: str
    base_value: float
    per_feature: dict[str, float]

    def sum_contribution(self) -> float:
        return sum(self.per_feature.values())

    def top_n(self, n: int) -> list[tuple[str, float]]:
        return sorted(self.per_feature.items(), key=lambda kv: -abs(kv[1]))[:n]


class LightGBMExplainer:
    def __init__(self, model: LightGBMClassifierV1) -> None:
        self._model = model

    def contributions(self, feature_vector: Mapping[str, float]) -> dict[str, ClassContribution]:
        raw = self._model.pred_contrib_single(feature_vector)  # (n_classes, n_features+1)
        fids = self._model.feature_ids
        out: dict[str, ClassContribution] = {}
        for c, name in enumerate(_CLASS_NAMES):
            per_feature = {fids[j]: float(raw[c, j]) for j in range(len(fids))}
            out[name] = ClassContribution(
                class_name=name,
                base_value=float(raw[c, -1]),
                per_feature=per_feature,
            )
        return out

    def top_n_for_class(
        self,
        feature_vector: Mapping[str, float],
        *,
        class_name: str,
        n: int = 3,
    ) -> list[tuple[str, float]]:
        if class_name not in _CLASS_NAMES:
            raise ValueError(f"class_name must be one of {_CLASS_NAMES}")
        return self.contributions(feature_vector)[class_name].top_n(n)
