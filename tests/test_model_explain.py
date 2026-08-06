"""Tests for the LightGBM SHAP explanation wrapper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade.features.types import TrainingFrame
from trade.model.explain import LightGBMExplainer
from trade.model.lightgbm_classifier import LightGBMClassifierV1


def _synthetic_frame(n: int = 300, seed: int = 42) -> TrainingFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    logits = x1 - 0.5 * x2
    labels = np.where(logits > 0.5, 1.0, np.where(logits < -0.5, -1.0, 0.0))
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return TrainingFrame(
        entity_ids=tuple(["BTCUSDT"] * n),
        event_times=tuple(base + timedelta(hours=i) for i in range(n)),
        labels=tuple(float(v) for v in labels),
        features={
            "x1@1": tuple(float(v) for v in x1),
            "x2@1": tuple(float(v) for v in x2),
        },
    )


def _fit_model() -> LightGBMClassifierV1:
    frame = _synthetic_frame()
    model = LightGBMClassifierV1()
    model.fit(frame=frame, feature_ids=["x1@1", "x2@1"])
    return model


def test_contributions_shape_and_class_names() -> None:
    model = _fit_model()
    explainer = LightGBMExplainer(model)
    contribs = explainer.contributions({"x1@1": 0.5, "x2@1": -0.3})
    assert set(contribs.keys()) == {"down", "flat", "up"}
    for name, entry in contribs.items():
        assert entry.class_name == name
        assert set(entry.per_feature.keys()) == {"x1@1", "x2@1"}


def test_top_n_returns_sorted_by_abs_contribution() -> None:
    model = _fit_model()
    explainer = LightGBMExplainer(model)
    top = explainer.top_n_for_class({"x1@1": 1.5, "x2@1": -0.5}, class_name="up", n=2)
    assert len(top) == 2
    assert abs(top[0][1]) >= abs(top[1][1])


def test_top_n_rejects_unknown_class() -> None:
    model = _fit_model()
    explainer = LightGBMExplainer(model)
    with pytest.raises(ValueError, match="class_name"):
        explainer.top_n_for_class({"x1@1": 0.0, "x2@1": 0.0}, class_name="sideways")


def test_pred_contrib_unfit_raises() -> None:
    model = LightGBMClassifierV1()
    with pytest.raises(RuntimeError):
        model.pred_contrib_single({"x1@1": 0.0})


def test_contributions_deterministic_across_repeated_calls() -> None:
    model = _fit_model()
    explainer = LightGBMExplainer(model)
    fv = {"x1@1": 0.2, "x2@1": -0.1}
    a = explainer.contributions(fv)
    b = explainer.contributions(fv)
    for name in ("down", "flat", "up"):
        assert a[name].per_feature == b[name].per_feature
        assert a[name].base_value == b[name].base_value
