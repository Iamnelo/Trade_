"""Tests for LightGBMClassifierV1: fit / predict_proba / feature-id round-trip."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade.features.types import TrainingFrame
from trade.model.lightgbm_classifier import (
    LightGBMClassifierV1,
    _label_to_int,
    training_frame_to_xy,
)


def _synthetic_frame(n: int = 300, seed: int = 42) -> TrainingFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    # Deterministic labelling with clear signal.
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


def test_label_to_int_mapping() -> None:
    assert _label_to_int(-1.0) == 0
    assert _label_to_int(0.0) == 1
    assert _label_to_int(1.0) == 2
    with pytest.raises(ValueError):
        _label_to_int(2.0)


def test_training_frame_to_xy_shapes_and_nan_handling() -> None:
    frame = TrainingFrame(
        entity_ids=("BTCUSDT",) * 3,
        event_times=(
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 2, tzinfo=UTC),
        ),
        labels=(-1.0, 0.0, 1.0),
        features={"a@1": (1.0, None, 3.0), "b@1": (10.0, 20.0, None)},
    )
    x, y = training_frame_to_xy(frame, ["a@1", "b@1"])
    assert x.shape == (3, 2)
    assert np.isnan(x[1, 0])
    assert np.isnan(x[2, 1])
    assert y.tolist() == [0, 1, 2]


def test_lightgbm_fit_predict_and_deterministic() -> None:
    frame = _synthetic_frame()
    ids = ["x1@1", "x2@1"]
    m1 = LightGBMClassifierV1()
    m1.fit(frame=frame, feature_ids=ids)
    p1 = m1.predict_proba_matrix(frame)
    assert p1.shape == (frame.n_rows, 3)
    assert np.allclose(p1.sum(axis=1), 1.0, atol=1e-6)

    m2 = LightGBMClassifierV1()
    m2.fit(frame=frame, feature_ids=ids)
    p2 = m2.predict_proba_matrix(frame)
    assert np.array_equal(p1, p2)


def test_predict_proba_single_returns_named_classes() -> None:
    frame = _synthetic_frame()
    model = LightGBMClassifierV1()
    model.fit(frame=frame, feature_ids=["x1@1", "x2@1"])
    probs = model.predict_proba_single({"x1@1": 0.0, "x2@1": 0.0})
    assert set(probs.keys()) == {"down", "flat", "up"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_before_fit_raises() -> None:
    model = LightGBMClassifierV1()
    with pytest.raises(RuntimeError, match="not been fit"):
        model.predict_proba_single({"x1@1": 0.0})


def test_fit_on_empty_frame_raises() -> None:
    empty = TrainingFrame(entity_ids=(), event_times=(), labels=(), features={"a@1": ()})
    with pytest.raises(ValueError):
        LightGBMClassifierV1().fit(frame=empty, feature_ids=["a@1"])
