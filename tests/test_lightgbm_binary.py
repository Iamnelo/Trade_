"""Tests for LightGBMBinaryClassifierV1."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade.features.types import TrainingFrame
from trade.model.lightgbm_classifier import LightGBMBinaryClassifierV1


def _frame_with_signal(n: int = 200) -> TrainingFrame:
    """x[0] > 0 predicts up=1; x[0] <= 0 predicts down=0."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, 2))
    y = np.where(x[:, 0] > 0, 1, -1).astype(np.float64)
    times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)]
    return TrainingFrame(
        entity_ids=tuple(["BTCUSDT"] * n),
        event_times=tuple(times),
        labels=tuple(y),
        features={
            "f1": tuple(x[:, 0].tolist()),
            "f2": tuple(x[:, 1].tolist()),
        },
    )


def test_binary_classifier_fits_and_predicts_shape() -> None:
    clf = LightGBMBinaryClassifierV1(params={"n_estimators": 20, "num_leaves": 3})
    frame = _frame_with_signal()
    clf.fit(frame=frame, feature_ids=["f1", "f2"])
    probs = clf.predict_proba_matrix(frame)
    assert probs.shape == (frame.n_rows, 2)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_binary_classifier_learns_signal() -> None:
    clf = LightGBMBinaryClassifierV1(params={"n_estimators": 100, "num_leaves": 5})
    frame = _frame_with_signal(400)
    clf.fit(frame=frame, feature_ids=["f1", "f2"])
    probs = clf.predict_proba_matrix(frame)
    # Rows where f1 > 0 should predict P(up) > 0.5.
    f1_positive = np.array(frame.features["f1"]) > 0
    p_up = probs[:, 1]
    assert (p_up[f1_positive] > 0.5).mean() > 0.9


def test_binary_classifier_rejects_unexpected_label_values() -> None:
    clf = LightGBMBinaryClassifierV1(params={"n_estimators": 5})
    bad_frame = TrainingFrame(
        entity_ids=("BTCUSDT",),
        event_times=(datetime(2024, 1, 1, tzinfo=UTC),),
        labels=(0.0,),  # flat is not allowed for binary
        features={"f1": (1.0,)},
    )
    with pytest.raises(ValueError, match="unexpected binary label"):
        clf.fit(frame=bad_frame, feature_ids=["f1"])


def test_predict_proba_single_returns_down_and_up_keys() -> None:
    clf = LightGBMBinaryClassifierV1(params={"n_estimators": 10})
    clf.fit(frame=_frame_with_signal(50), feature_ids=["f1", "f2"])
    got = clf.predict_proba_single({"f1": 1.0, "f2": 0.0})
    assert set(got.keys()) == {"down", "up"}
    assert 0.0 <= got["down"] <= 1.0
    assert 0.0 <= got["up"] <= 1.0
    assert got["down"] + got["up"] == pytest.approx(1.0)
