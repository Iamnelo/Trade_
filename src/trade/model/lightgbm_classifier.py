"""LightGBM 3-class classifier for {-1, 0, +1} triple-barrier labels.

Labels are mapped to LightGBM's expected integer range [0, num_class) as:

    -1  -> 0  (DOWN)
     0  -> 1  (FLAT)
    +1  -> 2  (UP)

`predict_proba` returns a dict keyed by "down"/"flat"/"up" to keep call
sites clear and to avoid strategy code hard-coding integer indices.

Missing feature values (`None` in TrainingFrame) are passed to LightGBM as
NaN; LGBM handles them natively via its default missing-value routing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier

from trade.features.types import TrainingFrame

_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "n_estimators": 100,
    "verbose": -1,
    "random_state": 42,
    "deterministic": True,
    "force_row_wise": True,
}

_CLASS_NAMES = ("down", "flat", "up")


def _label_to_int(label: float) -> int:
    # -1 -> 0, 0 -> 1, +1 -> 2. Anything else raises.
    if label == -1.0:
        return 0
    if label == 0.0:
        return 1
    if label == 1.0:
        return 2
    raise ValueError(f"unexpected label value {label!r}; expected -1, 0, or 1")


def training_frame_to_xy(
    frame: TrainingFrame,
    feature_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    n = frame.n_rows
    m = len(feature_ids)
    x = np.full((n, m), np.nan, dtype=np.float64)
    for j, fid in enumerate(feature_ids):
        column = frame.features.get(fid, ())
        for i, v in enumerate(column):
            if v is not None:
                x[i, j] = v
    y = np.array([_label_to_int(v) for v in frame.labels], dtype=np.int64)
    return x, y


class LightGBMClassifierV1:
    def __init__(self, *, params: Mapping[str, Any] | None = None) -> None:
        self._params: dict[str, Any] = dict(_DEFAULT_PARAMS)
        if params:
            self._params.update(params)
        self._model: LGBMClassifier | None = None
        self._feature_ids: tuple[str, ...] = ()

    @property
    def feature_ids(self) -> tuple[str, ...]:
        return self._feature_ids

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def fit(self, *, frame: TrainingFrame, feature_ids: Sequence[str]) -> None:
        x, y = training_frame_to_xy(frame, feature_ids)
        if x.shape[0] == 0:
            raise ValueError("training frame is empty; cannot fit")
        model = LGBMClassifier(**self._params)
        model.fit(x, y)
        self._model = model
        self._feature_ids = tuple(feature_ids)

    def predict_proba_matrix(self, frame: TrainingFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model has not been fit")
        x, _ = training_frame_to_xy(frame, self._feature_ids)
        # LGBM aligns columns to fitted feature order via feature_ids we own.
        probs = self._model.predict_proba(x)
        return np.asarray(probs, dtype=np.float64)

    def predict_proba_single(self, feature_vector: Mapping[str, float]) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("model has not been fit")
        x = np.array(
            [[feature_vector.get(fid, float("nan")) for fid in self._feature_ids]],
            dtype=np.float64,
        )
        row = np.asarray(self._model.predict_proba(x), dtype=np.float64)[0]
        return {name: float(row[i]) for i, name in enumerate(_CLASS_NAMES)}

    def pred_contrib_single(self, feature_vector: Mapping[str, float]) -> np.ndarray:
        """LightGBM SHAP contributions for a single feature vector.

        Returns a `(n_classes, n_features + 1)` array. Column j <= n_features-1
        is the contribution of `feature_ids[j]`; the last column is the base
        value (expected model output for the class). Sum of contributions
        approximates the class logit.
        """
        if self._model is None:
            raise RuntimeError("model has not been fit")
        x = np.array(
            [[feature_vector.get(fid, float("nan")) for fid in self._feature_ids]],
            dtype=np.float64,
        )
        raw = self._model.booster_.predict(x, pred_contrib=True)
        arr = np.asarray(raw, dtype=np.float64)
        # LGBM returns shape (1, n_classes * (n_features + 1)) for multiclass.
        n_feats = len(self._feature_ids)
        n_classes = len(_CLASS_NAMES)
        expected_cols = n_classes * (n_feats + 1)
        if arr.shape != (1, expected_cols):
            raise RuntimeError(
                f"unexpected pred_contrib shape {arr.shape}; expected (1, {expected_cols})"
            )
        return arr.reshape(n_classes, n_feats + 1)


# ---------------------------------------------------------------------------
# 2-class directional classifier
# ---------------------------------------------------------------------------

_BINARY_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "n_estimators": 100,
    "verbose": -1,
    "random_state": 42,
    "deterministic": True,
    "force_row_wise": True,
}

_BINARY_CLASS_NAMES = ("down", "up")


def _binary_label_to_int(label: float) -> int:
    if label == -1.0:
        return 0
    if label == 1.0:
        return 1
    raise ValueError(f"unexpected binary label value {label!r}; expected -1 or +1")


def binary_training_frame_to_xy(
    frame: TrainingFrame,
    feature_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    n = frame.n_rows
    m = len(feature_ids)
    x = np.full((n, m), np.nan, dtype=np.float64)
    for j, fid in enumerate(feature_ids):
        column = frame.features.get(fid, ())
        for i, v in enumerate(column):
            if v is not None:
                x[i, j] = v
    y = np.array([_binary_label_to_int(v) for v in frame.labels], dtype=np.int64)
    return x, y


class LightGBMBinaryClassifierV1:
    """Directional {-1, +1} classifier.

    Same feature-frame contract as `LightGBMClassifierV1` but with two
    classes only: down=0, up=1. `predict_proba_matrix` returns a
    (n_rows, 2) array with the same column order as `_BINARY_CLASS_NAMES`
    so downstream code can uniformly index probs[:, 0] = P(down),
    probs[:, 1] = P(up) whether the model is binary or ternary.
    """

    def __init__(self, *, params: Mapping[str, Any] | None = None) -> None:
        self._params: dict[str, Any] = dict(_BINARY_DEFAULT_PARAMS)
        if params:
            self._params.update(params)
        self._model: LGBMClassifier | None = None
        self._feature_ids: tuple[str, ...] = ()

    @property
    def feature_ids(self) -> tuple[str, ...]:
        return self._feature_ids

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def fit(self, *, frame: TrainingFrame, feature_ids: Sequence[str]) -> None:
        x, y = binary_training_frame_to_xy(frame, feature_ids)
        if x.shape[0] == 0:
            raise ValueError("training frame is empty; cannot fit")
        model = LGBMClassifier(**self._params)
        model.fit(x, y)
        self._model = model
        self._feature_ids = tuple(feature_ids)

    def predict_proba_matrix(self, frame: TrainingFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model has not been fit")
        x, _ = binary_training_frame_to_xy(frame, self._feature_ids)
        probs = self._model.predict_proba(x)
        return np.asarray(probs, dtype=np.float64)

    def predict_proba_single(self, feature_vector: Mapping[str, float]) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("model has not been fit")
        x = np.array(
            [[feature_vector.get(fid, float("nan")) for fid in self._feature_ids]],
            dtype=np.float64,
        )
        row = np.asarray(self._model.predict_proba(x), dtype=np.float64)[0]
        return {name: float(row[i]) for i, name in enumerate(_BINARY_CLASS_NAMES)}
