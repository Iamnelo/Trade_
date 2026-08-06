"""Isotonic calibration for multi-class probability outputs.

Fits one `IsotonicRegression` per class mapping the model's raw probability
for that class to the observed frequency in a held-out sample. After
per-class calibration, rows are renormalised to sum to 1 so downstream code
can treat the output as a proper probability distribution.

Fit on a HELD-OUT set (not the training set) or the calibration itself
overfits. The training pipeline in `trade.training.pipeline` handles this
by carving out a calibration slice from the training frame.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._per_class: dict[int, IsotonicRegression] = {}
        self._n_classes: int = 0

    @property
    def is_fitted(self) -> bool:
        return bool(self._per_class)

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        if raw_probs.ndim != 2:
            raise ValueError("raw_probs must be 2-D (n_rows, n_classes)")
        n_classes = raw_probs.shape[1]
        self._per_class = {}
        for c in range(n_classes):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw_probs[:, c], (y_true == c).astype(np.float64))
            self._per_class[c] = iso
        self._n_classes = n_classes

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._per_class:
            raise RuntimeError("calibrator has not been fit")
        out = np.zeros_like(raw_probs, dtype=np.float64)
        for c, iso in self._per_class.items():
            out[:, c] = iso.transform(raw_probs[:, c])
        # Renormalise so each row sums to 1. If a row sums to 0 (pathological),
        # fall back to uniform.
        row_sums = out.sum(axis=1, keepdims=True)
        uniform = np.full_like(out, 1.0 / self._n_classes)
        return np.where(row_sums > 0, out / np.where(row_sums == 0, 1.0, row_sums), uniform)

    def transform_single(self, probs: dict[str, float]) -> dict[str, float]:
        if not self._per_class:
            raise RuntimeError("calibrator has not been fit")
        # Preserve the caller's key order.
        keys = list(probs.keys())
        row = np.array([[probs[k] for k in keys]], dtype=np.float64)
        calibrated = self.transform(row)[0]
        return {k: float(calibrated[i]) for i, k in enumerate(keys)}
