"""Tests for isotonic multi-class calibrator."""

from __future__ import annotations

import numpy as np
import pytest

from trade.model.calibration import IsotonicCalibrator


def test_calibrator_transform_before_fit_raises() -> None:
    cal = IsotonicCalibrator()
    assert not cal.is_fitted
    with pytest.raises(RuntimeError):
        cal.transform(np.zeros((1, 3)))


def test_fit_transform_rows_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    raw = rng.dirichlet([1.0, 1.0, 1.0], size=500)
    # Ground truth follows the argmax of raw ~ correct model.
    y_true = raw.argmax(axis=1)

    cal = IsotonicCalibrator()
    cal.fit(raw, y_true)
    out = cal.transform(raw)
    assert out.shape == raw.shape
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_transform_single_preserves_key_order() -> None:
    rng = np.random.default_rng(0)
    raw = rng.dirichlet([1.0, 1.0, 1.0], size=200)
    y = raw.argmax(axis=1)
    cal = IsotonicCalibrator()
    cal.fit(raw, y)

    got = cal.transform_single({"down": 0.2, "flat": 0.5, "up": 0.3})
    assert list(got.keys()) == ["down", "flat", "up"]
    assert sum(got.values()) == pytest.approx(1.0, abs=1e-6)


def test_fit_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        IsotonicCalibrator().fit(np.array([0.1, 0.9]), np.array([0]))
