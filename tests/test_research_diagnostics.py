"""Tests for classifier diagnostics + Oracle capture ratio."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade.data.schemas import KlineRecord
from trade.research.diagnostics import (
    ClassifierDiagnostics,
    compute_classifier_diagnostics,
    compute_oracle_capture,
)


def _bars_from_closes(closes: list[float]) -> list[KlineRecord]:
    return [
        KlineRecord(
            source="synthetic",
            category="linear",
            symbol="TESTUSDT",
            interval="60",
            event_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            ingest_time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i, seconds=1),
            open=c,
            high=c,
            low=c,
            close=c,
            volume=1.0,
            turnover=c,
        )
        for i, c in enumerate(closes)
    ]


def test_diagnostics_empty_holdout() -> None:
    d = compute_classifier_diagnostics(
        probs=np.empty((0, 3)), y_true_int=np.empty((0,), dtype=np.int64)
    )
    assert d == ClassifierDiagnostics(
        n_test_samples=0,
        auc_roc_macro=None,
        auc_pr_macro=None,
        ece_macro=None,
        class_support=(0, 0, 0),
    )


def test_diagnostics_perfect_classifier_scores_one() -> None:
    # 3 rows, one per class, model puts full mass on the correct class.
    y = np.array([0, 1, 2], dtype=np.int64)
    probs = np.array(
        [
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
        ]
    )
    d = compute_classifier_diagnostics(probs=probs, y_true_int=y)
    assert d.n_test_samples == 3
    assert d.auc_roc_macro == pytest.approx(1.0)
    assert d.auc_pr_macro == pytest.approx(1.0)
    assert d.class_support == (1, 1, 1)


def test_diagnostics_ece_zero_when_probs_match_frequencies() -> None:
    # 10 samples, class 1 always. A perfectly calibrated model gives 1.0
    # probability to class 1 → confidence == accuracy in every bin.
    y = np.ones(10, dtype=np.int64)
    probs = np.zeros((10, 3))
    probs[:, 1] = 1.0
    d = compute_classifier_diagnostics(probs=probs, y_true_int=y)
    assert d.ece_macro is not None
    # Even with only one class present, ECE for that class should be near 0.
    assert d.ece_macro == pytest.approx(0.0, abs=1e-6)


def test_diagnostics_ece_nonzero_when_confidently_wrong() -> None:
    # Model puts 0.95 on class 0 for all rows, but reality is class 2.
    y = np.full(10, 2, dtype=np.int64)
    probs = np.zeros((10, 3))
    probs[:, 0] = 0.95
    probs[:, 1] = 0.03
    probs[:, 2] = 0.02
    d = compute_classifier_diagnostics(probs=probs, y_true_int=y)
    assert d.ece_macro is not None
    assert d.ece_macro > 0.5  # a lot of miscalibration


def test_diagnostics_auc_none_when_a_class_missing_from_holdout() -> None:
    # Only classes 0 and 1 present — AUC-ROC macro is undefined.
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.6, 0.3, 0.1],
            [0.3, 0.6, 0.1],
        ]
    )
    d = compute_classifier_diagnostics(probs=probs, y_true_int=y)
    assert d.auc_roc_macro is None
    assert d.auc_pr_macro is None
    assert d.class_support == (2, 2, 0)


def test_oracle_capture_full_capture() -> None:
    # Perfect strategy — final equity equals oracle's.
    closes = [100.0, 110.0, 121.0, 133.1]  # steady 10% up
    bars = _bars_from_closes(closes)
    cap = compute_oracle_capture(
        test_bars=bars,
        strategy_final_equity=1331.0,
        initial_equity=1000.0,
    )
    assert cap.capture_ratio_vs_long_only == pytest.approx(1.0)


def test_oracle_capture_negative_when_strategy_loses() -> None:
    # Oracle would profit, strategy lost money.
    closes = [100.0, 110.0, 121.0, 133.1]
    bars = _bars_from_closes(closes)
    cap = compute_oracle_capture(
        test_bars=bars,
        strategy_final_equity=900.0,  # lost 10%
        initial_equity=1000.0,
    )
    assert cap.capture_ratio_vs_long_only < 0.0


def test_oracle_capture_positive_partial_capture() -> None:
    closes = [100.0 * math.exp(0.001 * i) for i in range(50)]  # smooth 5%
    bars = _bars_from_closes(closes)
    cap = compute_oracle_capture(
        test_bars=bars,
        strategy_final_equity=1020.0,  # captured 2%
        initial_equity=1000.0,
    )
    # Oracle captures roughly all of the ~5% rise; strategy captured ~40% of it.
    assert 0.0 < cap.capture_ratio_vs_long_only < 1.0
