"""Per-fold classifier diagnostics + Oracle capture ratio.

Classifier diagnostics answer "how well is the model itself doing" —
independent of the trading strategy's threshold or sizing. We score the
trained model on the fold's HOLDOUT test slice (with triple-barrier
labels generated on that slice, horizon-trimmed so outcomes are
observable), then compute:

- macro AUC-ROC across the 3 classes (down / flat / up)
- macro average precision (AUC-PR) across the 3 classes
- expected calibration error (ECE) averaged across classes, 10 bins

Oracle capture ratio is the strategy's realised total return divided by
the perfect-foresight Oracle's return over the SAME test slice — 1.0
means we captured everything the ideal strategy could, 0.0 means we
captured nothing, negative means we lost on a series the Oracle would
have profited on.

Both diagnostics are pure research outputs. `score_holdout` never
touches training labels; the strategy replay layer never sees these
numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from trade.data.schemas import KlineRecord
from trade.features.protocol import Feature
from trade.features.store import InMemoryFeatureStore
from trade.features.types import LabelRow
from trade.labels.triple_barrier import triple_barrier_labels
from trade.model.calibration import IsotonicCalibrator
from trade.model.lightgbm_classifier import LightGBMClassifierV1
from trade.research.oracle import capture_ratio, oracle_max_pnl

_CLASS_LABELS_INT = (0, 1, 2)  # down, flat, up in LGBM label space


@dataclass(frozen=True, slots=True)
class ClassifierDiagnostics:
    n_test_samples: int
    auc_roc_macro: float | None
    auc_pr_macro: float | None
    ece_macro: float | None
    class_support: tuple[int, int, int]  # count per (down, flat, up) in holdout


@dataclass(frozen=True, slots=True)
class OracleCapture:
    strategy_final_equity: float
    initial_equity: float
    oracle_final_equity_long_only: float
    oracle_final_equity_long_short: float
    capture_ratio_vs_long_only: float
    capture_ratio_vs_long_short: float


def _label_to_int(v: float) -> int:
    if v == -1.0:
        return 0
    if v == 0.0:
        return 1
    if v == 1.0:
        return 2
    raise ValueError(f"unexpected label value {v!r}")


def score_holdout(
    *,
    sorted_bars: Sequence[KlineRecord],
    symbol: str,
    features: Sequence[Feature],
    model: LightGBMClassifierV1,
    calibrator: IsotonicCalibrator | None,
    test_start_idx: int,
    test_end_idx: int,
    label_horizon_bars: int,
    label_up_pct: float,
    label_down_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, true_labels_int) evaluated on the horizon-trimmed test slice.

    Returns empty arrays if the test slice yields no observable labels.
    """
    test_bars = sorted_bars[test_start_idx:test_end_idx]
    labels_raw = triple_barrier_labels(
        test_bars,
        horizon_bars=label_horizon_bars,
        up_pct=label_up_pct,
        down_pct=label_down_pct,
    )
    labels_raw = labels_raw[: max(0, len(labels_raw) - label_horizon_bars)]
    if not labels_raw:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int64)

    store = InMemoryFeatureStore()
    for feat in features:
        store.materialize(feature=feat, entity_id=symbol, bars=sorted_bars)

    label_rows = [
        LabelRow(entity_id=symbol, event_time=lb.event_time, label=lb.label) for lb in labels_raw
    ]
    frame = store.point_in_time_join(
        labels=label_rows, feature_ids=[f.spec.full_id for f in features]
    )
    if frame.n_rows == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int64)

    probs = model.predict_proba_matrix(frame)
    if calibrator is not None:
        probs = calibrator.transform(probs)
    y_true = np.array([_label_to_int(v) for v in frame.labels], dtype=np.int64)
    return probs, y_true


def _ece(probs: np.ndarray, y_true_int: np.ndarray, *, n_bins: int = 10) -> float:
    """Expected calibration error, averaged across the 3 classes.

    For each class, we bin the model's predicted probability of that
    class, compare bin-mean confidence to bin-mean empirical frequency,
    and weight by bin population. Classes with zero test-set support
    are skipped.
    """
    ece_per_class: list[float] = []
    n = y_true_int.shape[0]
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edge_pairs = list(pairwise(edges.tolist()))
    for cls_int in _CLASS_LABELS_INT:
        y_bin = (y_true_int == cls_int).astype(np.float64)
        if y_bin.sum() == 0.0 and (1 - y_bin).sum() == 0.0:
            continue
        p_cls = probs[:, cls_int]
        e = 0.0
        for lo, hi in edge_pairs:
            in_bin = (p_cls >= lo) & (p_cls < hi if hi < 1.0 else p_cls <= hi)
            n_in = int(in_bin.sum())
            if n_in == 0:
                continue
            bin_conf = float(p_cls[in_bin].mean())
            bin_acc = float(y_bin[in_bin].mean())
            e += (n_in / n) * abs(bin_conf - bin_acc)
        ece_per_class.append(e)
    if not ece_per_class:
        return float("nan")
    return float(np.mean(ece_per_class))


def compute_classifier_diagnostics(
    *,
    probs: np.ndarray,
    y_true_int: np.ndarray,
) -> ClassifierDiagnostics:
    """Aggregate multiclass AUC + calibration into a single dataclass.

    Returns None fields (rather than raising) when a metric is undefined
    for the given holdout — usually because a class had zero support in
    the test slice.
    """
    n = int(y_true_int.shape[0])
    if n == 0:
        return ClassifierDiagnostics(
            n_test_samples=0,
            auc_roc_macro=None,
            auc_pr_macro=None,
            ece_macro=None,
            class_support=(0, 0, 0),
        )
    support = tuple(int((y_true_int == c).sum()) for c in _CLASS_LABELS_INT)
    # AUC family require every class present.
    all_classes_present = all(s > 0 for s in support)

    auc_roc: float | None = None
    auc_pr: float | None = None
    if all_classes_present:
        try:
            auc_roc = float(roc_auc_score(y_true_int, probs, multi_class="ovr", average="macro"))
        except ValueError:
            auc_roc = None
        try:
            # sklearn's average_precision_score expects one-hot y_true for multiclass.
            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(n), y_true_int] = 1.0
            auc_pr = float(average_precision_score(y_onehot, probs, average="macro"))
        except ValueError:
            auc_pr = None

    ece_val = _ece(probs, y_true_int)
    ece_out: float | None = None if np.isnan(ece_val) else ece_val

    return ClassifierDiagnostics(
        n_test_samples=n,
        auc_roc_macro=auc_roc,
        auc_pr_macro=auc_pr,
        ece_macro=ece_out,
        class_support=(support[0], support[1], support[2]),
    )


def compute_oracle_capture(
    *,
    test_bars: Sequence[KlineRecord],
    strategy_final_equity: float,
    initial_equity: float,
) -> OracleCapture:
    oracle = oracle_max_pnl(test_bars, initial_equity=initial_equity)
    return OracleCapture(
        strategy_final_equity=strategy_final_equity,
        initial_equity=initial_equity,
        oracle_final_equity_long_only=oracle.final_equity_long_only,
        oracle_final_equity_long_short=oracle.final_equity_long_short,
        capture_ratio_vs_long_only=capture_ratio(
            strategy_final_equity=strategy_final_equity,
            oracle_final_equity=oracle.final_equity_long_only,
            initial_equity=initial_equity,
        ),
        capture_ratio_vs_long_short=capture_ratio(
            strategy_final_equity=strategy_final_equity,
            oracle_final_equity=oracle.final_equity_long_short,
            initial_equity=initial_equity,
        ),
    )
