"""Training pipeline: features -> PIT-join -> LGBM -> calibration -> hash.

Every model release goes through this pipeline. It emits a
`ReproducibilityHash` computed over the exact bytes that determined the
model, and (optionally) writes an ExperimentRecord to MLflow via
`trade.tracking.mlflow.log_experiment_record`.

The pipeline consumes features ONLY via `feature_store.point_in_time_join`
(the HARD REQUIREMENT). It does not touch the feature source data directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from trade.features.store import FeatureStore
from trade.features.types import LabelRow, TrainingFrame
from trade.model.calibration import IsotonicCalibrator
from trade.model.lightgbm_classifier import LightGBMClassifierV1, training_frame_to_xy
from trade.reproducibility.hash import compute_reproducibility_hash


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    model: LightGBMClassifierV1
    calibrator: IsotonicCalibrator | None
    feature_ids: tuple[str, ...]
    train_rows: int
    calibration_rows: int
    reproducibility_hash: str
    model_config: dict[str, Any]


def _slice_frame(frame: TrainingFrame, start: int, end: int) -> TrainingFrame:
    if start < 0 or end > frame.n_rows or start >= end:
        raise ValueError(f"invalid slice [{start}, {end}) for frame of {frame.n_rows} rows")
    return TrainingFrame(
        entity_ids=frame.entity_ids[start:end],
        event_times=frame.event_times[start:end],
        labels=frame.labels[start:end],
        features={fid: col[start:end] for fid, col in frame.features.items()},
    )


def train_model(
    *,
    feature_store: FeatureStore,
    feature_ids: Sequence[str],
    labels: Sequence[LabelRow],
    dataset_manifest_ids: Sequence[str],
    feature_manifest_ids: Sequence[str],
    code_git_sha: str,
    python_lockfile_sha: str,
    model_config: Mapping[str, Any] | None = None,
    calibration_fraction: float = 0.2,
    fit_calibrator: bool = True,
) -> TrainingArtifacts:
    """Assemble a training frame from labels via PIT-join, then fit LGBM + isotonic.

    The calibration slice is carved from the TAIL of the training frame so
    it is temporally later than the training portion (avoids look-ahead in
    the calibrator itself).
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0.0, 1.0)")

    fids = tuple(feature_ids)
    frame = feature_store.point_in_time_join(labels=labels, feature_ids=fids)
    if frame.n_rows == 0:
        raise ValueError("PIT-join produced an empty training frame")

    n_cal = max(1, round(frame.n_rows * calibration_fraction))
    n_train = frame.n_rows - n_cal
    if n_train < 1:
        raise ValueError(f"frame too small: {frame.n_rows} rows total, {n_cal} for calibration")

    train_slice = _slice_frame(frame, 0, n_train)
    cal_slice = _slice_frame(frame, n_train, frame.n_rows)

    model = LightGBMClassifierV1(params=model_config)
    model.fit(frame=train_slice, feature_ids=fids)

    calibrator: IsotonicCalibrator | None = None
    if fit_calibrator:
        cal_x, cal_y = training_frame_to_xy(cal_slice, fids)
        if cal_x.shape[0] >= 2 and len(np.unique(cal_y)) >= 2:
            raw = model.predict_proba_matrix(cal_slice)
            calibrator = IsotonicCalibrator()
            calibrator.fit(raw, cal_y)

    hash_config = {"model": model.params, "calibration_fraction": calibration_fraction}
    repro_hash = compute_reproducibility_hash(
        dataset_manifest_ids=dataset_manifest_ids,
        feature_manifest_ids=feature_manifest_ids,
        model_config=hash_config,
        code_git_sha=code_git_sha,
        python_lockfile_sha=python_lockfile_sha,
    )
    return TrainingArtifacts(
        model=model,
        calibrator=calibrator,
        feature_ids=fids,
        train_rows=n_train,
        calibration_rows=n_cal,
        reproducibility_hash=repro_hash,
        model_config=hash_config,
    )
