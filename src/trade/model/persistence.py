"""Model artifact persistence.

Writes a `TrainingArtifacts` bundle to a directory laid out as:

    {path}/
        manifest.json         # feature_ids, reproducibility_hash, config
        model.joblib          # pickled LightGBMClassifierV1
        calibrator.joblib     # pickled IsotonicCalibrator (if fit)

Loading reconstructs the same `TrainingArtifacts`. The reproducibility
hash is stored verbatim so a downstream verifier can compare against a
recomputed hash from committed artifacts (V1_SPEC HARD REQUIREMENT).

Format version is captured in `manifest.json` so a future breaking change
can be detected and rejected rather than silently misloaded.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib

from trade.model.calibration import IsotonicCalibrator
from trade.model.lightgbm_classifier import LightGBMClassifierV1
from trade.training.pipeline import TrainingArtifacts

_MANIFEST_NAME = "manifest.json"
_MODEL_NAME = "model.joblib"
_CALIBRATOR_NAME = "calibrator.joblib"
_MANIFEST_VERSION = 1


def save_training_artifacts(artifacts: TrainingArtifacts, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, path / _MODEL_NAME)
    if artifacts.calibrator is not None:
        joblib.dump(artifacts.calibrator, path / _CALIBRATOR_NAME)
    manifest: dict[str, Any] = {
        "manifest_version": _MANIFEST_VERSION,
        "feature_ids": list(artifacts.feature_ids),
        "reproducibility_hash": artifacts.reproducibility_hash,
        "model_config": artifacts.model_config,
        "train_rows": artifacts.train_rows,
        "calibration_rows": artifacts.calibration_rows,
        "calibrator_present": artifacts.calibrator is not None,
        "saved_at": datetime.now(UTC).isoformat(),
    }
    (path / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_training_artifacts(path: Path) -> TrainingArtifacts:
    manifest_path = path / _MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {_MANIFEST_NAME} in {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_version") != _MANIFEST_VERSION:
        raise ValueError(
            f"unsupported manifest_version {manifest.get('manifest_version')!r}; "
            f"expected {_MANIFEST_VERSION}"
        )
    model = cast(LightGBMClassifierV1, joblib.load(path / _MODEL_NAME))
    calibrator: IsotonicCalibrator | None = None
    if manifest.get("calibrator_present"):
        calibrator = cast(IsotonicCalibrator, joblib.load(path / _CALIBRATOR_NAME))
    return TrainingArtifacts(
        model=model,
        calibrator=calibrator,
        feature_ids=tuple(manifest["feature_ids"]),
        train_rows=int(manifest["train_rows"]),
        calibration_rows=int(manifest["calibration_rows"]),
        reproducibility_hash=str(manifest["reproducibility_hash"]),
        model_config=dict(manifest["model_config"]),
    )
