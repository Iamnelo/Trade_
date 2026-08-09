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
from trade.model.lightgbm_classifier import (
    LightGBMBinaryClassifierV1,
    LightGBMClassifierV1,
)
from trade.training.pipeline import BinaryTrainingArtifacts, TrainingArtifacts

_MANIFEST_NAME = "manifest.json"
_MODEL_NAME = "model.joblib"
_CALIBRATOR_NAME = "calibrator.joblib"
_MANIFEST_VERSION = 1

# artifact_kind discriminates the two disjoint classifier families so a
# loader can reconstruct the correct concrete type. Absent in v1 manifests
# written before the binary pipeline existed, which are always 3-class.
_KIND_3CLASS = "3class"
_KIND_BINARY = "2class_directional"


def save_training_artifacts(artifacts: TrainingArtifacts, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, path / _MODEL_NAME)
    if artifacts.calibrator is not None:
        joblib.dump(artifacts.calibrator, path / _CALIBRATOR_NAME)
    manifest: dict[str, Any] = {
        "manifest_version": _MANIFEST_VERSION,
        "artifact_kind": _KIND_3CLASS,
        "feature_ids": list(artifacts.feature_ids),
        "reproducibility_hash": artifacts.reproducibility_hash,
        "model_config": artifacts.model_config,
        "train_rows": artifacts.train_rows,
        "calibration_rows": artifacts.calibration_rows,
        "calibrator_present": artifacts.calibrator is not None,
        "saved_at": datetime.now(UTC).isoformat(),
    }
    (path / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def save_binary_training_artifacts(artifacts: BinaryTrainingArtifacts, path: Path) -> None:
    """Persist a 2-class-directional bundle. Same on-disk layout as the
    3-class variant, distinguished by `artifact_kind` in the manifest."""
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, path / _MODEL_NAME)
    if artifacts.calibrator is not None:
        joblib.dump(artifacts.calibrator, path / _CALIBRATOR_NAME)
    manifest: dict[str, Any] = {
        "manifest_version": _MANIFEST_VERSION,
        "artifact_kind": _KIND_BINARY,
        "feature_ids": list(artifacts.feature_ids),
        "reproducibility_hash": artifacts.reproducibility_hash,
        "model_config": artifacts.model_config,
        "train_rows": artifacts.train_rows,
        "calibration_rows": artifacts.calibration_rows,
        "calibrator_present": artifacts.calibrator is not None,
        "saved_at": datetime.now(UTC).isoformat(),
    }
    (path / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / _MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {_MANIFEST_NAME} in {path}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    if manifest.get("manifest_version") != _MANIFEST_VERSION:
        raise ValueError(
            f"unsupported manifest_version {manifest.get('manifest_version')!r}; "
            f"expected {_MANIFEST_VERSION}"
        )
    return manifest


def load_training_artifacts(path: Path) -> TrainingArtifacts:
    manifest = _read_manifest(path)
    kind = manifest.get("artifact_kind", _KIND_3CLASS)
    if kind != _KIND_3CLASS:
        raise ValueError(
            f"artifact at {path} is {kind!r}, not 3-class; "
            f"use load_binary_training_artifacts or load_any_training_artifacts"
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


def load_binary_training_artifacts(path: Path) -> BinaryTrainingArtifacts:
    manifest = _read_manifest(path)
    kind = manifest.get("artifact_kind")
    if kind != _KIND_BINARY:
        raise ValueError(
            f"artifact at {path} is {kind!r}, not {_KIND_BINARY!r}; "
            f"use load_training_artifacts or load_any_training_artifacts"
        )
    model = cast(LightGBMBinaryClassifierV1, joblib.load(path / _MODEL_NAME))
    calibrator: IsotonicCalibrator | None = None
    if manifest.get("calibrator_present"):
        calibrator = cast(IsotonicCalibrator, joblib.load(path / _CALIBRATOR_NAME))
    return BinaryTrainingArtifacts(
        model=model,
        calibrator=calibrator,
        feature_ids=tuple(manifest["feature_ids"]),
        train_rows=int(manifest["train_rows"]),
        calibration_rows=int(manifest["calibration_rows"]),
        reproducibility_hash=str(manifest["reproducibility_hash"]),
        model_config=dict(manifest["model_config"]),
    )


def load_any_training_artifacts(path: Path) -> TrainingArtifacts | BinaryTrainingArtifacts:
    """Dispatch on `artifact_kind` and reconstruct the right bundle.

    Use when the caller does not know the label mode a priori (e.g. a
    forward-test harness that loads whatever winner was frozen). v1
    manifests without an `artifact_kind` field are treated as 3-class.
    """
    manifest = _read_manifest(path)
    kind = manifest.get("artifact_kind", _KIND_3CLASS)
    if kind == _KIND_BINARY:
        return load_binary_training_artifacts(path)
    return load_training_artifacts(path)
