"""Save/load round-trip for TrainingArtifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from trade.data.schemas import KlineRecord
from trade.features.definitions.log_return import LogReturnN
from trade.features.definitions.realized_vol import RealizedVolN
from trade.features.store import InMemoryFeatureStore
from trade.labels.triple_barrier import triple_barrier_labels
from trade.model.persistence import (
    load_training_artifacts,
    save_training_artifacts,
)
from trade.training.pipeline import train_model


def _bars(n: int, seed: int = 42) -> list[KlineRecord]:
    rng = np.random.default_rng(seed)
    price = 100.0
    base = datetime(2024, 1, 1, tzinfo=UTC)
    out: list[KlineRecord] = []
    for i in range(n):
        price *= 1.0 + float(rng.normal(0.0005, 0.01))
        h = price * (1 + abs(float(rng.normal(0, 0.005))))
        lo = price * (1 - abs(float(rng.normal(0, 0.005))))
        out.append(
            KlineRecord(
                source="bybit",
                category="linear",
                symbol="BTCUSDT",
                interval="60",
                event_time=base + timedelta(hours=i),
                ingest_time=base + timedelta(hours=i, seconds=1),
                open=price,
                high=h,
                low=lo,
                close=price,
                volume=1.0,
                turnover=price,
            )
        )
    return out


def _train() -> tuple:
    bars = _bars(400)
    feats = [LogReturnN(window=5), RealizedVolN(window=20)]
    store = InMemoryFeatureStore()
    for feat in feats:
        store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)
    labels = triple_barrier_labels(bars, horizon_bars=6, up_pct=0.01, down_pct=0.01)
    artifacts = train_model(
        feature_store=store,
        feature_ids=[f.spec.full_id for f in feats],
        labels=labels,
        dataset_manifest_ids=["ds1"],
        feature_manifest_ids=[f.spec.full_id for f in feats],
        code_git_sha="deadbeef",
        python_lockfile_sha="cafef00d",
    )
    return artifacts, store, labels


def test_save_load_round_trip(tmp_path: Path) -> None:
    artifacts, store, labels = _train()
    out_dir = tmp_path / "model-v1"
    save_training_artifacts(artifacts, out_dir)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["manifest_version"] == 1
    assert manifest["feature_ids"] == list(artifacts.feature_ids)
    assert manifest["reproducibility_hash"] == artifacts.reproducibility_hash
    assert manifest["calibrator_present"] is True
    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "calibrator.joblib").exists()

    loaded = load_training_artifacts(out_dir)
    assert loaded.feature_ids == artifacts.feature_ids
    assert loaded.reproducibility_hash == artifacts.reproducibility_hash
    assert loaded.train_rows == artifacts.train_rows
    assert loaded.calibration_rows == artifacts.calibration_rows
    assert loaded.model_config == artifacts.model_config

    # Predictions must match bit-for-bit after a save/load round-trip.
    frame = store.point_in_time_join(labels=labels, feature_ids=artifacts.feature_ids)
    p_original = artifacts.model.predict_proba_matrix(frame)
    p_loaded = loaded.model.predict_proba_matrix(frame)
    assert np.array_equal(p_original, p_loaded)


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_training_artifacts(tmp_path / "does-not-exist")


def test_load_rejects_unknown_manifest_version(tmp_path: Path) -> None:
    out_dir = tmp_path / "model-v2-future"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(json.dumps({"manifest_version": 99}))
    with pytest.raises(ValueError, match="manifest_version"):
        load_training_artifacts(out_dir)


def test_save_without_calibrator_and_load_back(tmp_path: Path) -> None:
    # Small corpus, disable calibrator to test the None branch.
    bars = _bars(120, seed=99)
    feats = [LogReturnN(window=5)]
    store = InMemoryFeatureStore()
    for feat in feats:
        store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)
    labels = triple_barrier_labels(bars, horizon_bars=3, up_pct=0.01, down_pct=0.01)
    artifacts = train_model(
        feature_store=store,
        feature_ids=[f.spec.full_id for f in feats],
        labels=labels,
        dataset_manifest_ids=["ds1"],
        feature_manifest_ids=[f.spec.full_id for f in feats],
        code_git_sha="deadbeef",
        python_lockfile_sha="cafef00d",
        fit_calibrator=False,
    )
    assert artifacts.calibrator is None

    out_dir = tmp_path / "no-cal"
    save_training_artifacts(artifacts, out_dir)
    assert not (out_dir / "calibrator.joblib").exists()
    loaded = load_training_artifacts(out_dir)
    assert loaded.calibrator is None
