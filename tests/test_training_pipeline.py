"""End-to-end tests for the training pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade.data.schemas import KlineRecord
from trade.features.definitions.log_return import LogReturnN
from trade.features.definitions.realized_vol import RealizedVolN
from trade.features.store import InMemoryFeatureStore
from trade.features.types import LabelRow
from trade.labels.triple_barrier import triple_barrier_labels
from trade.training.pipeline import train_model


def _bars(n: int, seed: int = 42) -> list[KlineRecord]:
    rng = np.random.default_rng(seed)
    price = 100.0
    out: list[KlineRecord] = []
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(n):
        # Random walk with mild upward drift and heteroskedastic vol.
        price *= 1.0 + float(rng.normal(0.0005, 0.01))
        h = price * (1 + abs(float(rng.normal(0, 0.005))))
        low = price * (1 - abs(float(rng.normal(0, 0.005))))
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
                low=low,
                close=price,
                volume=1.0,
                turnover=price,
            )
        )
    return out


def _fit_features(store: InMemoryFeatureStore, feats: list, bars: list[KlineRecord]) -> None:
    for feat in feats:
        store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)


def test_train_model_produces_artifacts_and_reproducibility_hash() -> None:
    bars = _bars(400)
    feats = [LogReturnN(window=5), RealizedVolN(window=20)]

    store = InMemoryFeatureStore()
    _fit_features(store, feats, bars)

    labels: list[LabelRow] = triple_barrier_labels(bars, horizon_bars=6, up_pct=0.01, down_pct=0.01)

    artifacts = train_model(
        feature_store=store,
        feature_ids=[f.spec.full_id for f in feats],
        labels=labels,
        dataset_manifest_ids=["bybit_kline_BTCUSDT_60"],
        feature_manifest_ids=[f.spec.full_id for f in feats],
        code_git_sha="deadbeef",
        python_lockfile_sha="cafef00d",
    )
    assert artifacts.model is not None
    assert artifacts.calibrator is not None
    assert artifacts.feature_ids == tuple(f.spec.full_id for f in feats)
    assert artifacts.train_rows > 0
    assert artifacts.calibration_rows > 0
    assert len(artifacts.reproducibility_hash) == 64  # sha256 hex


def test_train_model_is_reproducible() -> None:
    """Same inputs => byte-identical model output and identical repro hash."""
    bars = _bars(400)
    feats = [LogReturnN(window=5), RealizedVolN(window=20)]
    labels = triple_barrier_labels(bars, horizon_bars=6, up_pct=0.01, down_pct=0.01)

    def _train():
        store = InMemoryFeatureStore()
        _fit_features(store, feats, bars)
        return train_model(
            feature_store=store,
            feature_ids=[f.spec.full_id for f in feats],
            labels=labels,
            dataset_manifest_ids=["bybit_kline_BTCUSDT_60"],
            feature_manifest_ids=[f.spec.full_id for f in feats],
            code_git_sha="deadbeef",
            python_lockfile_sha="cafef00d",
        )

    a = _train()
    b = _train()
    assert a.reproducibility_hash == b.reproducibility_hash

    # Predicted probabilities must match bit-for-bit.
    store = InMemoryFeatureStore()
    _fit_features(store, feats, bars)
    frame = store.point_in_time_join(labels=labels, feature_ids=[f.spec.full_id for f in feats])
    pa = a.model.predict_proba_matrix(frame)
    pb = b.model.predict_proba_matrix(frame)
    assert np.array_equal(pa, pb)


def test_train_model_rejects_bad_calibration_fraction() -> None:
    with pytest.raises(ValueError):
        train_model(
            feature_store=InMemoryFeatureStore(),
            feature_ids=[],
            labels=[],
            dataset_manifest_ids=[],
            feature_manifest_ids=[],
            code_git_sha="deadbeef",
            python_lockfile_sha="cafef00d",
            calibration_fraction=1.5,
        )


def test_train_model_uses_only_pit_join_api() -> None:
    """Regression guard: the store still has no bypass methods."""
    store = InMemoryFeatureStore()
    banned = {"get_latest_features", "load_latest", "snapshot_now", "features_now"}
    assert banned.isdisjoint(set(dir(store)))
