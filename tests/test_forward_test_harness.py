"""Forward-test harness plumbing: evaluate a frozen model on a held-out slice.

Exercises `scripts/forward_test.py::evaluate_forward` end-to-end without any
network or committed forward data: a small synthetic model is frozen to a
tmp dir, then replayed over the post-cutoff slice. Asserts the result shape,
the gate keys, and the no-retraining contract (loaded model reused as-is).
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from trade.data.schemas import KlineRecord
from trade.features.catalog import build_features
from trade.features.store import InMemoryFeatureStore
from trade.labels.triple_barrier import triple_barrier_labels
from trade.model.persistence import save_training_artifacts
from trade.training.pipeline import train_model

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "forward_test", _REPO_ROOT / "scripts" / "forward_test.py"
)
assert _SPEC is not None and _SPEC.loader is not None
forward_test = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(forward_test)


def _bars(n: int, seed: int = 3) -> list[KlineRecord]:
    rng = np.random.default_rng(seed)
    price = 100.0
    base = datetime(2024, 1, 1, tzinfo=UTC)
    out: list[KlineRecord] = []
    for i in range(n):
        price *= 1.0 + float(rng.normal(0.0005, 0.012))
        h = price * (1 + abs(float(rng.normal(0, 0.006))))
        lo = price * (1 - abs(float(rng.normal(0, 0.006))))
        out.append(
            KlineRecord(
                source="csv",
                category="linear",
                symbol="BTCUSDT",
                interval="D",
                event_time=base + timedelta(days=i),
                ingest_time=base + timedelta(days=i, seconds=1),
                open=price,
                high=h,
                low=lo,
                close=price,
                volume=1000.0,
                turnover=price * 1000.0,
            )
        )
    return out


def _freeze_synthetic(tmp_path: Path) -> tuple[dict[str, Any], list[KlineRecord], datetime]:
    bars = _bars(600)
    cutoff_idx = 400
    train_bars = bars[:cutoff_idx]
    cutoff = train_bars[-1].event_time

    feats = build_features(["log_return@5", "realized_vol@20"])
    store = InMemoryFeatureStore()
    for feat in feats:
        store.materialize(feature=feat, entity_id="BTCUSDT", bars=train_bars)
    labels = triple_barrier_labels(train_bars, horizon_bars=6, up_pct=0.01, down_pct=0.01)
    labels = labels[: max(0, len(labels) - 6)]
    artifacts = train_model(
        feature_store=store,
        feature_ids=[f.spec.full_id for f in feats],
        labels=labels,
        dataset_manifest_ids=["ds"],
        feature_manifest_ids=[f.spec.full_id for f in feats],
        code_git_sha="deadbeef",
        python_lockfile_sha="cafef00d",
    )
    out_dir = tmp_path / "winner_synth"
    save_training_artifacts(artifacts, out_dir)

    entry = {
        "symbol": "BTCUSDT",
        "spec_name": "winner_synth",
        "artifacts_dir": str(out_dir),  # absolute; REPO_ROOT / abs == abs
        "reproducibility_hash": artifacts.reproducibility_hash,
        "interval": "D",
        "label_mode": "3class",
        "label_horizon_bars": 6,
        "label_up_pct": 0.01,
        "label_down_pct": 0.01,
        "feature_ids": [f.spec.full_id for f in feats],
        "confidence_threshold": 0.52,
        "notional_fraction": 0.5,
        "allow_short": True,
        "bars_per_year": 365,
        "fee_bps": 5.5,
        "slippage_bps": 5.0,
        "initial_equity": 10_000.0,
        "cutoff_event_time": cutoff.isoformat(),
    }
    return entry, bars, cutoff


def test_evaluate_forward_shape_and_gates(tmp_path: Path) -> None:
    entry, bars, cutoff = _freeze_synthetic(tmp_path)
    result = forward_test.evaluate_forward(entry=entry, stitched=bars, cutoff=cutoff)

    assert result["symbol"] == "BTCUSDT"
    assert result["forward_window"]["n_bars"] == 200
    m = result["metrics"]
    for key in (
        "cost_adjusted_sharpe",
        "max_drawdown_pct",
        "n_fills",
        "forward_auc_roc",
        "forward_ece",
        "reference_auc_roc",
        "reference_ece",
    ):
        assert key in m
    assert isinstance(result["gate"]["passed"], bool)
    assert set(result["gate"]["thresholds"]) == {
        "min_cas",
        "max_dd_pct",
        "min_fills",
        "calibration_tolerance_x",
    }
    # A failing verdict must carry at least one attribution hint; a passing
    # one must carry none.
    if result["gate"]["passed"]:
        assert result["failure_attribution"] == []
    else:
        assert result["failure_attribution"]


def test_evaluate_forward_rejects_hash_mismatch(tmp_path: Path) -> None:
    entry, bars, cutoff = _freeze_synthetic(tmp_path)
    entry["reproducibility_hash"] = "0" * 64
    try:
        forward_test.evaluate_forward(entry=entry, stitched=bars, cutoff=cutoff)
    except ValueError as e:
        assert "hash" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on hash mismatch")


def test_no_bars_after_cutoff_raises(tmp_path: Path) -> None:
    entry, bars, _ = _freeze_synthetic(tmp_path)
    late = bars[-1].event_time  # cutoff at the very last bar -> no forward bars
    try:
        forward_test.evaluate_forward(entry=entry, stitched=bars, cutoff=late)
    except ValueError as e:
        assert "no bars after cutoff" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when no forward bars exist")
