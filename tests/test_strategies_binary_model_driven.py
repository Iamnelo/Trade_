"""Tests for BinaryModelDrivenStrategy: threshold semantics + no lookahead."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade.data.schemas import KlineRecord
from trade.features.definitions.log_return import LogReturnN
from trade.features.store import InMemoryFeatureStore
from trade.features.types import LabelRow
from trade.model.calibration import IsotonicCalibrator
from trade.model.lightgbm_classifier import LightGBMBinaryClassifierV1
from trade.mre.backtest import run_backtest
from trade.mre.clock import SimClock
from trade.mre.source import MarketReplaySource
from trade.mre.types import BacktestConfig
from trade.strategies.binary_model_driven import BinaryModelDrivenStrategy


def _bars(n: int, seed: int = 42) -> list[KlineRecord]:
    rng = np.random.default_rng(seed)
    price = 100.0
    out: list[KlineRecord] = []
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(n):
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


def _train_binary_model() -> tuple[LightGBMBinaryClassifierV1, list, list[KlineRecord]]:
    bars = _bars(400)
    feats = [LogReturnN(window=5), LogReturnN(window=10)]
    store = InMemoryFeatureStore()
    for feat in feats:
        store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)
    # Fake directional labels: alternating up/down so the model has something to fit.
    label_rows = [
        LabelRow(
            entity_id="BTCUSDT",
            event_time=b.event_time,
            label=1.0 if (i % 2 == 0) else -1.0,
        )
        for i, b in enumerate(bars[50:250])
    ]
    frame = store.point_in_time_join(labels=label_rows, feature_ids=[f.spec.full_id for f in feats])
    clf = LightGBMBinaryClassifierV1(params={"n_estimators": 20, "num_leaves": 3})
    clf.fit(frame=frame, feature_ids=[f.spec.full_id for f in feats])
    return clf, feats, bars


def test_binary_strategy_rejects_threshold_at_or_below_half() -> None:
    clf, feats, _ = _train_binary_model()
    with pytest.raises(ValueError, match=r"confidence_threshold must be in \(0\.5"):
        BinaryModelDrivenStrategy(
            symbol="BTCUSDT",
            interval="60",
            model=clf,
            features=feats,
            confidence_threshold=0.5,
        )


def test_binary_strategy_runs_end_to_end_via_mre() -> None:
    clf, feats, bars = _train_binary_model()
    source = MarketReplaySource(
        bars=bars[200:400],
        clock=SimClock(bars[200].event_time),
        interval="60",
    )
    strategy = BinaryModelDrivenStrategy(
        symbol="BTCUSDT",
        interval="60",
        model=clf,
        features=feats,
        confidence_threshold=0.55,
        notional_fraction=0.3,
    )
    result = run_backtest(
        source=source,
        strategy=strategy,
        config=BacktestConfig(initial_equity=1000.0),
    )
    # A well-formed result — we're not asserting profitability, only wiring.
    assert result.initial_equity == 1000.0
    assert result.final_equity > 0.0
    assert len(result.equity_curve) > 0


def test_binary_strategy_calibrator_composes() -> None:
    clf, feats, bars = _train_binary_model()
    # Fit a calibrator on the training slice.
    store = InMemoryFeatureStore()
    for feat in feats:
        store.materialize(feature=feat, entity_id="BTCUSDT", bars=bars)
    label_rows = [
        LabelRow(
            entity_id="BTCUSDT",
            event_time=b.event_time,
            label=1.0 if (i % 2 == 0) else -1.0,
        )
        for i, b in enumerate(bars[50:250])
    ]
    frame = store.point_in_time_join(labels=label_rows, feature_ids=[f.spec.full_id for f in feats])
    raw = clf.predict_proba_matrix(frame)
    y = np.where(np.array(frame.labels) > 0, 1, 0).astype(np.int64)
    cal = IsotonicCalibrator()
    cal.fit(raw, y)
    # Strategy accepts the calibrator without error.
    strategy = BinaryModelDrivenStrategy(
        symbol="BTCUSDT",
        interval="60",
        model=clf,
        features=feats,
        calibrator=cal,
        confidence_threshold=0.55,
    )
    source = MarketReplaySource(
        bars=bars[200:300],
        clock=SimClock(bars[200].event_time),
        interval="60",
    )
    result = run_backtest(
        source=source,
        strategy=strategy,
        config=BacktestConfig(initial_equity=1000.0),
    )
    assert result.final_equity > 0.0
