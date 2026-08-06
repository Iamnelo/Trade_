"""Tests for ModelDrivenStrategy — end-to-end through the MRE."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade.data.schemas import KlineRecord
from trade.features.definitions.atr14 import ATR14
from trade.features.definitions.log_return import LogReturnN
from trade.features.definitions.realized_vol import RealizedVolN
from trade.features.store import InMemoryFeatureStore
from trade.labels.triple_barrier import triple_barrier_labels
from trade.mre.backtest import run_backtest
from trade.mre.clock import SimClock
from trade.mre.source import MarketReplaySource
from trade.mre.types import BacktestConfig
from trade.strategies.model_driven import ModelDrivenStrategy
from trade.training.pipeline import train_model


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


def _train() -> tuple[list[KlineRecord], list, object, object]:
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
    return bars, feats, artifacts.model, artifacts.calibrator


def test_model_driven_strategy_runs_end_to_end() -> None:
    _, feats, model, calibrator = _train()

    # Test on a fresh out-of-sample chunk.
    test_bars = _bars(200, seed=99)
    src = MarketReplaySource(bars=test_bars, clock=SimClock(test_bars[0].event_time), interval="60")
    strat = ModelDrivenStrategy(
        symbol="BTCUSDT",
        interval="60",
        model=model,
        features=feats,
        calibrator=calibrator,
        confidence_threshold=0.35,  # low so we get some trades on a small test set
        notional_fraction=0.25,
    )
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    assert result.strategy_name == "model_lgbm(BTCUSDT)"
    # Equity curve populated for every bar.
    assert len(result.equity_curve) == len(test_bars)


def test_model_driven_flat_when_history_insufficient() -> None:
    _, feats, model, calibrator = _train()
    # Too few bars for the longest feature's lookback -> no fills.
    short_bars = _bars(5, seed=7)
    src = MarketReplaySource(
        bars=short_bars, clock=SimClock(short_bars[0].event_time), interval="60"
    )
    strat = ModelDrivenStrategy(
        symbol="BTCUSDT",
        interval="60",
        model=model,
        features=feats,
        calibrator=calibrator,
    )
    result = run_backtest(
        source=src,
        strategy=strat,
        config=BacktestConfig(initial_equity=1000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    assert result.fills == ()


def test_model_driven_rejects_feature_mismatch() -> None:
    _, feats, model, _ = _train()
    # Swap one feature for a different id -> mismatch.
    with pytest.raises(ValueError, match="feature_ids mismatch"):
        ModelDrivenStrategy(
            symbol="BTCUSDT",
            interval="60",
            model=model,
            features=[feats[0], ATR14()],
        )


def test_model_driven_rejects_bad_thresholds() -> None:
    _, feats, model, _ = _train()
    with pytest.raises(ValueError):
        ModelDrivenStrategy(
            symbol="BTCUSDT",
            interval="60",
            model=model,
            features=feats,
            notional_fraction=0.0,
        )
    with pytest.raises(ValueError):
        ModelDrivenStrategy(
            symbol="BTCUSDT",
            interval="60",
            model=model,
            features=feats,
            confidence_threshold=1.5,
        )
