"""Tests for ExperimentSpec: validation + JSON roundtrip + fingerprint."""

from __future__ import annotations

import pytest

from trade.research.experiment import (
    BacktestSpec,
    DataSpec,
    ExperimentSpec,
    LabelSpec,
    ModelSpec,
    RobustnessGateSpec,
    StrategySpec,
    WFOSpec,
)


def _base_spec(**overrides) -> ExperimentSpec:
    kwargs = {
        "name": "test",
        "data": DataSpec(csv_path="x.csv", symbol="BTCUSDT"),
        "features": ("log_return@5",),
        "wfo": WFOSpec(train_bars=100, test_bars=20, step_bars=20),
    }
    kwargs.update(overrides)
    return ExperimentSpec(**kwargs)


def test_json_roundtrip_preserves_all_fields() -> None:
    spec = _base_spec(
        model=ModelSpec(n_estimators=250, learning_rate=0.03),
        strategy=StrategySpec(confidence_threshold=0.62, notional_fraction=0.3),
        label=LabelSpec(horizon_bars=12, up_pct=0.015, down_pct=0.015),
        backtest=BacktestSpec(initial_equity=5000.0, fee_bps=7.0),
    )
    reloaded = ExperimentSpec.from_json(spec.to_json())
    assert reloaded == spec


def test_fingerprint_is_deterministic() -> None:
    a = _base_spec()
    b = _base_spec()
    assert a.fingerprint == b.fingerprint


def test_fingerprint_changes_when_any_field_changes() -> None:
    base = _base_spec()
    changed = _base_spec(strategy=StrategySpec(confidence_threshold=0.60))
    assert base.fingerprint != changed.fingerprint


def test_missing_name_rejected() -> None:
    with pytest.raises(ValueError, match="name"):
        _base_spec(name="")


def test_empty_features_rejected() -> None:
    with pytest.raises(ValueError, match="features"):
        _base_spec(features=())


def test_confidence_threshold_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        StrategySpec(confidence_threshold=0.0)
    with pytest.raises(ValueError, match="confidence_threshold"):
        StrategySpec(confidence_threshold=1.5)


def test_notional_fraction_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="notional_fraction"):
        StrategySpec(notional_fraction=0.0)


def test_label_horizon_must_be_positive() -> None:
    with pytest.raises(ValueError, match="horizon_bars"):
        LabelSpec(horizon_bars=0)


def test_unknown_label_kind_rejected() -> None:
    with pytest.raises(ValueError, match="unknown label kind"):
        LabelSpec(kind="meta")


def test_wfo_bars_must_be_positive() -> None:
    with pytest.raises(ValueError, match="train_bars"):
        WFOSpec(train_bars=0, test_bars=10, step_bars=10)


def test_from_dict_accepts_default_sections() -> None:
    minimal = {
        "name": "minimal",
        "data": {"csv_path": "x.csv", "symbol": "BTCUSDT"},
        "features": ["log_return@5"],
        "wfo": {"train_bars": 100, "test_bars": 20, "step_bars": 20},
    }
    spec = ExperimentSpec.from_dict(minimal)
    assert spec.label == LabelSpec()
    assert spec.model == ModelSpec()
    assert spec.strategy == StrategySpec()
    assert spec.backtest == BacktestSpec()
    assert spec.gates == RobustnessGateSpec()
