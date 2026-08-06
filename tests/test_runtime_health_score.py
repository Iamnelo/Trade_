"""Tests for AI Health Score composite."""

from __future__ import annotations

import pytest

from trade.runtime.health_score import (
    DEFAULT_HEALTH_CONFIG,
    HealthScoreComputer,
    HealthScoreConfig,
)


def test_composite_is_weighted_average() -> None:
    config = HealthScoreConfig(weights={"a": 1.0, "b": 3.0}, floors={"a": 50.0, "b": 50.0})
    computer = HealthScoreComputer(config)
    score = computer.compute({"a": 100.0, "b": 60.0})
    # weighted = (100 * 1 + 60 * 3) / 4 = 280 / 4 = 70
    assert score.composite == pytest.approx(70.0)


def test_below_floor_lists_names_under_configured_minimum() -> None:
    config = HealthScoreConfig(weights={"a": 1.0, "b": 1.0}, floors={"a": 90.0, "b": 30.0})
    computer = HealthScoreComputer(config)
    score = computer.compute({"a": 80.0, "b": 50.0})
    assert score.below_floor == ("a",)


def test_missing_sub_score_raises() -> None:
    computer = HealthScoreComputer(
        HealthScoreConfig(weights={"a": 1.0, "b": 1.0}, floors={"a": 50.0, "b": 50.0})
    )
    with pytest.raises(ValueError, match="missing sub-scores"):
        computer.compute({"a": 80.0})


def test_out_of_range_sub_score_raises() -> None:
    computer = HealthScoreComputer(HealthScoreConfig(weights={"a": 1.0}, floors={"a": 50.0}))
    with pytest.raises(ValueError, match="in \\[0, 100\\]"):
        computer.compute({"a": 150.0})


def test_config_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        HealthScoreConfig(weights={"a": -1.0}, floors={"a": 50.0})


def test_config_rejects_zero_total_weight() -> None:
    with pytest.raises(ValueError, match="sum to > 0"):
        HealthScoreConfig(weights={"a": 0.0}, floors={"a": 50.0})


def test_config_rejects_out_of_range_floor() -> None:
    with pytest.raises(ValueError, match="\\[0, 100\\]"):
        HealthScoreConfig(weights={"a": 1.0}, floors={"a": 150.0})


def test_default_config_covers_v1_spec_sub_scores() -> None:
    expected = {
        "calibration",
        "feature_drift",
        "prediction_drift",
        "rolling_hit_rate",
        "confidence_coherence",
        "operational",
    }
    assert set(DEFAULT_HEALTH_CONFIG.weights) == expected
    assert set(DEFAULT_HEALTH_CONFIG.floors) == expected


def test_default_config_with_perfect_sub_scores_yields_100() -> None:
    computer = HealthScoreComputer(DEFAULT_HEALTH_CONFIG)
    score = computer.compute(dict.fromkeys(DEFAULT_HEALTH_CONFIG.weights, 100.0))
    assert score.composite == pytest.approx(100.0)
    assert score.below_floor == ()
