"""Tests for the feature catalog registry."""

from __future__ import annotations

import pytest

from trade.features.catalog import (
    build_feature,
    build_features,
    registered_feature_names,
)


def test_registered_feature_names_are_stable() -> None:
    # This set is intentionally small in V1; extending it is a deliberate change.
    names = set(registered_feature_names())
    assert names >= {"log_return", "realized_vol", "atr", "macd_hist", "rsi_close"}


def test_build_feature_round_trip_matches_spec() -> None:
    feat = build_feature("log_return@5")
    assert feat.spec.full_id == "log_return@5"
    assert feat.spec.lookback_bars == 6

    macd = build_feature("macd_hist@12_26_9")
    assert macd.spec.full_id == "macd_hist@12_26_9"

    atr = build_feature("atr@14")
    assert atr.spec.full_id == "atr@14"


def test_build_features_preserves_order() -> None:
    ids = ["log_return@5", "atr@14", "realized_vol@20"]
    got = build_features(ids)
    assert [f.spec.full_id for f in got] == ids


def test_build_feature_rejects_bad_ids() -> None:
    with pytest.raises(ValueError, match="name@version"):
        build_feature("log_return")
    with pytest.raises(KeyError, match="unknown feature"):
        build_feature("nonsense@1")
    with pytest.raises(ValueError):
        build_feature("macd_hist@12")  # wrong version arity
    with pytest.raises(ValueError):
        build_feature("rsi_close@7")  # only @14 is registered


def test_build_feature_rejects_negative_windows() -> None:
    with pytest.raises(ValueError):
        build_feature("log_return@0")
    with pytest.raises(ValueError):
        build_feature("realized_vol@1")
