"""Tests for MatrixSpec expansion, override paths, and ranking."""

from __future__ import annotations

import pytest

from trade.research.experiment import DataSpec, ExperimentSpec, WFOSpec
from trade.research.matrix import (
    MatrixCellRow,
    MatrixSpec,
    expand_matrix,
    rank_matrix,
)


def _base_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="base",
        data=DataSpec(csv_path="x.csv", symbol="BTCUSDT"),
        features=("log_return@5",),
        wfo=WFOSpec(train_bars=100, test_bars=20, step_bars=20),
    )


def test_matrix_json_roundtrip() -> None:
    m = MatrixSpec(
        name="m",
        base_spec=_base_spec(),
        grid={
            "label.mode": ("3class", "2class_directional"),
            "label.horizon_bars": (6, 24),
        },
    )
    reloaded = MatrixSpec.from_dict(m.to_dict())
    assert reloaded == m
    assert reloaded.n_cells == 4


def test_matrix_expands_cartesian_product() -> None:
    m = MatrixSpec(
        name="m",
        base_spec=_base_spec(),
        grid={
            "label.mode": ("3class", "2class_directional"),
            "label.horizon_bars": (6, 12, 24),
        },
    )
    specs = expand_matrix(m)
    assert len(specs) == 6
    modes = {s.label.mode for s in specs}
    horizons = {s.label.horizon_bars for s in specs}
    assert modes == {"3class", "2class_directional"}
    assert horizons == {6, 12, 24}
    # Names are unique (they encode the overrides).
    assert len({s.name for s in specs}) == 6
    # Fingerprints are unique (specs differ in at least one grid value).
    assert len({s.fingerprint for s in specs}) == 6


def test_matrix_supports_features_override() -> None:
    m = MatrixSpec(
        name="m",
        base_spec=_base_spec(),
        grid={
            "features": (
                ("log_return@5",),
                ("log_return@5", "realized_vol@20"),
            ),
        },
    )
    specs = expand_matrix(m)
    assert len(specs) == 2
    assert specs[0].features == ("log_return@5",)
    assert specs[1].features == ("log_return@5", "realized_vol@20")


def test_matrix_rejects_unknown_section() -> None:
    with pytest.raises(ValueError, match="unknown section"):
        MatrixSpec(
            name="bad",
            base_spec=_base_spec(),
            grid={"nonexistent.thing": ("a", "b")},
        )


def test_matrix_rejects_empty_grid() -> None:
    with pytest.raises(ValueError, match="grid must be non-empty"):
        MatrixSpec(name="empty", base_spec=_base_spec(), grid={})


def test_matrix_rejects_empty_value_list() -> None:
    with pytest.raises(ValueError, match="has no values"):
        MatrixSpec(
            name="empty_vals",
            base_spec=_base_spec(),
            grid={"label.horizon_bars": ()},
        )


def test_rank_matrix_orders_by_pass_then_consistency() -> None:
    def _row(name: str, passed: bool, cons: float, mean: float = 0.0) -> MatrixCellRow:
        return MatrixCellRow(
            cell_index=0,
            name=name,
            fingerprint="abc",
            passed=passed,
            mean_cost_adjusted_sharpe=mean,
            consistency_score=cons,
            pct_folds_positive_cas=0.5,
            max_fold_drawdown_pct=2.0,
            annualized_turnover=5.0,
            total_fills=10,
            reasons_failed=(),
            spec=_base_spec(),
            result=None,  # type: ignore[arg-type]
        )

    rows = [
        _row("lucky_failed", passed=False, cons=99.0),
        _row("mid_pass", passed=True, cons=0.5),
        _row("high_pass", passed=True, cons=1.5),
    ]
    ordered = rank_matrix(rows)
    assert [r.name for r in ordered] == ["high_pass", "mid_pass", "lucky_failed"]
