"""Declarative experiment specification for the research framework.

An `ExperimentSpec` is the complete, reproducible description of one
walk-forward evaluation: the raw data, the feature set, the label rule,
the model hyperparameters, the strategy execution rules, the WFO
schedule, the backtest cost model, and the robustness gates. Serialising
one to JSON and re-loading it must yield the identical experiment down
to the reproducibility hash.

The `fingerprint` property is a stable sha256 over the canonical JSON of
the spec — two experiments with the same fingerprint are guaranteed to
produce the same fold results (given the same input data + code).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DataSpec:
    csv_path: str  # relative to repo root, or absolute
    symbol: str
    interval: str = "60"


@dataclass(frozen=True, slots=True)
class LabelSpec:
    kind: str = "triple_barrier"
    horizon_bars: int = 6
    up_pct: float = 0.01
    down_pct: float = 0.01
    # "3class" = down/flat/up multiclass model (default, historical behavior).
    # "2class_directional" = drop flat rows from training and fit a binary
    # down-vs-up classifier + BinaryModelDrivenStrategy at inference.
    mode: str = "3class"

    def __post_init__(self) -> None:
        if self.kind != "triple_barrier":
            raise ValueError(f"unknown label kind {self.kind!r}")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.up_pct <= 0 or self.down_pct <= 0:
            raise ValueError("up_pct and down_pct must be > 0")
        if self.mode not in {"3class", "2class_directional"}:
            raise ValueError(
                f"label mode must be '3class' or '2class_directional'; got {self.mode!r}"
            )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    n_estimators: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 5
    max_depth: int = -1
    calibration_fraction: float = 0.2

    def to_lightgbm_config(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "max_depth": self.max_depth,
        }


@dataclass(frozen=True, slots=True)
class StrategySpec:
    confidence_threshold: float = 0.55
    notional_fraction: float = 0.5
    allow_short: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not 0.0 < self.notional_fraction <= 1.0:
            raise ValueError("notional_fraction must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class WFOSpec:
    train_bars: int
    test_bars: int
    step_bars: int
    expanding: bool = False

    def __post_init__(self) -> None:
        if self.train_bars < 1 or self.test_bars < 1 or self.step_bars < 1:
            raise ValueError("train_bars, test_bars, step_bars must all be >= 1")


@dataclass(frozen=True, slots=True)
class BacktestSpec:
    initial_equity: float = 10_000.0
    fee_bps: float = 5.5
    slippage_bps: float = 5.0
    bars_per_year: int = 24 * 365

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be > 0")
        if self.bars_per_year < 1:
            raise ValueError("bars_per_year must be >= 1")


@dataclass(frozen=True, slots=True)
class RobustnessGateSpec:
    """Hard-fail rules; an experiment that fails ANY of these is 'rejected'."""

    max_fold_drawdown_pct: float = 25.0
    min_pct_folds_positive_cas: float = 0.5
    min_fills_per_fold: int = 0
    min_folds_with_trades: int = 1
    max_annualized_turnover: float = 100.0


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    data: DataSpec
    features: tuple[str, ...]
    label: LabelSpec = field(default_factory=LabelSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    strategy: StrategySpec = field(default_factory=StrategySpec)
    wfo: WFOSpec = field(
        default_factory=lambda: WFOSpec(train_bars=8640, test_bars=1440, step_bars=1440)
    )
    backtest: BacktestSpec = field(default_factory=BacktestSpec)
    gates: RobustnessGateSpec = field(default_factory=RobustnessGateSpec)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name must be non-empty")
        if not self.features:
            raise ValueError("features must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["features"] = list(self.features)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentSpec:
        return cls(
            name=d["name"],
            data=DataSpec(**d["data"]),
            features=tuple(d["features"]),
            label=LabelSpec(**d.get("label", {})),
            model=ModelSpec(**d.get("model", {})),
            strategy=StrategySpec(**d.get("strategy", {})),
            wfo=WFOSpec(**d["wfo"]),
            backtest=BacktestSpec(**d.get("backtest", {})),
            gates=RobustnessGateSpec(**d.get("gates", {})),
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ExperimentSpec:
        return cls.from_dict(json.loads(text))

    @property
    def fingerprint(self) -> str:
        """Deterministic sha256 fingerprint over the canonical spec JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
