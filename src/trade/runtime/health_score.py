"""AI Health Score composite.

`HealthScoreComputer.compute` takes a dict of pre-computed sub-scores
(each in [0, 100]) and returns:

- a weighted composite (also in [0, 100])
- the list of sub-score names whose value is below its configured floor

Sub-score COMPUTATION lives elsewhere (Phase 4b2). The V1_SPEC-listed
sub-scores are: calibration (ECE / Brier), feature_drift (PSI/KS),
prediction_drift (KS), rolling_hit_rate, confidence_coherence,
operational (latency, DQ). This composite scaffold is independent of
which sub-scores are supplied — new ones can be added without changing
this module, as long as their weights + floors are provided in the
config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class HealthScoreConfig:
    weights: Mapping[str, float]
    floors: Mapping[str, float]

    def __post_init__(self) -> None:
        for name, w in self.weights.items():
            if w < 0:
                raise ValueError(f"weight for {name!r} must be non-negative (got {w})")
        for name, f in self.floors.items():
            if not 0.0 <= f <= 100.0:
                raise ValueError(f"floor for {name!r} must be in [0, 100] (got {f})")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("weights must sum to > 0")


@dataclass(frozen=True, slots=True)
class HealthScore:
    composite: float
    sub_scores: Mapping[str, float]
    below_floor: tuple[str, ...] = field(default_factory=tuple)


class HealthScoreComputer:
    def __init__(self, config: HealthScoreConfig) -> None:
        self._config = config

    @property
    def config(self) -> HealthScoreConfig:
        return self._config

    def compute(self, sub_scores: Mapping[str, float]) -> HealthScore:
        weights = self._config.weights
        floors = self._config.floors

        missing = set(weights) - set(sub_scores)
        if missing:
            raise ValueError(f"missing sub-scores for weighted composite: {sorted(missing)}")
        for name, value in sub_scores.items():
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"sub-score {name!r} must be in [0, 100] (got {value})")

        total_weight = sum(weights.values())
        weighted = sum(sub_scores[name] * weights[name] for name in weights)
        composite = weighted / total_weight

        below = tuple(
            sorted(
                name
                for name, value in sub_scores.items()
                if name in floors and value < floors[name]
            )
        )
        return HealthScore(composite=composite, sub_scores=dict(sub_scores), below_floor=below)


DEFAULT_HEALTH_CONFIG = HealthScoreConfig(
    weights={
        "calibration": 1.0,
        "feature_drift": 1.0,
        "prediction_drift": 1.0,
        "rolling_hit_rate": 1.0,
        "confidence_coherence": 1.0,
        "operational": 1.0,
    },
    floors={
        "calibration": 60.0,
        "feature_drift": 60.0,
        "prediction_drift": 60.0,
        "rolling_hit_rate": 40.0,
        "confidence_coherence": 40.0,
        "operational": 80.0,
    },
)
