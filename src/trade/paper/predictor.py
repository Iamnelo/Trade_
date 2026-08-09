"""Frozen-winner bundles and structured prediction records.

A `PaperSymbolBundle` loads one frozen winner (from the Phase 5c freeze
manifest), rebuilds its feature set and the matching strategy — 3-class
`ModelDrivenStrategy` or 2-class `BinaryModelDrivenStrategy` — and produces a
`SymbolDecision` on each confirmed bar.

The strategy is the SAME object used by the backtest and the forward test, so
paper decisions match what was validated bit-for-bit given the same bars. The
`SymbolDecision` additionally carries the calibrated class probabilities (for
the journal) computed over the identical feature vector; a test asserts the
recorded direction agrees with the strategy's target sign.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from trade.data.schemas import KlineRecord
from trade.features.catalog import build_features
from trade.features.protocol import Feature
from trade.model.persistence import load_any_training_artifacts
from trade.mre.source import MarketReplaySource
from trade.mre.types import PortfolioSnapshot, TargetPosition
from trade.strategies.binary_model_driven import BinaryModelDrivenStrategy
from trade.strategies.model_driven import ModelDrivenStrategy
from trade.training.pipeline import BinaryTrainingArtifacts, TrainingArtifacts


@dataclass(frozen=True, slots=True)
class SymbolDecision:
    symbol: str
    event_time: datetime
    close: float
    status: str  # "warmup" | "decided"
    probs: dict[str, float] | None
    chosen_class: str | None
    confidence: float | None
    meets_threshold: bool
    direction: str  # "long" | "short" | "flat"
    target_qty: float | None


class PaperSymbolBundle:
    """One frozen winner wired for live prediction."""

    def __init__(
        self,
        *,
        entry: dict[str, Any],
        repo_root: Path,
        notional_fraction: float | None = None,
    ) -> None:
        self.symbol: str = entry["symbol"]
        self.interval: str = entry["interval"]
        self.label_mode: str = entry["label_mode"]
        self.threshold: float = float(entry["confidence_threshold"])
        self.reproducibility_hash: str = entry["reproducibility_hash"]
        self._allow_short: bool = bool(entry["allow_short"])
        nf = (
            notional_fraction
            if notional_fraction is not None
            else float(entry["notional_fraction"])
        )

        artifacts = load_any_training_artifacts(repo_root / entry["artifacts_dir"])
        if artifacts.reproducibility_hash != self.reproducibility_hash:
            raise ValueError(
                f"{self.symbol}: frozen artifact hash "
                f"{artifacts.reproducibility_hash[:12]} != manifest "
                f"{self.reproducibility_hash[:12]}"
            )
        self._artifacts: TrainingArtifacts | BinaryTrainingArtifacts = artifacts
        self._features: tuple[Feature, ...] = tuple(build_features(list(entry["feature_ids"])))
        self.max_lookback: int = max(f.spec.lookback_bars for f in self._features)

        if self.label_mode == "3class":
            if not isinstance(artifacts, TrainingArtifacts):
                raise TypeError(f"{self.symbol}: expected 3-class artifacts")
            self._strategy: ModelDrivenStrategy | BinaryModelDrivenStrategy = ModelDrivenStrategy(
                symbol=self.symbol,
                interval=self.interval,
                model=artifacts.model,
                features=list(self._features),
                calibrator=artifacts.calibrator,
                confidence_threshold=self.threshold,
                notional_fraction=nf,
                allow_short=self._allow_short,
            )
        elif self.label_mode == "2class_directional":
            if not isinstance(artifacts, BinaryTrainingArtifacts):
                raise TypeError(f"{self.symbol}: expected binary artifacts")
            self._strategy = BinaryModelDrivenStrategy(
                symbol=self.symbol,
                interval=self.interval,
                model=artifacts.model,
                features=list(self._features),
                calibrator=artifacts.calibrator,
                confidence_threshold=self.threshold,
                notional_fraction=nf,
                allow_short=self._allow_short,
            )
        else:
            raise ValueError(f"{self.symbol}: unknown label_mode {self.label_mode!r}")

    @property
    def strategy_name(self) -> str:
        return self._strategy.name

    def _feature_vector(self, source: MarketReplaySource) -> dict[str, float] | None:
        fv: dict[str, float] = {}
        for feat in self._features:
            history = source.history(self.symbol, self.interval, lookback=feat.spec.lookback_bars)
            value = feat.compute(history)
            if value is None:
                return None
            fv[feat.spec.full_id] = value
        return fv

    def _probs(self, fv: dict[str, float]) -> dict[str, float]:
        probs = self._artifacts.model.predict_proba_single(fv)
        if self._artifacts.calibrator is not None:
            probs = self._artifacts.calibrator.transform_single(probs)
        return probs

    def _direction(self, probs: dict[str, float]) -> tuple[str, str, float, bool]:
        """Return (chosen_class, direction, confidence, meets_threshold).

        Mirrors the strategy's own decision rule exactly so the journalled
        record and the executed target never disagree.
        """
        if self.label_mode == "3class":
            chosen = max(probs, key=lambda k: probs[k])
            conf = probs[chosen]
            meets = conf >= self.threshold
            if not meets or chosen == "flat":
                return chosen, "flat", conf, meets
            if chosen == "down" and not self._allow_short:
                return chosen, "flat", conf, meets
            return chosen, ("long" if chosen == "up" else "short"), conf, meets
        # binary: symmetric threshold on P(up)
        p_up = probs["up"]
        if p_up > self.threshold:
            return "up", "long", p_up, True
        if p_up < 1.0 - self.threshold:
            direction = "flat" if not self._allow_short else "short"
            return "down", direction, 1.0 - p_up, True
        chosen = "up" if p_up >= 0.5 else "down"
        return chosen, "flat", max(p_up, 1.0 - p_up), False

    def decide(
        self,
        *,
        source: MarketReplaySource,
        bar: KlineRecord,
        snapshot: PortfolioSnapshot,
    ) -> SymbolDecision:
        fv = self._feature_vector(source)
        if fv is None:
            return SymbolDecision(
                symbol=self.symbol,
                event_time=bar.event_time,
                close=bar.close,
                status="warmup",
                probs=None,
                chosen_class=None,
                confidence=None,
                meets_threshold=False,
                direction="flat",
                target_qty=None,
            )
        probs = self._probs(fv)
        chosen, direction, conf, meets = self._direction(probs)
        targets: Sequence[TargetPosition] = self._strategy.on_bar(bar, source, snapshot)
        target_qty = targets[0].target_qty if targets else 0.0
        return SymbolDecision(
            symbol=self.symbol,
            event_time=bar.event_time,
            close=bar.close,
            status="decided",
            probs=probs,
            chosen_class=chosen,
            confidence=conf,
            meets_threshold=meets,
            direction=direction,
            target_qty=target_qty,
        )
