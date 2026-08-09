"""Paper-trading engine: master switch, simulated execution, journal, safety.

Freezes a tiny synthetic 3-class winner to a tmp dir, wires a bundle + engine,
and drives it with a ReplayFeed — no network, no real artifacts. Covers the
two safety guarantees (master switch off = zero fills; paper-only venue),
armed execution with exits, kill-switch halting, and audit-chain integrity.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from trade.data.schemas import KlineRecord
from trade.features.catalog import build_features
from trade.features.store import InMemoryFeatureStore
from trade.labels.triple_barrier import triple_barrier_labels
from trade.model.persistence import save_training_artifacts
from trade.mre.venue import SimulatedVenue
from trade.paper.config import PaperTradingConfig, TelegramConfig
from trade.paper.engine import PaperTradingEngine
from trade.paper.feed import ReplayFeed
from trade.paper.journal import PaperJournal
from trade.paper.notifier import NullNotifier
from trade.paper.predictor import PaperSymbolBundle
from trade.training.pipeline import train_model

_FEATURES = ["log_return@5", "realized_vol@20"]


def _bars(n: int, *, symbol: str = "BTCUSDT", seed: int = 5) -> list[KlineRecord]:
    rng = np.random.default_rng(seed)
    price = 100.0
    base = datetime(2024, 1, 1, tzinfo=UTC)
    out: list[KlineRecord] = []
    for i in range(n):
        price *= 1.0 + float(rng.normal(0.0008, 0.02))
        h = price * (1 + abs(float(rng.normal(0, 0.008))))
        lo = price * (1 - abs(float(rng.normal(0, 0.008))))
        out.append(
            KlineRecord(
                source="csv",
                category="linear",
                symbol=symbol,
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


def _freeze_synth(tmp_path: Path, *, threshold: float = 0.34) -> tuple[dict[str, Any], list]:
    bars = _bars(240)
    train_bars = bars[:160]
    feats = build_features(_FEATURES)
    store = InMemoryFeatureStore()
    for feat in feats:
        store.materialize(feature=feat, entity_id="BTCUSDT", bars=train_bars)
    labels = triple_barrier_labels(train_bars, horizon_bars=5, up_pct=0.05, down_pct=0.05)
    labels = labels[: max(0, len(labels) - 5)]
    artifacts = train_model(
        feature_store=store,
        feature_ids=_FEATURES,
        labels=labels,
        dataset_manifest_ids=["ds"],
        feature_manifest_ids=_FEATURES,
        code_git_sha="deadbeef",
        python_lockfile_sha="cafef00d",
    )
    out_dir = tmp_path / "winner_synth"
    save_training_artifacts(artifacts, out_dir)
    entry = {
        "symbol": "BTCUSDT",
        "spec_name": "winner_synth",
        "artifacts_dir": str(out_dir),
        "reproducibility_hash": artifacts.reproducibility_hash,
        "interval": "D",
        "label_mode": "3class",
        "label_horizon_bars": 5,
        "label_up_pct": 0.05,
        "label_down_pct": 0.05,
        "feature_ids": _FEATURES,
        "confidence_threshold": threshold,
        "notional_fraction": 0.5,
        "allow_short": True,
        "bars_per_year": 365,
        "fee_bps": 5.5,
        "slippage_bps": 5.0,
        "initial_equity": 10_000.0,
    }
    return entry, bars


def _engine(
    tmp_path: Path, *, execution_enabled: bool, entry: dict[str, Any]
) -> PaperTradingEngine:
    config = PaperTradingConfig(
        execution_enabled=execution_enabled,
        symbols=("BTCUSDT",),
        journal_dir=tmp_path / "journal",
    )
    bundle = PaperSymbolBundle(entry=entry, repo_root=Path("/"))  # abs artifacts_dir
    journal = PaperJournal(config.journal_dir)
    return PaperTradingEngine(
        config=config, bundles=[bundle], journal=journal, notifier=NullNotifier()
    )


def test_master_switch_off_places_no_orders(tmp_path: Path) -> None:
    entry, bars = _freeze_synth(tmp_path)
    engine = _engine(tmp_path, execution_enabled=False, entry=entry)
    for bar in bars:
        engine.process_batch([bar])

    st = engine.state()
    assert st.armed is False
    assert st.n_fills == 0
    assert st.equity == entry["initial_equity"]  # untouched
    assert all(qty == 0.0 for qty in st.positions.values())
    assert st.n_decisions > 0
    engine._journal.verify()  # audit chain intact


def test_master_switch_on_trades_and_exits(tmp_path: Path) -> None:
    entry, bars = _freeze_synth(tmp_path, threshold=0.34)
    engine = _engine(tmp_path, execution_enabled=True, entry=entry)
    for bar in bars:
        engine.process_batch([bar])

    st = engine.state()
    assert st.armed is True
    assert st.n_fills > 0  # low threshold => it trades
    engine._journal.verify()

    lines = _read_jsonl(tmp_path / "journal" / "decisions.jsonl")
    kinds = [line_["kind"] for line_ in lines]
    assert "fill" in kinds  # journal kind not clobbered by payload
    assert "decision" in kinds
    fill_kinds = {line_["fill_kind"] for line_ in lines if line_["kind"] == "fill"}
    assert "OPEN" in fill_kinds  # entries recorded
    assert "EXIT" in fill_kinds  # automatic exits recorded


def test_engine_uses_only_simulated_venue(tmp_path: Path) -> None:
    entry, _ = _freeze_synth(tmp_path)
    engine = _engine(tmp_path, execution_enabled=True, entry=entry)
    assert type(engine._venue) is SimulatedVenue


def test_kill_switch_staleness_halts(tmp_path: Path) -> None:
    entry, bars = _freeze_synth(tmp_path)
    engine = _engine(tmp_path, execution_enabled=True, entry=entry)
    engine.process_batch([bars[0]])
    # Far in the future relative to the last bar => stale.
    engine.check_staleness(now=bars[0].event_time + timedelta(days=10))
    assert engine.state().halted is True
    assert "data_staleness" in engine.state().halted_reasons


def test_async_run_start_stop(tmp_path: Path) -> None:
    entry, bars = _freeze_synth(tmp_path)
    engine = _engine(tmp_path, execution_enabled=False, entry=entry)
    asyncio.run(engine.run(ReplayFeed(bars)))
    lines = _read_jsonl(tmp_path / "journal" / "decisions.jsonl")
    kinds = {line_["kind"] for line_ in lines}
    assert {"engine_start", "engine_stop"} <= kinds
    engine._journal.verify()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def test_config_defaults_execution_disabled() -> None:
    assert PaperTradingConfig().execution_enabled is False


def test_telegram_config_requires_both_fields() -> None:
    assert TelegramConfig(bot_token="x", chat_id=None).is_configured is False
    assert TelegramConfig(bot_token="x", chat_id="y").is_configured is True
