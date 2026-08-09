"""Freeze the Phase 5b.1.7 daily winners into on-disk artifacts.

This is the first half of the forward-test protocol. It trains each winning
configuration ONCE on ALL available daily bars up to the data cutoff
(2026-08-07), serialises the trained LightGBM booster + isotonic calibrator,
and records a freeze manifest pinning:

- the data cutoff (last training bar event_time),
- today's git SHA + lockfile SHA,
- the exact feature ids, confidence threshold, and label parameters,
- the reproducibility hash of the training run.

Training on the full history through the cutoff (rather than a single WFO
fold's train slice) is deliberate: it is the model you would actually deploy,
and it uses strictly more data than any fold saw. Triple-barrier labels in
the final `horizon_bars` cannot resolve and are trimmed, so the last resolved
label sits ~horizon bars before the cutoff — PIT-safe by construction.

The frozen artifacts are evaluated later by `scripts/forward_test.py` against
genuinely-unseen bars with NO retraining. Freezing today is what lets the
forward window provably post-date the model.

Run:  uv run python scripts/freeze_winners.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade.features.catalog import build_features
from trade.features.store import InMemoryFeatureStore
from trade.labels.triple_barrier import (
    triple_barrier_labels,
    triple_barrier_labels_directional,
)
from trade.model.persistence import (
    save_binary_training_artifacts,
    save_training_artifacts,
)
from trade.reproducibility.git import current_git_sha, lockfile_sha
from trade.research.experiment import ExperimentSpec
from trade.research.runner import load_klines_csv
from trade.training.pipeline import train_model, train_model_binary

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_DIR = REPO_ROOT / "artifacts" / "frozen"


def freeze_one(
    *,
    spec_path: Path,
    code_git_sha: str,
    lockfile_sha_str: str,
    data_root: Path,
    out_root: Path,
) -> dict[str, Any]:
    spec = ExperimentSpec.from_json(spec_path.read_text())
    bars = load_klines_csv(
        data_root / spec.data.csv_path, symbol=spec.data.symbol, interval=spec.data.interval
    )
    if not bars:
        raise ValueError(f"{spec.data.csv_path}: no bars loaded")
    cutoff = bars[-1].event_time

    features = build_features(list(spec.features))
    feature_ids = [f.spec.full_id for f in features]

    store = InMemoryFeatureStore()
    for feat in features:
        store.materialize(feature=feat, entity_id=spec.data.symbol, bars=bars)

    if spec.label.mode == "3class":
        labels = triple_barrier_labels(
            bars,
            horizon_bars=spec.label.horizon_bars,
            up_pct=spec.label.up_pct,
            down_pct=spec.label.down_pct,
        )
    else:
        labels = triple_barrier_labels_directional(
            bars,
            horizon_bars=spec.label.horizon_bars,
            up_pct=spec.label.up_pct,
            down_pct=spec.label.down_pct,
        )
    # Trim the unresolvable horizon tail (labels needing bars past the cutoff).
    labels = labels[: max(0, len(labels) - spec.label.horizon_bars)]
    if not labels:
        raise ValueError(f"{spec.name}: no resolvable labels after horizon trim")

    dataset_id = f"csv:{spec.data.symbol}:{spec.data.interval}:{spec.data.csv_path}"
    out_dir = out_root / spec.name

    if spec.label.mode == "3class":
        artifacts_3c = train_model(
            feature_store=store,
            feature_ids=feature_ids,
            labels=labels,
            dataset_manifest_ids=[dataset_id],
            feature_manifest_ids=feature_ids,
            code_git_sha=code_git_sha,
            python_lockfile_sha=lockfile_sha_str,
            model_config=spec.model.to_lightgbm_config(),
            calibration_fraction=spec.model.calibration_fraction,
        )
        save_training_artifacts(artifacts_3c, out_dir)
        repro_hash = artifacts_3c.reproducibility_hash
        train_rows = artifacts_3c.train_rows
        cal_rows = artifacts_3c.calibration_rows
    else:
        artifacts_bin = train_model_binary(
            feature_store=store,
            feature_ids=feature_ids,
            labels=labels,
            dataset_manifest_ids=[dataset_id],
            feature_manifest_ids=feature_ids,
            code_git_sha=code_git_sha,
            python_lockfile_sha=lockfile_sha_str,
            model_config=spec.model.to_lightgbm_config(),
            calibration_fraction=spec.model.calibration_fraction,
        )
        save_binary_training_artifacts(artifacts_bin, out_dir)
        repro_hash = artifacts_bin.reproducibility_hash
        train_rows = artifacts_bin.train_rows
        cal_rows = artifacts_bin.calibration_rows

    entry = {
        "symbol": spec.data.symbol,
        "spec_name": spec.name,
        "spec_path": str(spec_path.relative_to(REPO_ROOT)),
        "spec_fingerprint": spec.fingerprint,
        "interval": spec.data.interval,
        "artifacts_dir": str(out_dir.relative_to(REPO_ROOT)),
        "reproducibility_hash": repro_hash,
        "label_mode": spec.label.mode,
        "label_horizon_bars": spec.label.horizon_bars,
        "label_up_pct": spec.label.up_pct,
        "label_down_pct": spec.label.down_pct,
        "feature_ids": feature_ids,
        "confidence_threshold": spec.strategy.confidence_threshold,
        "notional_fraction": spec.strategy.notional_fraction,
        "allow_short": spec.strategy.allow_short,
        "bars_per_year": spec.backtest.bars_per_year,
        "fee_bps": spec.backtest.fee_bps,
        "slippage_bps": spec.backtest.slippage_bps,
        "initial_equity": spec.backtest.initial_equity,
        "n_train_labels": len(labels),
        "train_rows": train_rows,
        "calibration_rows": cal_rows,
        "cutoff_event_time": cutoff.isoformat(),
        "last_bar_event_time": bars[-1].event_time.isoformat(),
        "n_bars_through_cutoff": len(bars),
    }
    print(
        f"[{spec.data.symbol}] froze {spec.name} ({spec.label.mode}, {len(feature_ids)} features) "
        f"-> {out_dir.relative_to(REPO_ROOT)}\n"
        f"    cutoff={cutoff.isoformat()}  labels={len(labels)}  "
        f"repro_hash={repro_hash[:12]}"
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--btc-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "winner_btc_daily.json",
    )
    parser.add_argument(
        "--eth-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "winner_eth_daily.json",
    )
    parser.add_argument("--out-root", type=Path, default=FROZEN_DIR)
    parser.add_argument("--lockfile", type=Path, default=REPO_ROOT / "uv.lock")
    args = parser.parse_args()

    git_sha = current_git_sha(cwd=REPO_ROOT)
    lock_sha = lockfile_sha(args.lockfile) if args.lockfile.exists() else "no-lockfile"

    args.out_root.mkdir(parents=True, exist_ok=True)
    entries = [
        freeze_one(
            spec_path=args.btc_spec,
            code_git_sha=git_sha,
            lockfile_sha_str=lock_sha,
            data_root=REPO_ROOT,
            out_root=args.out_root,
        ),
        freeze_one(
            spec_path=args.eth_spec,
            code_git_sha=git_sha,
            lockfile_sha_str=lock_sha,
            data_root=REPO_ROOT,
            out_root=args.out_root,
        ),
    ]

    manifest = {
        "frozen_at": datetime.now(tz=UTC).isoformat(),
        "code_git_sha": git_sha,
        "lockfile_sha": lock_sha,
        "note": (
            "Winners trained on ALL daily bars through the data cutoff. "
            "Evaluate with scripts/forward_test.py on bars strictly after "
            "cutoff_event_time, with NO retraining."
        ),
        "winners": entries,
    }
    manifest_path = args.out_root / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    print(f"\nwrote {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
