"""End-to-end 2-year walk-forward evaluation for BTCUSDT and ETHUSDT.

Loads raw hourly klines from the CSVs at the repo root, wires the five
price-only catalog features, then runs `run_walk_forward_model` per
symbol with a rolling 12-month train / 2-month test / 2-month step
schedule (~6 folds per symbol across year 2 of the data).

Emits a JSON report with per-fold metrics and reproducibility hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from trade.data.schemas import KlineRecord
from trade.features.catalog import build_features
from trade.mre.types import BacktestConfig
from trade.reproducibility.git import current_git_sha, lockfile_sha
from trade.wfo.model_runner import ModelFoldResult, run_walk_forward_model
from trade.wfo.splitter import walk_forward_splits

REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURE_IDS: tuple[str, ...] = (
    "log_return@5",
    "realized_vol@20",
    "atr@14",
    "macd_hist@12_26_9",
    "rsi_close@14",
)

# Hourly bars, 24 * 365 = 8760 bars/year.
BARS_PER_YEAR = 24 * 365
TRAIN_BARS = 24 * 30 * 12  # ~12 months
TEST_BARS = 24 * 30 * 2  # ~2 months
STEP_BARS = TEST_BARS

INITIAL_EQUITY = 10_000.0
FEE_BPS = 5.5
SLIPPAGE_BPS = 5.0

LABEL_HORIZON_BARS = 6
LABEL_UP_PCT = 0.01
LABEL_DOWN_PCT = 0.01

CONFIDENCE_THRESHOLD = 0.55
NOTIONAL_FRACTION = 0.5


def load_csv_bars(path: Path, *, symbol: str, interval: str = "60") -> list[KlineRecord]:
    """Read a CSV with header event_time_ms,open,high,low,close,volume,turnover."""
    ingest_time = datetime.now(tz=UTC)
    bars: list[KlineRecord] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_time = datetime.fromtimestamp(int(row["event_time_ms"]) / 1000, tz=UTC)
            bars.append(
                KlineRecord(
                    source="bybit",
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    event_time=event_time,
                    ingest_time=ingest_time,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    turnover=float(row["turnover"]),
                )
            )
    bars.sort(key=lambda b: b.event_time)
    _validate_hourly_gaps(bars, path=path)
    return bars


def _validate_hourly_gaps(bars: Sequence[KlineRecord], *, path: Path) -> None:
    prev = None
    for b in bars:
        if prev is not None:
            gap = (b.event_time - prev).total_seconds()
            if gap != 3600.0:
                raise ValueError(
                    f"{path.name}: non-hourly gap at {b.event_time.isoformat()} "
                    f"(previous {prev.isoformat()}, delta {gap:.0f}s)"
                )
        prev = b.event_time


def summarize_fold(fold: ModelFoldResult) -> dict[str, object]:
    r = fold.report
    return {
        "train_start_idx": fold.split.train_start,
        "train_end_idx": fold.split.train_end,
        "test_start_idx": fold.split.test_start,
        "test_end_idx": fold.split.test_end,
        "reproducibility_hash": fold.reproducibility_hash,
        "n_fills": len(fold.backtest.fills),
        "final_equity": fold.backtest.final_equity,
        "total_return": r.total_return,
        "annualized_return": r.annualized_return,
        "sharpe": r.sharpe,
        "cost_adjusted_sharpe": r.cost_adjusted_sharpe,
        "sortino": r.sortino,
        "max_drawdown": r.max_drawdown,
        "hit_rate": r.hit_rate,
        "n_trades": r.n_trades,
    }


def run_for_symbol(
    *,
    symbol: str,
    csv_path: Path,
    git_sha: str,
    lock_sha: str,
) -> dict[str, object]:
    bars = load_csv_bars(csv_path, symbol=symbol)
    print(f"[{symbol}] loaded {len(bars):,} bars ({bars[0].event_time} -> {bars[-1].event_time})")

    splits = walk_forward_splits(
        n_bars=len(bars),
        train_bars=TRAIN_BARS,
        test_bars=TEST_BARS,
        step_bars=STEP_BARS,
        expanding=False,
    )
    print(f"[{symbol}] scheduled {len(splits)} rolling folds "
          f"(train={TRAIN_BARS}h ~12mo, test={TEST_BARS}h ~2mo)")

    features = build_features(list(FEATURE_IDS))
    config = BacktestConfig(
        initial_equity=INITIAL_EQUITY,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )

    dataset_id = f"bybit_kline_{symbol}_60_2024-08-06_2026-08-06"
    report = run_walk_forward_model(
        bars=bars,
        symbol=symbol,
        interval="60",
        features=features,
        label_horizon_bars=LABEL_HORIZON_BARS,
        label_up_pct=LABEL_UP_PCT,
        label_down_pct=LABEL_DOWN_PCT,
        splits=splits,
        config=config,
        bars_per_year=BARS_PER_YEAR,
        dataset_manifest_ids=[dataset_id],
        feature_manifest_ids=list(FEATURE_IDS),
        code_git_sha=git_sha,
        python_lockfile_sha=lock_sha,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        notional_fraction=NOTIONAL_FRACTION,
        allow_short=True,
        cost_bps_per_side=FEE_BPS,
    )

    folds = [summarize_fold(f) for f in report.folds]
    mean_cas = report.mean_cost_adjusted_sharpe
    for i, f in enumerate(folds):
        print(f"[{symbol}] fold {i}: "
              f"sharpe={f['sharpe']:.2f} "
              f"cas={f['cost_adjusted_sharpe']:.2f} "
              f"mdd={f['max_drawdown']:.2%} "
              f"hit={f['hit_rate']:.2%} "
              f"n_trades={f['n_trades']}")
    print(f"[{symbol}] mean cost-adjusted sharpe = {mean_cas:.3f}")
    return {
        "symbol": symbol,
        "n_bars": len(bars),
        "bar_start": bars[0].event_time.isoformat(),
        "bar_end": bars[-1].event_time.isoformat(),
        "n_folds": len(folds),
        "mean_cost_adjusted_sharpe": mean_cas,
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval_reports" / "wfo_2y.json")
    args = parser.parse_args()

    git_sha = current_git_sha(cwd=REPO_ROOT)
    lock_path = REPO_ROOT / "uv.lock"
    lock_sha = lockfile_sha(lock_path) if lock_path.exists() else "no-lockfile"
    if lock_sha == "no-lockfile":
        print("WARNING: uv.lock not present; reproducibility hash uses placeholder")

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "code_git_sha": git_sha,
        "lockfile_sha": lock_sha,
        "config": {
            "features": list(FEATURE_IDS),
            "train_bars": TRAIN_BARS,
            "test_bars": TEST_BARS,
            "step_bars": STEP_BARS,
            "label_horizon_bars": LABEL_HORIZON_BARS,
            "label_up_pct": LABEL_UP_PCT,
            "label_down_pct": LABEL_DOWN_PCT,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "notional_fraction": NOTIONAL_FRACTION,
            "initial_equity": INITIAL_EQUITY,
            "fee_bps": FEE_BPS,
            "slippage_bps": SLIPPAGE_BPS,
        },
        "symbols": [
            run_for_symbol(
                symbol="BTCUSDT",
                csv_path=REPO_ROOT / "BTCUSDT_60_2y.csv",
                git_sha=git_sha,
                lock_sha=lock_sha,
            ),
            run_for_symbol(
                symbol="ETHUSDT",
                csv_path=REPO_ROOT / "ETHUSDT_60_2y.csv",
                git_sha=git_sha,
                lock_sha=lock_sha,
            ),
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
