"""Focused funding-feature ablation against the Phase 5b.1.7 winners.

For each of the two winning cells (BTC and ETH), runs the WFO twice:
- Baseline: the cell's original feature set
- Baseline + funding_3: same + funding_rate@1 + funding_zscore@21 + funding_regime@9_63

Because funding features carry pre-loaded data at construction, they
can't come from the string-based catalog. This script builds them
directly from a funding CSV and hands the full feature list to the WFO
runner.

Every WFO invocation uses the same 3y-train/6mo-test/6mo-step schedule,
same fixed θ=0.55, same fee+slippage model, same gates — apples to
apples with 5b.1.7. Point-in-time correctness is guaranteed by the
FundingRate/FundingZScore/FundingRegime `_latest_index` binary search:
no funding record with event_time > bar.event_time is ever visible.

Emits one markdown + JSON report per symbol comparing baseline vs
funded on every requested metric (mean CAS, consistency, pct positive
folds, max DD, annual turnover, total fills, expectancy, win rate,
profit factor, AUC-ROC, ECE, oracle capture ratio, net return).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade.data.schemas import FundingRecord
from trade.features.catalog import build_features
from trade.features.definitions.funding import (
    FundingRate,
    FundingRegime,
    FundingZScoreN,
)
from trade.features.protocol import Feature
from trade.metrics.trade_metrics import compute_trade_metrics
from trade.mre.types import BacktestConfig
from trade.reproducibility.git import current_git_sha, lockfile_sha
from trade.research.diagnostics import (
    compute_classifier_diagnostics,
    compute_oracle_capture,
    score_holdout,
)
from trade.research.experiment import ExperimentSpec
from trade.research.robustness import compute_robustness, evaluate_gates
from trade.research.runner import load_klines_csv
from trade.wfo.model_runner import run_walk_forward_threshold_sweep
from trade.wfo.splitter import walk_forward_splits

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Funding CSV loader
# ---------------------------------------------------------------------------


def load_funding_csv(path: Path, *, symbol: str) -> list[FundingRecord]:
    """CSV columns: event_time_ms, funding_rate."""
    ingest_time = datetime.now(tz=UTC)
    out: list[FundingRecord] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_time = datetime.fromtimestamp(int(row["event_time_ms"]) / 1000, tz=UTC)
            out.append(
                FundingRecord(
                    source="csv",
                    category="linear",
                    symbol=symbol,
                    event_time=event_time,
                    ingest_time=ingest_time,
                    funding_rate=float(row["funding_rate"]),
                )
            )
    out.sort(key=lambda r: r.event_time)
    return out


# ---------------------------------------------------------------------------
# Ablation arm: run one WFO + collect all metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmResult:
    name: str
    fingerprint_prefix: str
    n_folds: int
    n_folds_with_trades: int
    pct_folds_positive_cas: float
    mean_cost_adjusted_sharpe: float
    median_cost_adjusted_sharpe: float
    min_cost_adjusted_sharpe: float
    std_cost_adjusted_sharpe: float
    max_fold_drawdown_pct: float
    annualized_turnover: float
    consistency_score: float
    total_fills: int
    gate_passed: bool
    gate_reasons_failed: tuple[str, ...]
    # Per-fold aggregates (mean across folds where computable)
    mean_total_return_pct: float
    mean_sharpe: float
    mean_sortino: float
    mean_hit_rate: float
    mean_win_rate: float
    mean_expectancy_per_trade: float
    mean_profit_factor: float | None
    mean_auc_roc: float | None
    mean_ece: float | None
    mean_oracle_capture_long_short: float


def _run_arm(
    *,
    spec: ExperimentSpec,
    features: Sequence[Feature],
    code_git_sha: str,
    lockfile_sha_str: str,
    data_root: Path,
) -> ArmResult:
    bars = load_klines_csv(
        data_root / spec.data.csv_path, symbol=spec.data.symbol, interval=spec.data.interval
    )
    config = BacktestConfig(
        initial_equity=spec.backtest.initial_equity,
        fee_bps=spec.backtest.fee_bps,
        slippage_bps=spec.backtest.slippage_bps,
    )
    splits = walk_forward_splits(
        n_bars=len(bars),
        train_bars=spec.wfo.train_bars,
        test_bars=spec.wfo.test_bars,
        step_bars=spec.wfo.step_bars,
        expanding=spec.wfo.expanding,
    )
    dataset_id = f"csv:{spec.data.symbol}:{spec.data.interval}:{spec.data.csv_path}"
    feature_ids = [f.spec.full_id for f in features]

    sweep = run_walk_forward_threshold_sweep(
        bars=bars,
        symbol=spec.data.symbol,
        interval=spec.data.interval,
        features=features,
        label_horizon_bars=spec.label.horizon_bars,
        label_up_pct=spec.label.up_pct,
        label_down_pct=spec.label.down_pct,
        splits=splits,
        config=config,
        bars_per_year=spec.backtest.bars_per_year,
        dataset_manifest_ids=[dataset_id],
        feature_manifest_ids=feature_ids,
        code_git_sha=code_git_sha,
        python_lockfile_sha=lockfile_sha_str,
        thresholds=[spec.strategy.confidence_threshold],
        model_config=spec.model.to_lightgbm_config(),
        notional_fraction=spec.strategy.notional_fraction,
        allow_short=spec.strategy.allow_short,
        calibration_fraction=spec.model.calibration_fraction,
        cost_bps_per_side=spec.backtest.fee_bps,
        label_mode=spec.label.mode,
    )

    fold_reports = [fold.per_threshold[0].result.report for fold in sweep.folds]
    rb = compute_robustness(
        fold_reports,
        bars_per_year=spec.backtest.bars_per_year,
        test_bars_per_fold=spec.wfo.test_bars,
    )
    gate = evaluate_gates(rb, gates=spec.gates, fold_reports=fold_reports)

    # Per-fold classifier diagnostics + oracle capture + trade metrics
    aucs: list[float] = []
    eces: list[float] = []
    win_rates: list[float] = []
    expectancies: list[float] = []
    pfs_finite: list[float] = []
    oc_ls: list[float] = []
    for fold in sweep.folds:
        cell = fold.per_threshold[0]
        # Classifier diagnostics on the horizon-trimmed holdout
        probs, y_true = score_holdout(
            sorted_bars=fold.trained.sorted_bars,
            symbol=spec.data.symbol,
            features=list(fold.trained.features),
            model=fold.trained.artifacts.model,
            calibrator=fold.trained.artifacts.calibrator,
            test_start_idx=fold.split.test_start,
            test_end_idx=fold.split.test_end,
            label_horizon_bars=spec.label.horizon_bars,
            label_up_pct=spec.label.up_pct,
            label_down_pct=spec.label.down_pct,
            label_mode=spec.label.mode,
        )
        clf = compute_classifier_diagnostics(probs=probs, y_true_int=y_true)
        if clf.auc_roc_macro is not None:
            aucs.append(clf.auc_roc_macro)
        if clf.ece_macro is not None:
            eces.append(clf.ece_macro)
        # Trade metrics on the fold's fills
        tm = compute_trade_metrics(cell.result.backtest.fills)
        if tm.n_trades > 0:
            win_rates.append(tm.win_rate)
            expectancies.append(tm.expectancy_per_trade)
            if tm.profit_factor != float("inf"):
                pfs_finite.append(tm.profit_factor)
        # Oracle capture on the test slice
        test_bars_slice = fold.trained.sorted_bars[fold.split.test_start : fold.split.test_end]
        oc = compute_oracle_capture(
            test_bars=test_bars_slice,
            strategy_final_equity=cell.result.backtest.final_equity,
            initial_equity=cell.result.backtest.initial_equity,
        )
        oc_ls.append(oc.capture_ratio_vs_long_short)

    def _mean_or_none(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    mean_total_return_pct = (
        sum(f.total_return_pct for f in fold_reports) / len(fold_reports) if fold_reports else 0.0
    )
    return ArmResult(
        name=spec.name,
        fingerprint_prefix=spec.fingerprint[:12],
        n_folds=rb.n_folds,
        n_folds_with_trades=rb.n_folds_with_trades,
        pct_folds_positive_cas=rb.pct_folds_positive_cas,
        mean_cost_adjusted_sharpe=rb.mean_cost_adjusted_sharpe,
        median_cost_adjusted_sharpe=rb.median_cost_adjusted_sharpe,
        min_cost_adjusted_sharpe=rb.min_cost_adjusted_sharpe,
        std_cost_adjusted_sharpe=rb.std_cost_adjusted_sharpe,
        max_fold_drawdown_pct=rb.max_fold_drawdown_pct,
        annualized_turnover=rb.annualized_turnover,
        consistency_score=rb.consistency_score,
        total_fills=rb.total_fills,
        gate_passed=gate.passed,
        gate_reasons_failed=gate.reasons_failed,
        mean_total_return_pct=mean_total_return_pct,
        mean_sharpe=sum(f.sharpe for f in fold_reports) / len(fold_reports)
        if fold_reports
        else 0.0,
        mean_sortino=sum(f.sortino for f in fold_reports) / len(fold_reports)
        if fold_reports
        else 0.0,
        mean_hit_rate=sum(f.hit_rate for f in fold_reports if f.n_fills > 0)
        / max(1, sum(1 for f in fold_reports if f.n_fills > 0)),
        mean_win_rate=_mean_or_none(win_rates) or 0.0,
        mean_expectancy_per_trade=_mean_or_none(expectancies) or 0.0,
        mean_profit_factor=_mean_or_none(pfs_finite),
        mean_auc_roc=_mean_or_none(aucs),
        mean_ece=_mean_or_none(eces),
        mean_oracle_capture_long_short=_mean_or_none(oc_ls) or 0.0,
    )


# ---------------------------------------------------------------------------
# Public entry: run one symbol's ablation
# ---------------------------------------------------------------------------


def run_symbol_ablation(
    *,
    spec_path: Path,
    funding_csv_path: Path,
    code_git_sha: str,
    lockfile_sha_str: str,
    data_root: Path,
) -> dict[str, Any]:
    spec = ExperimentSpec.from_json(spec_path.read_text())
    funding = load_funding_csv(funding_csv_path, symbol=spec.data.symbol)
    print(
        f"[{spec.data.symbol}] loaded spec={spec.name} ({len(spec.features)} baseline features), "
        f"funding_records={len(funding)} "
        f"({funding[0].event_time} -> {funding[-1].event_time})"
    )

    baseline_features = build_features(list(spec.features))
    funding_features = [
        FundingRate(funding),
        FundingZScoreN(funding, window=21),
        FundingRegime(funding, short_window=9, long_window=63),
    ]

    print(f"[{spec.data.symbol}] arm A — baseline ({len(baseline_features)} features)")
    arm_baseline = _run_arm(
        spec=spec,
        features=baseline_features,
        code_git_sha=code_git_sha,
        lockfile_sha_str=lockfile_sha_str,
        data_root=data_root,
    )
    print(
        f"[{spec.data.symbol}] arm B — baseline + funding_3 ({len(baseline_features) + 3} features)"
    )
    arm_funded = _run_arm(
        spec=spec,
        features=[*baseline_features, *funding_features],
        code_git_sha=code_git_sha,
        lockfile_sha_str=lockfile_sha_str,
        data_root=data_root,
    )

    return {
        "symbol": spec.data.symbol,
        "spec_name": spec.name,
        "baseline_arm": asdict(arm_baseline),
        "funded_arm": asdict(arm_funded),
        "deltas": {
            "d_mean_cas": arm_funded.mean_cost_adjusted_sharpe
            - arm_baseline.mean_cost_adjusted_sharpe,
            "d_consistency": arm_funded.consistency_score - arm_baseline.consistency_score,
            "d_pct_folds_positive": arm_funded.pct_folds_positive_cas
            - arm_baseline.pct_folds_positive_cas,
            "d_max_dd": arm_funded.max_fold_drawdown_pct - arm_baseline.max_fold_drawdown_pct,
            "d_annualized_turnover": arm_funded.annualized_turnover
            - arm_baseline.annualized_turnover,
            "d_total_fills": arm_funded.total_fills - arm_baseline.total_fills,
            "d_win_rate": arm_funded.mean_win_rate - arm_baseline.mean_win_rate,
            "d_expectancy_per_trade": arm_funded.mean_expectancy_per_trade
            - arm_baseline.mean_expectancy_per_trade,
            "d_auc_roc": ((arm_funded.mean_auc_roc or 0.0) - (arm_baseline.mean_auc_roc or 0.0)),
            "d_ece": ((arm_funded.mean_ece or 0.0) - (arm_baseline.mean_ece or 0.0)),
            "d_oracle_capture_ls": arm_funded.mean_oracle_capture_long_short
            - arm_baseline.mean_oracle_capture_long_short,
            "gate_transition": (
                "PASS→PASS"
                if arm_baseline.gate_passed and arm_funded.gate_passed
                else "PASS→FAIL"
                if arm_baseline.gate_passed and not arm_funded.gate_passed
                else "FAIL→PASS"
                if not arm_baseline.gate_passed and arm_funded.gate_passed
                else "FAIL→FAIL"
            ),
        },
    }


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
    parser.add_argument("--btc-funding", type=Path, default=REPO_ROOT / "BTCUSDT_funding_5y.csv")
    parser.add_argument("--eth-funding", type=Path, default=REPO_ROOT / "ETHUSDT_funding_5y.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "eval_reports" / "ablations" / "funding_ablation_v1.json",
    )
    parser.add_argument("--lockfile", type=Path, default=REPO_ROOT / "uv.lock")
    args = parser.parse_args()

    git_sha = current_git_sha(cwd=REPO_ROOT)
    lock_sha = lockfile_sha(args.lockfile) if args.lockfile.exists() else "no-lockfile"

    result = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "code_git_sha": git_sha,
        "lockfile_sha": lock_sha,
        "arms": [
            run_symbol_ablation(
                spec_path=args.btc_spec,
                funding_csv_path=args.btc_funding,
                code_git_sha=git_sha,
                lockfile_sha_str=lock_sha,
                data_root=REPO_ROOT,
            ),
            run_symbol_ablation(
                spec_path=args.eth_spec,
                funding_csv_path=args.eth_funding,
                code_git_sha=git_sha,
                lockfile_sha_str=lock_sha,
                data_root=REPO_ROOT,
            ),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {args.out}")

    for arm in result["arms"]:
        _print_arm_summary(arm)


def _print_arm_summary(arm: dict[str, Any]) -> None:
    b = arm["baseline_arm"]
    f = arm["funded_arm"]
    d = arm["deltas"]
    b_gate = "PASS" if b["gate_passed"] else "FAIL"
    f_gate = "PASS" if f["gate_passed"] else "FAIL"
    print(f"\n=== {arm['symbol']} ({arm['spec_name']}) ===")
    print(
        f"  baseline: cas={b['mean_cost_adjusted_sharpe']:.3f}  "
        f"cons={b['consistency_score']:.3f}  "
        f"pct+={b['pct_folds_positive_cas']:.2f}  "
        f"dd={b['max_fold_drawdown_pct']:.2f}%  "
        f"tv={b['annualized_turnover']:.2f}  "
        f"fills={b['total_fills']}  gate={b_gate}"
    )
    print(
        f"  funded:   cas={f['mean_cost_adjusted_sharpe']:.3f}  "
        f"cons={f['consistency_score']:.3f}  "
        f"pct+={f['pct_folds_positive_cas']:.2f}  "
        f"dd={f['max_fold_drawdown_pct']:.2f}%  "
        f"tv={f['annualized_turnover']:.2f}  "
        f"fills={f['total_fills']}  gate={f_gate}"
    )
    print(
        f"  Δ:        Δcas={d['d_mean_cas']:+.3f}  "
        f"Δcons={d['d_consistency']:+.3f}  "
        f"Δpct+={d['d_pct_folds_positive']:+.2f}  "
        f"Δdd={d['d_max_dd']:+.2f}%  "
        f"Δtv={d['d_annualized_turnover']:+.2f}  "
        f"ΔAUC={d['d_auc_roc']:+.3f}  "
        f"gate: {d['gate_transition']}"
    )


if __name__ == "__main__":
    main()
