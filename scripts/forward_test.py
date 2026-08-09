"""Push-button forward test for the frozen daily winners.

Second half of the forward-test protocol. Loads the artifacts frozen by
`scripts/freeze_winners.py` and evaluates each winner ONCE, with NO
retraining, on genuinely-unseen bars that post-date the freeze cutoff.

Because the frozen model is loaded from disk and only *replayed*, this is a
strict out-of-sample test: the model provably could not have seen any bar in
the forward window (they did not exist when it was trained/frozen).

Forward window construction
---------------------------
The historical CSV (ends at the cutoff) is stitched to a forward CSV (bars
after the cutoff) so the strategy has enough warmup history to compute its
longest-lookback feature on the very first forward decision. Only bars with
event_time strictly greater than the manifest cutoff count as the test slice.

Gates (hard-coded — NOT tunable from the CLI, on purpose)
---------------------------------------------------------
A symbol PASSES the forward test iff ALL hold on its forward window:
  1. cost-adjusted Sharpe (CAS) > 0
  2. max drawdown <= 15%
  3. >= 5 fills
  4. calibration stable: forward ECE <= reference ECE * 2.0
Verdicts are per-symbol and independent — BTC can pass while ETH fails.

Passing is permission to begin paper-trading validation, NOT proof of a
profitable system: three months of daily bars is a small sample. Failing is
NOT a cue to tune — the report emits a failure-attribution block so a human
can decide whether the cause is model degradation, regime change, small
sample, costs, or calibration drift.

Run once fresh data exists:
    uv run python scripts/forward_test.py \
        --btc-forward BTCUSDT_D_forward.csv \
        --eth-forward ETHUSDT_D_forward.csv

Smoke-test the plumbing against committed data (NOT a real result):
    uv run python scripts/forward_test.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade.data.schemas import KlineRecord
from trade.features.catalog import build_features
from trade.metrics.trade_metrics import compute_trade_metrics
from trade.model.persistence import load_any_training_artifacts
from trade.mre.types import BacktestConfig
from trade.research.diagnostics import compute_classifier_diagnostics, score_holdout
from trade.research.experiment import ExperimentSpec
from trade.research.runner import load_klines_csv
from trade.wfo.model_runner import TrainedFold, backtest_trained_fold
from trade.wfo.splitter import Split

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_DIR = REPO_ROOT / "artifacts" / "frozen"

# Forward-window gate thresholds. Deliberately hard-coded so a forward test
# cannot be quietly loosened to manufacture a pass.
GATE_MIN_CAS = 0.0
GATE_MAX_DD_PCT = 15.0
GATE_MIN_FILLS = 5
CALIBRATION_TOLERANCE = 2.0  # forward ECE may be at most 2x the in-sample ref
REFERENCE_WINDOW_BARS = 180  # pre-cutoff tail used as the in-sample calib ref


def _parse_cutoff(entry: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(entry["cutoff_event_time"])


def _stitch(historical: Sequence[KlineRecord], forward: Sequence[KlineRecord]) -> list[KlineRecord]:
    """Concatenate + dedup by event_time (forward wins ties), then sort."""
    by_time: dict[datetime, KlineRecord] = {b.event_time: b for b in historical}
    for b in forward:
        by_time[b.event_time] = b
    return sorted(by_time.values(), key=lambda b: b.event_time)


def _first_index_after(bars: Sequence[KlineRecord], cutoff: datetime) -> int:
    for i, b in enumerate(bars):
        if b.event_time > cutoff:
            return i
    return len(bars)


def _gate(rep: Any, fwd_clf: Any, ref_ece: float | None) -> tuple[list[str], bool]:
    """Apply the four hard-coded forward gates. Returns (reasons_failed, calibration_ok)."""
    reasons: list[str] = []
    if not rep.cost_adjusted_sharpe > GATE_MIN_CAS:
        reasons.append(f"CAS {rep.cost_adjusted_sharpe:.3f} <= {GATE_MIN_CAS}")
    if rep.max_drawdown_pct > GATE_MAX_DD_PCT:
        reasons.append(f"max_dd {rep.max_drawdown_pct:.2f}% > {GATE_MAX_DD_PCT}%")
    if rep.n_fills < GATE_MIN_FILLS:
        reasons.append(f"fills {rep.n_fills} < {GATE_MIN_FILLS}")
    calibration_ok = True
    if fwd_clf.ece_macro is None:
        reasons.append("forward ECE unavailable (too few labelled forward rows)")
        calibration_ok = False
    elif ref_ece is not None and fwd_clf.ece_macro > ref_ece * CALIBRATION_TOLERANCE:
        reasons.append(
            f"calibration drift: forward ECE {fwd_clf.ece_macro:.3f} > "
            f"{CALIBRATION_TOLERANCE}x reference {ref_ece:.3f}"
        )
        calibration_ok = False
    return reasons, calibration_ok


def _attribute(
    rep: Any, tm: Any, fwd_clf: Any, ref_auc: float | None, *, calibration_ok: bool
) -> list[str]:
    """Heuristic failure-cause hints. Final judgment is a human's."""
    out: list[str] = []
    if rep.n_fills < GATE_MIN_FILLS:
        out.append(
            "SMALL SAMPLE / INACTIVITY: too few trade decisions crossed the "
            "threshold in the forward window; extend the window before judging."
        )
    if ref_auc is not None and fwd_clf.auc_roc_macro is not None:
        d_auc = fwd_clf.auc_roc_macro - ref_auc
        if d_auc < -0.05:
            out.append(
                f"MODEL DEGRADATION / REGIME CHANGE: forward AUC "
                f"{fwd_clf.auc_roc_macro:.3f} fell {-d_auc:.3f} below reference "
                f"{ref_auc:.3f} — the classifier's ranking broke down out-of-sample."
            )
    if not calibration_ok and fwd_clf.ece_macro is not None:
        out.append(
            "CALIBRATION DRIFT: probabilities are miscalibrated out-of-sample; "
            "the isotonic map fit in-sample no longer holds."
        )
    if tm.n_trades > 0 and tm.expectancy_per_trade < 0 and rep.turnover > 0:
        out.append(
            f"COSTS: negative expectancy ({tm.expectancy_per_trade:.2f}/trade) with "
            f"turnover {rep.turnover:.2f} — fees/slippage are eating the edge."
        )
    if not out:
        out.append("UNCLEAR: no single dominant cause; inspect the equity curve and fills.")
    return out


def evaluate_forward(
    *,
    entry: dict[str, Any],
    stitched: Sequence[KlineRecord],
    cutoff: datetime,
) -> dict[str, Any]:
    """Replay one frozen winner over the post-cutoff slice of `stitched`."""
    symbol = entry["symbol"]
    features = build_features(list(entry["feature_ids"]))
    max_lookback = max(f.spec.lookback_bars for f in features)
    artifacts = load_any_training_artifacts(REPO_ROOT / entry["artifacts_dir"])

    # Verify the frozen artifact is the one the manifest claims.
    if artifacts.reproducibility_hash != entry["reproducibility_hash"]:
        raise ValueError(
            f"{symbol}: frozen artifact hash {artifacts.reproducibility_hash[:12]} "
            f"!= manifest hash {entry['reproducibility_hash'][:12]}"
        )

    test_start = _first_index_after(stitched, cutoff)
    test_end = len(stitched)
    n_forward = test_end - test_start
    if n_forward <= 0:
        raise ValueError(
            f"{symbol}: no bars after cutoff {cutoff.isoformat()} — forward test "
            f"cannot run yet (need fresh bars)."
        )

    warmup_start = max(0, test_start - max_lookback)
    trained = TrainedFold(
        split=Split(train_start=0, train_end=test_start, test_start=test_start, test_end=test_end),
        artifacts=artifacts,
        features=tuple(features),
        sorted_bars=tuple(stitched),
        symbol=symbol,
        interval=entry["interval"],
        warmup_start_idx=warmup_start,
        max_lookback=max_lookback,
    )
    config = BacktestConfig(
        initial_equity=entry["initial_equity"],
        fee_bps=entry["fee_bps"],
        slippage_bps=entry["slippage_bps"],
    )
    fold = backtest_trained_fold(
        trained,
        config=config,
        bars_per_year=entry["bars_per_year"],
        confidence_threshold=entry["confidence_threshold"],
        notional_fraction=entry["notional_fraction"],
        allow_short=entry["allow_short"],
        cost_bps_per_side=entry["fee_bps"],
    )
    rep = fold.report

    # Classifier diagnostics on the forward window (calibration + ranking).
    fwd_probs, fwd_true = score_holdout(
        sorted_bars=stitched,
        symbol=symbol,
        features=features,
        model=artifacts.model,
        calibrator=artifacts.calibrator,
        test_start_idx=test_start,
        test_end_idx=test_end,
        label_horizon_bars=entry["label_horizon_bars"],
        label_up_pct=entry["label_up_pct"],
        label_down_pct=entry["label_down_pct"],
        label_mode=entry["label_mode"],
    )
    fwd_clf = compute_classifier_diagnostics(probs=fwd_probs, y_true_int=fwd_true)

    # In-sample reference calibration on the pre-cutoff tail (bars the model
    # DID train on) — the yardstick for "calibration remains stable".
    ref_start = max(0, test_start - REFERENCE_WINDOW_BARS)
    ref_auc: float | None = None
    ref_ece: float | None = None
    if test_start - ref_start >= 10:
        ref_probs, ref_true = score_holdout(
            sorted_bars=stitched,
            symbol=symbol,
            features=features,
            model=artifacts.model,
            calibrator=artifacts.calibrator,
            test_start_idx=ref_start,
            test_end_idx=test_start,
            label_horizon_bars=entry["label_horizon_bars"],
            label_up_pct=entry["label_up_pct"],
            label_down_pct=entry["label_down_pct"],
            label_mode=entry["label_mode"],
        )
        ref_clf = compute_classifier_diagnostics(probs=ref_probs, y_true_int=ref_true)
        ref_auc = ref_clf.auc_roc_macro
        ref_ece = ref_clf.ece_macro

    tm = compute_trade_metrics(fold.backtest.fills)
    reasons, calibration_ok = _gate(rep, fwd_clf, ref_ece)
    passed = not reasons
    attribution = (
        [] if passed else _attribute(rep, tm, fwd_clf, ref_auc, calibration_ok=calibration_ok)
    )

    return {
        "symbol": symbol,
        "spec_name": entry["spec_name"],
        "reproducibility_hash": entry["reproducibility_hash"],
        "forward_window": {
            "n_bars": n_forward,
            "start": stitched[test_start].event_time.isoformat(),
            "end": stitched[test_end - 1].event_time.isoformat(),
        },
        "metrics": {
            "cost_adjusted_sharpe": rep.cost_adjusted_sharpe,
            "sharpe": rep.sharpe,
            "sortino": rep.sortino,
            "total_return_pct": rep.total_return_pct,
            "max_drawdown_pct": rep.max_drawdown_pct,
            "n_fills": rep.n_fills,
            "hit_rate": rep.hit_rate,
            "turnover": rep.turnover,
            "final_equity": rep.final_equity,
            "n_trades": tm.n_trades,
            "win_rate": tm.win_rate if tm.n_trades else None,
            "expectancy_per_trade": tm.expectancy_per_trade if tm.n_trades else None,
            "profit_factor": (
                tm.profit_factor if tm.n_trades and tm.profit_factor != float("inf") else None
            ),
            "forward_auc_roc": fwd_clf.auc_roc_macro,
            "forward_ece": fwd_clf.ece_macro,
            "reference_auc_roc": ref_auc,
            "reference_ece": ref_ece,
        },
        "gate": {
            "passed": passed,
            "reasons_failed": reasons,
            "thresholds": {
                "min_cas": GATE_MIN_CAS,
                "max_dd_pct": GATE_MAX_DD_PCT,
                "min_fills": GATE_MIN_FILLS,
                "calibration_tolerance_x": CALIBRATION_TOLERANCE,
            },
        },
        "failure_attribution": attribution,
    }


def _load_forward_csv(path: Path | None, *, symbol: str, interval: str) -> list[KlineRecord]:
    if path is None or not path.exists():
        return []
    return load_klines_csv(path, symbol=symbol, interval=interval)


def run(
    *,
    manifest_path: Path,
    forward_csvs: dict[str, Path | None],
    self_test: bool,
    out_path: Path,
) -> int:
    manifest = json.loads(manifest_path.read_text())
    results: list[dict[str, Any]] = []
    blocked: list[str] = []

    for entry in manifest["winners"]:
        symbol = entry["symbol"]
        cutoff = _parse_cutoff(entry)
        historical = load_klines_csv(
            REPO_ROOT / _spec_csv_path(entry),
            symbol=symbol,
            interval=entry["interval"],
        )
        if self_test:
            # Carve the last REFERENCE_WINDOW_BARS of committed history as a
            # STAND-IN forward window. These bars were seen during selection
            # AND training, so the numbers are meaningless as a result — this
            # only proves the harness computes every metric end-to-end.
            if len(historical) <= REFERENCE_WINDOW_BARS + 30:
                blocked.append(f"{symbol}: history too short for self-test")
                continue
            split_at = len(historical) - REFERENCE_WINDOW_BARS
            cutoff = historical[split_at - 1].event_time
            stitched = historical
        else:
            forward = _load_forward_csv(
                forward_csvs.get(symbol), symbol=symbol, interval=entry["interval"]
            )
            forward = [b for b in forward if b.event_time > cutoff]
            if not forward:
                blocked.append(
                    f"{symbol}: no forward bars after {cutoff.isoformat()} "
                    f"(supply --{symbol[:3].lower()}-forward with post-cutoff daily bars)"
                )
                continue
            stitched = _stitch(historical, forward)

        results.append(evaluate_forward(entry=entry, stitched=stitched, cutoff=cutoff))

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "self_test": self_test,
        "code_git_sha_at_freeze": manifest.get("code_git_sha"),
        "results": results,
        "blocked": blocked,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    _print_report(report)
    if blocked and not results:
        return 2  # nothing ran — forward data not available yet
    return 0


def _spec_csv_path(entry: dict[str, Any]) -> str:
    """Recover the historical CSV path from the frozen spec."""
    spec = ExperimentSpec.from_json((REPO_ROOT / entry["spec_path"]).read_text())
    return spec.data.csv_path


def _print_report(report: dict[str, Any]) -> None:
    if report["self_test"]:
        print("=" * 72)
        print("SELF-TEST MODE — stand-in window is IN-SAMPLE. Numbers are NOT a")
        print("forward-test result. This only proves the harness runs end-to-end.")
        print("=" * 72)
    for r in report["results"]:
        m = r["metrics"]
        verdict = "PASS" if r["gate"]["passed"] else "FAIL"
        fw = r["forward_window"]
        print(f"\n=== {r['symbol']} ({r['spec_name']}) — {verdict} ===")
        print(f"  window: {fw['n_bars']} bars  {fw['start']} -> {fw['end']}")
        print(
            f"  CAS={m['cost_adjusted_sharpe']:.3f}  "
            f"maxDD={m['max_drawdown_pct']:.2f}%  fills={m['n_fills']}  "
            f"ret={m['total_return_pct']:.2f}%  hit={m['hit_rate']:.2%}"
        )
        auc = m["forward_auc_roc"]
        ece = m["forward_ece"]
        print(
            f"  forward AUC={auc if auc is None else round(auc, 3)}  "
            f"forward ECE={ece if ece is None else round(ece, 3)}  "
            f"(ref AUC={m['reference_auc_roc']}, ref ECE={m['reference_ece']})"
        )
        if not r["gate"]["passed"]:
            print(f"  FAILED: {'; '.join(r['gate']['reasons_failed'])}")
            for a in r["failure_attribution"]:
                print(f"    - {a}")
    for b in report["blocked"]:
        print(f"\n[blocked] {b}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=FROZEN_DIR / "freeze_manifest.json")
    parser.add_argument("--btc-forward", type=Path, default=REPO_ROOT / "BTCUSDT_D_forward.csv")
    parser.add_argument("--eth-forward", type=Path, default=REPO_ROOT / "ETHUSDT_D_forward.csv")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Smoke-test the harness on committed (in-sample) bars. NOT a real result.",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "eval_reports" / "forward" / "forward_test_v1.json"
    )
    args = parser.parse_args()

    code = run(
        manifest_path=args.manifest,
        forward_csvs={"BTCUSDT": args.btc_forward, "ETHUSDT": args.eth_forward},
        self_test=args.self_test,
        out_path=args.out,
    )
    print(f"\nwrote {args.out.relative_to(REPO_ROOT)}")
    if code == 2:
        print(
            "\nNo forward data available yet. Freeze is in place; re-run this "
            "once >=3 months of post-cutoff daily bars exist."
        )
    sys.exit(code)


if __name__ == "__main__":
    main()
