"""Writers for the threshold-sweep result: CSV, markdown, plotly HTML.

Each writer takes a `ThresholdSweepResult` and emits one artifact. The
CSV is the source of truth (dense, machine-readable, one row per
(fold, threshold)); markdown is the human-narrative summary; HTML is
interactive exploration built on plotly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trade.research.sweep import ThresholdSweepResult


def write_cell_csv(result: ThresholdSweepResult, path: Path) -> None:
    """One row per (fold, threshold) cell — the dense trading-metrics table."""
    fieldnames = [
        "symbol",
        "fold_index",
        "threshold",
        "n_fills",
        "total_return_pct",
        "sharpe",
        "sortino",
        "cost_adjusted_sharpe",
        "max_drawdown_pct",
        "hit_rate",
        "turnover",
        "n_trades",
        "win_rate",
        "expectancy_per_trade",
        "profit_factor",
        "avg_win_pnl",
        "avg_loss_pnl",
        "largest_win_pnl",
        "largest_loss_pnl",
        "capture_ratio_vs_long_only",
        "capture_ratio_vs_long_short",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in result.per_cell:
            w.writerow(
                {
                    "symbol": result.base_spec.data.symbol,
                    "fold_index": c.fold_index,
                    "threshold": c.threshold,
                    "n_fills": c.n_fills,
                    "total_return_pct": c.total_return_pct,
                    "sharpe": c.sharpe,
                    "sortino": c.sortino,
                    "cost_adjusted_sharpe": c.cost_adjusted_sharpe,
                    "max_drawdown_pct": c.max_drawdown_pct,
                    "hit_rate": c.hit_rate,
                    "turnover": c.turnover,
                    "n_trades": c.trade_metrics.n_trades,
                    "win_rate": c.trade_metrics.win_rate,
                    "expectancy_per_trade": c.trade_metrics.expectancy_per_trade,
                    "profit_factor": (
                        ""
                        if c.trade_metrics.profit_factor == float("inf")
                        else c.trade_metrics.profit_factor
                    ),
                    "avg_win_pnl": c.trade_metrics.avg_win_pnl,
                    "avg_loss_pnl": c.trade_metrics.avg_loss_pnl,
                    "largest_win_pnl": c.trade_metrics.largest_win_pnl,
                    "largest_loss_pnl": c.trade_metrics.largest_loss_pnl,
                    "capture_ratio_vs_long_only": c.oracle_capture.capture_ratio_vs_long_only,
                    "capture_ratio_vs_long_short": c.oracle_capture.capture_ratio_vs_long_short,
                }
            )


def write_threshold_csv(result: ThresholdSweepResult, path: Path) -> None:
    """One row per threshold — the side-by-side aggregate table."""
    fieldnames = [
        "symbol",
        "threshold",
        "n_folds_with_trades",
        "mean_cost_adjusted_sharpe",
        "median_cost_adjusted_sharpe",
        "min_cost_adjusted_sharpe",
        "std_cost_adjusted_sharpe",
        "pct_folds_positive_cas",
        "max_fold_drawdown_pct",
        "mean_hit_rate",
        "annualized_turnover",
        "total_fills",
        "consistency_score",
        "gate_passed",
        "gate_reasons_failed",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in result.per_threshold:
            rb = t.robustness
            w.writerow(
                {
                    "symbol": result.base_spec.data.symbol,
                    "threshold": t.threshold,
                    "n_folds_with_trades": rb.n_folds_with_trades,
                    "mean_cost_adjusted_sharpe": rb.mean_cost_adjusted_sharpe,
                    "median_cost_adjusted_sharpe": rb.median_cost_adjusted_sharpe,
                    "min_cost_adjusted_sharpe": rb.min_cost_adjusted_sharpe,
                    "std_cost_adjusted_sharpe": rb.std_cost_adjusted_sharpe,
                    "pct_folds_positive_cas": rb.pct_folds_positive_cas,
                    "max_fold_drawdown_pct": rb.max_fold_drawdown_pct,
                    "mean_hit_rate": rb.mean_hit_rate,
                    "annualized_turnover": rb.annualized_turnover,
                    "total_fills": rb.total_fills,
                    "consistency_score": rb.consistency_score,
                    "gate_passed": t.gate.passed,
                    "gate_reasons_failed": " ; ".join(t.gate.reasons_failed),
                }
            )


def write_classifier_csv(result: ThresholdSweepResult, path: Path) -> None:
    """One row per fold — the classifier-only diagnostics table."""
    fieldnames = [
        "symbol",
        "fold_index",
        "reproducibility_hash",
        "n_test_samples",
        "auc_roc_macro",
        "auc_pr_macro",
        "ece_macro",
        "support_down",
        "support_flat",
        "support_up",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for f_row in result.per_fold_classifier:
            d = f_row.diagnostics
            support_down, support_flat, support_up = d.class_support
            w.writerow(
                {
                    "symbol": result.base_spec.data.symbol,
                    "fold_index": f_row.fold_index,
                    "reproducibility_hash": f_row.reproducibility_hash,
                    "n_test_samples": d.n_test_samples,
                    "auc_roc_macro": "" if d.auc_roc_macro is None else d.auc_roc_macro,
                    "auc_pr_macro": "" if d.auc_pr_macro is None else d.auc_pr_macro,
                    "ece_macro": "" if d.ece_macro is None else d.ece_macro,
                    "support_down": support_down,
                    "support_flat": support_flat,
                    "support_up": support_up,
                }
            )


def _fmt_or_dash(v: float | None, fmt: str = "{:.3f}") -> str:
    return "—" if v is None else fmt.format(v)


def _md_header(result: ThresholdSweepResult) -> list[str]:
    spec = result.base_spec
    return [
        f"# Threshold sweep — {spec.name}",
        "",
        f"- **Symbol**: `{spec.data.symbol}`  ",
        f"- **Features**: `{', '.join(spec.features)}`  ",
        (
            f"- **WFO**: train={spec.wfo.train_bars}h  test={spec.wfo.test_bars}h  "
            f"step={spec.wfo.step_bars}h  expanding={spec.wfo.expanding}  "
        ),
        (
            f"- **Label**: triple-barrier, horizon={spec.label.horizon_bars}, "
            f"±{spec.label.up_pct:.2%} barriers  "
        ),
        f"- **Bars used**: {result.n_bars:,}  ",
        f"- **Generated**: `{result.generated_at}`  ",
        f"- **git**: `{result.code_git_sha[:12]}`  lock: `{result.lockfile_sha[:12]}`",
        "",
        "## Recommendation",
        "",
        f"> {result.recommendation_notes}",
        "",
    ]


def _md_per_threshold(result: ThresholdSweepResult) -> list[str]:
    out: list[str] = [
        "## Per-threshold aggregate",
        "",
        "| threshold | mean_cas | pct_pos | max_dd% | ann_turnover | consistency | gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for t in result.per_threshold:
        rb = t.robustness
        star = " ⭐" if result.recommended_threshold == t.threshold else ""
        out.append(
            f"| {t.threshold:.2f}{star} | {rb.mean_cost_adjusted_sharpe:.3f} "
            f"| {rb.pct_folds_positive_cas:.2f} | {rb.max_fold_drawdown_pct:.2f} "
            f"| {rb.annualized_turnover:.2f} | {rb.consistency_score:.3f} "
            f"| {'PASS' if t.gate.passed else 'FAIL'} |"
        )
    out.append("")
    fails = [t for t in result.per_threshold if not t.gate.passed]
    if fails:
        out.append("### Gate-failure detail")
        for t in fails:
            reasons = "; ".join(t.gate.reasons_failed) or "(no explicit reason)"
            out.append(f"- `θ={t.threshold:.2f}`: {reasons}")
        out.append("")
    return out


def _md_classifier(result: ThresholdSweepResult) -> list[str]:
    out: list[str] = [
        "## Per-fold classifier diagnostics",
        "",
        (
            "These describe the trained model's predictive quality, independent "
            "of any trading threshold."
        ),
        "",
        "| fold | n_test | AUC-ROC | AUC-PR | ECE | support (down/flat/up) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for f_row in result.per_fold_classifier:
        d = f_row.diagnostics
        out.append(
            f"| {f_row.fold_index} | {d.n_test_samples} "
            f"| {_fmt_or_dash(d.auc_roc_macro)} "
            f"| {_fmt_or_dash(d.auc_pr_macro)} "
            f"| {_fmt_or_dash(d.ece_macro)} "
            f"| {d.class_support[0]} / {d.class_support[1]} / {d.class_support[2]} |"
        )
    out.append("")
    return out


def write_markdown_summary(result: ThresholdSweepResult, path: Path) -> None:
    lines: list[str] = []
    lines.extend(_md_header(result))
    lines.extend(_md_per_threshold(result))
    lines.extend(_md_classifier(result))

    lines.append("## Trade-level metrics at the recommended threshold")
    lines.append("")
    if result.recommended_threshold is not None:
        chosen = [c for c in result.per_cell if c.threshold == result.recommended_threshold]
        if chosen:
            lines.append(
                "| fold | n_trades | win_rate | expectancy | profit_factor | oracle_cap_ls |"
            )
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
            for c in chosen:
                tm = c.trade_metrics
                pf = "∞" if tm.profit_factor == float("inf") else f"{tm.profit_factor:.2f}"
                lines.append(
                    f"| {c.fold_index} | {tm.n_trades} | {tm.win_rate:.2%} "
                    f"| {tm.expectancy_per_trade:.2f} | {pf} "
                    f"| {c.oracle_capture.capture_ratio_vs_long_short:.2f} |"
                )
    else:
        lines.append("_No gate-passing threshold to report on._")

    path.write_text("\n".join(lines))


def write_html_report(result: ThresholdSweepResult, path: Path) -> None:
    """Interactive plotly HTML: per-threshold aggregate + fold matrix."""
    thresholds = list(result.thresholds)
    per_threshold = {t.threshold: t for t in result.per_threshold}

    cas = [per_threshold[t].robustness.mean_cost_adjusted_sharpe for t in thresholds]
    cons = [per_threshold[t].robustness.consistency_score for t in thresholds]
    dd = [per_threshold[t].robustness.max_fold_drawdown_pct for t in thresholds]
    tv = [per_threshold[t].robustness.annualized_turnover for t in thresholds]
    passes = [per_threshold[t].gate.passed for t in thresholds]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Mean cost-adjusted Sharpe vs threshold",
            "Consistency score vs threshold",
            "Max fold drawdown % vs threshold",
            "Annualised turnover vs threshold",
        ),
    )
    colors = ["#2ca02c" if p else "#d62728" for p in passes]
    fig.add_trace(
        go.Bar(x=thresholds, y=cas, marker_color=colors, name="mean CAS", showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=thresholds, y=cons, marker_color=colors, name="consistency", showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=thresholds, y=dd, marker_color=colors, name="max DD %", showlegend=False),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=thresholds, y=tv, marker_color=colors, name="ann TV", showlegend=False),
        row=2,
        col=2,
    )
    rec_str = (
        f"threshold={result.recommended_threshold:.2f}"
        if result.recommended_threshold is not None
        else "NO passer"
    )
    fig.update_layout(
        title_text=(
            f"Threshold sweep — {result.base_spec.name}<br>"
            f"<sub>Green = gate pass, red = gate fail. Recommendation: {rec_str}</sub>"
        ),
        height=800,
        showlegend=False,
    )
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
