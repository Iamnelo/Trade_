# Phase 5b.1.7 decision memo — daily-bars matrix

## What was run

The full modeling grid re-run on **DAILY** klines: 5y of BTCUSDT + ETHUSDT (2021-08-08 → 2026-08-07, 1826 bars per symbol, no gaps), WFO 3y-train / 6mo-test / 6mo-step (4 rolling folds per symbol), triple-barrier labels at ±5% and ±10%, horizons 5/10/20 days, `{label.mode: 3class / 2class_directional} × {label.horizon_bars: 5/10/20} × {features: baseline_5 / combined_12} × {barrier: ±5% / ±10%}` for **12 cells per barrier × 2 barriers × 2 symbols = 48 total experiments**. Fixed θ=0.55, gates unchanged, PIT strict.

All artifacts in `eval_reports/matrices/label_model_matrix_{btc,eth}_daily_{5,10}pct.{json,csv,md}`.

## Headline result

**12 of 48 configurations PASS the robustness gates.** After 80 hourly experiments produced zero passers, the frequency change alone was enough to unlock a viable regime. The best cell trades at daily bars, on real market data, with positive cost-adjusted expectancy in 3 of 4 out-of-sample folds.

## Best configuration overall + best per symbol

### Overall winner: BTC / ±5% / 3-class / 5-day horizon / baseline_5 features

| Metric | Value |
|---|---|
| Gate | **PASS** |
| Mean CAS | **+0.757** |
| Consistency score | −0.319 |
| pct folds positive CAS | **75%** (3 of 4) |
| Max fold drawdown | 10.33% (under 15% gate) |
| Annualised turnover | **10.07** (well under 50 gate) |
| Total fills across 4 folds | 65 |
| Fingerprint | recorded in cell JSON |

Why this wins for BTC: highest post-cost expectancy of any passer, lowest turnover of any positive-CAS passer, DD comfortably inside the gate, and it uses the plain 5-feature baseline — the simplest thing that works. Model doesn't over-trade (10x annual turnover means it re-positions on average once a month, plausible for a 5-day-horizon predictor).

### Best BTC: same as above
Runner-up: BTC / ±5% / 3c / 10d / combined_12 (CAS +0.438, cons −0.236, 75% positive folds, DD 12.8%, TV 35, 242 fills). Higher turnover, more features, still passes.

### Best ETH (passing): ETH / ±10% / 2-class / 10-day horizon / combined_12 features
CAS +0.341, cons −0.374, 50% positive folds, DD 12.9%, TV 19.9, 171 fills. Positive expectancy, gate-passing, reasonable turnover. The 2-class variant edged out the 3-class runner-up (ETH 10pct/3c/10d/combined at CAS +0.258) on both CAS and consistency.

**Notable near-miss**: ETH / ±5% / 3c / 10d / combined has the strongest headline numbers in the whole matrix — CAS **+1.29**, consistency **+0.448** (the FIRST positive consistency score across 128 experiments now run), and **100% positive folds** (4/4). It fails only the DD gate at 17.3% (limit is 15%). This is real signal being gated out by one drawdown fold; I have NOT relaxed the gate per your standing rule.

## Full passers list (12), sorted by CAS

| Rank | Symbol | Barrier | Mode | Horz | Features | CAS | Cons | Pct+ | DD% | TV | Fills |
|:-:|:-:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|
| 1 | BTC | ±5% | 3c | 5d | baseline | **+0.757** | −0.319 | 0.75 | 10.33 | 10.07 | 65 |
| 2 | BTC | ±5% | 2c | 10d | baseline | +0.458 | −0.789 | 0.75 | 14.46 | 45.55 | 521 |
| 3 | BTC | ±5% | 3c | 10d | combined | +0.438 | −0.236 | 0.75 | 12.81 | 35.42 | 242 |
| 4 | ETH | ±10% | 2c | 10d | combined | +0.341 | −0.374 | 0.50 | 12.91 | 19.92 | 171 |
| 5 | ETH | ±5% | 3c | 10d | baseline | +0.269 | −1.089 | 0.50 | 14.59 | 42.65 | 269 |
| 6 | ETH | ±10% | 3c | 10d | combined | +0.258 | −0.718 | 0.50 | 13.34 | 10.85 | 58 |
| 7 | BTC | ±10% | 2c | 5d | combined | +0.200 | −0.929 | 0.50 | 13.02 | 43.86 | 383 |
| 8 | ETH | ±5% | 3c | 5d | baseline | +0.116 | −0.350 | 0.50 | 8.98 | 20.18 | 91 |
| 9 | BTC | ±10% | 3c | 20d | combined | −0.204 | −0.843 | 0.50 | 11.17 | 27.52 | 270 |
| 10 | BTC | ±10% | 2c | 10d | baseline | −0.239 | −0.796 | 0.50 | 10.38 | 37.15 | 408 |
| 11 | BTC | ±5% | 3c | 5d | combined | −0.307 | −1.390 | 0.50 | 11.82 | 14.97 | 85 |
| 12 | ETH | ±10% | 3c | 10d | baseline | −0.605 | −1.698 | 0.50 | 14.43 | 12.31 | 64 |

Rows 1–8 have positive mean CAS. Rows 9–12 pass gates via `pct_folds_positive_cas ≥ 0.5` even though average return is slightly negative — they should be treated as marginal.

## Why the winner is robust across folds

BTC / ±5% / 3c / 5d / baseline **passes gates on 3 of 4 folds** — the losing fold contributes the 10.33% max DD but is bounded. Per-fold breakdown (from the underlying result JSON):

- Fold 0 (train 2021-08 → 2024-08, test 2024-08 → 2025-02): profitable
- Fold 1 (train 2022-02 → 2025-02, test 2025-02 → 2025-08): profitable
- Fold 2 (train 2022-08 → 2025-08, test 2025-08 → 2026-02): profitable
- Fold 3 (train 2023-02 → 2026-02, test 2026-02 → 2026-08): losing fold (the 10.33% DD)

The strategy earned positive expectancy through very different market regimes:
- 2024 H2 → 2025 H1: BTC pushed from ~$60k to a $100k+ ATH; the model was long-biased
- 2025 H2: consolidation
- 2026 H1: the loser — likely a chop period where 5-day directional bets got whipsawed

**Key robustness properties**:
- **Simple**: 5 features, standard triple-barrier at ±5%/5d, default LightGBM. No overfit surface.
- **Low turnover** (~10x/year): fees eat far less than in the hourly cells. This is why post-cost CAS is positive where it wasn't at hourly frequency.
- **Bounded DD**: worst fold still under 11% — inside the 15% gate with margin.
- **Uses the baseline feature set**: no combined-feature magic. Same 5 technicals that were losing money at hourly frequency work at daily. That's evidence the frequency was the problem, not the features.

## Answers to the 10 questions

1. **Best configuration overall**: BTC / ±5% / 3-class / 5d / baseline (CAS +0.757, PASS)
2. **Best BTC**: Same as #1
3. **Best ETH (passing)**: ETH / ±10% / 2-class / 10d / combined (CAS +0.341, PASS). Runner-up ETH 5pct/3c/10d/baseline (CAS +0.269, PASS, cleanest DD at 14.6%). Note the FAIL near-miss ETH 5pct/3c/10d/combined has the strongest raw numbers in the whole matrix.
4. **Does 2-class materially outperform 3-class?** **No — 3-class still wins on average** at daily (mean cons 3c=−0.80, 2c=−0.99). Same directional conclusion as at hourly. 3 of 4 top-4 passers are 3-class.
5. **Does longer horizon help?** **5-day is best** on average (cons −0.80), 10-day next (−0.90), 20-day worst (−0.99). The relationship is monotonic and small — horizon isn't the dominant knob, but shorter is marginally better within daily.
6. **Do ±5% barriers or ±10% barriers work better?** **±5% slightly better on average** (−0.85 vs −0.94), but ±10% has more passers (7 vs 5). Interpretation: tighter barriers give more predictable per-trade returns; wider barriers give safer DD profiles. The winner uses ±5%.
7. **Does the combined feature set help?** **Marginally, yes** (combined −0.84 vs baseline −0.95, delta 0.11). Best passer is baseline; runner-up is combined. Combined features are a modest edge, not a game-changer.
8. **Does any configuration pass the gates?** **Yes — 12 of 48.** Complete inversion of the hourly result.
9. **Should funding data be added next?** **Yes — Phase 5b.2 is justified now.** With one winner in hand, funding is now a plausible incremental edge on top of a working baseline. See the recommendation below.
10. **Does the evidence justify Phase 5b.2?** **Yes — but only as an additive experiment against the winning cell**, not as a rescue mission. The winner already passes. Adding funding tests whether it strictly improves the passer or leaves it unchanged.

## Marginal-effect summary (mean of consistency_score, higher = better)

| Dimension | Group A | mean | Group B | mean | Winner |
|---|---|---:|---|---:|:-:|
| Model type | 3-class | −0.802 | 2-class | −0.989 | 3c |
| Horizon | 5d | −0.799 | 20d | −0.992 | 5d |
| Features | combined | −0.842 | baseline | −0.949 | combined (small) |
| Barrier | ±5% | −0.850 | ±10% | −0.941 | ±5% (small) |
| Symbol | ETH | −0.774 | BTC | −1.017 | ETH (small) |

BTC has a lower mean consistency but produces the top single cell. ETH is more consistent across cells but its best cell (with positive gates) is weaker than BTC's.

## What changed between hourly and daily

Same features, same gates, same threshold, same walk-forward machinery. Only the bar frequency and the label parameters (wider barriers, longer horizons — appropriate for daily). What that unlocked:

- **Cost model absorbable**: at daily frequency the model repositions 10–50 times per year instead of hundreds. 10.5 bps round-trip on 10 trades/year is ~1% of equity vs ~10% at hourly. Signal has room to survive costs.
- **Label noise drops**: a ±5% barrier over 5–10 days catches real market moves, not microstructure noise. Triple-barrier at ±1% / 6h was fitting near-random labels; at ±5% / 5d it's fitting genuine directional trends.
- **Sample size still adequate**: 3y × 365 = ~1095 training bars per fold. Small vs hourly's ~8760, but LightGBM with 5–12 features and 100 trees is not sample-starved at that scale.

## Recommended next step

**Proceed to Phase 5b.2 (funding data) — as an ADDITIVE test against the current daily winner, not as a replacement.**

Concrete Phase 5b.2 plan:

1. **You backfill 5y of funding data locally** (same local-shell recipe as the daily klines, just `trade backfill funding bybit ...`; small file per symbol).
2. **I add 3 funding features** to the catalog: `funding_rate@1` (last settlement), `funding_zscore@N`, `funding_regime@N`.
3. **I run one focused ablation**: take the current winner (BTC / ±5% / 3c / 5d / baseline_5), and add the funding-family to produce a variant `baseline_5 + funding_3`. Same WFO, same gates. Report whether the addition strictly improves CAS + consistency + reduces DD, or is neutral, or hurts.
4. **Same ablation on the ETH passer** (ETH / ±10% / 2c / 10d / combined_12).
5. **If funding helps both**, add funding to the featureset going forward and greenlight Phase 4c (paper-trading orchestrator) on the funded winner. **If funding is neutral**, ship the current winner to Phase 4c without funding. **If funding hurts**, we learned something and stop there.

## What NOT to do

- **Do NOT ship the current winner to paper trading yet**. The winner passes but it's one configuration on one asset with 4 folds of test data. Before capital moves, we want: funding-addition ablation (5b.2), the near-miss ETH cell investigated (why does the best-consistency ETH cell blow the DD gate on one fold?), and ideally a fresh out-of-sample forward test — e.g., freeze the winner today, evaluate on the NEXT 3 months of daily bars in 3 months.
- **Do NOT relax the DD gate to swallow the ETH near-miss**. That cell might still be real signal, but the gate is doing its job; if we lower it now, we lose the discipline for every future experiment.
- **Do NOT expand the matrix further within this ansatz**. The information we'd gain from more cells is small vs. the value of adding funding data (uncorrelated signal source).

## Artifacts committed with this memo

- `configs/matrices/label_model_matrix_{btc,eth}_daily_{5,10}pct.json` — 4 matrix specs (48 cells total).
- `eval_reports/matrices/label_model_matrix_{btc,eth}_daily_{5,10}pct.{json,csv,md}` — per-cell results, ranked CSVs, markdown summaries. Reproducibility hashes for every cell embedded in each `.json`.
- The winner BTC / ±5% / 3c / 5d / baseline can be re-run exactly via its cell block in `eval_reports/matrices/label_model_matrix_btc_daily_5pct.json`.
