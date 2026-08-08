# Phase 5b.1.6 decision memo

## What was run

The 16-cell modeling grid per user spec: `{label.mode: 3class / 2class_directional} × {label.horizon_bars: 6 / 24} × {label.up_pct = down_pct: 0.01 / 0.02} × {features: baseline_5 / combined_12}` — evaluated independently for BTCUSDT and ETHUSDT. Total 32 experiments over 2y hourly klines (2024-08-06 → 2026-08-05), 6 rolling walk-forward folds each. Fixed threshold θ = 0.55 for every cell (no test-fold leakage into threshold selection). Baseline feature set = 5 price-only technicals; combined feature set = baseline + 4 time-cyclic + skew/kurtosis + vol_regime = 12 features.

Reproducibility hashes, per-cell JSON, per-cell ranked CSV, and markdown summaries all live in `eval_reports/matrices/label_model_matrix_{btc,eth}_{1pct,2pct}.*`. Any single cell can be re-materialised exactly from its `spec` block in the JSON.

## Headline result

**0 of 32 configurations pass the robustness gates.** Fixing labels, changing model type, or adding features moved individual metrics around but none produced a robust economic edge after costs.

## Answers to the 10 questions

### 1. Best configuration overall
**BTC / ±2% barriers / 3-class / 6h horizon / baseline features.** Consistency −1.246, CAS −0.51, 0% positive folds, 2 fills total across 6 folds. This is the least-broken cell, not a viable strategy — its "safety" comes from trading only twice in 6 months.

### 2. Best BTC configuration
Same as above. Runner-up: BTC / ±2% / 3-class / 6h / combined features (cons −1.579, cas −0.65, 3 fills). Both essentially do nothing.

### 3. Best ETH configuration
**ETH / ±2% / 2-class / 6h / baseline features.** Consistency −1.426, CAS −5.11, 0% positive folds, **max DD 20.3% and 818x annualized turnover**. This is "best" only in that variance across folds is a bit tighter; it is aggressively unprofitable. Runner-up: ETH / ±2% / 3-class / 24h / combined (cons −1.929, cas −1.62, 17% positive folds, 5% DD, 215x turnover). The runner-up is more useful evidence — moderate horizon + wider barriers gets a fold or two to positive CAS with less carnage.

### 4. Does 2-class materially outperform 3-class?
**No.** Mean consistency: 2c = −3.24, 3c = −3.01 → **3-class is marginally BETTER on average.** Best cells are close (2c best −1.43 vs 3c best −1.25) but 2-class does not dominate at either the mean or the top. The hypothesis "dropping the flat class lets the model be more confident on decisive moves" is not supported by the data. Plausible reason: dropping flat rows shrinks the training set by 40–70% depending on barrier width, and the resulting binary classifier over-trades because it always has a preferred direction.

### 5. Does 24h materially outperform 6h?
**No — 24h is WORSE.** Mean consistency: 24h = −3.53, 6h = −2.72 → 6h wins by ~0.8. Longer horizons introduce more label noise (many bars have both barriers touched over 24h, forcing them to "ambiguous/flat"), reducing effective training signal. The hypothesis is rejected.

### 6. Do ±2% barriers materially improve robustness?
**Marginally, within noise.** Mean consistency: ±2% = −3.05, ±1% = −3.20 (delta 0.15). ±2%'s best cell (−1.246) beats ±1%'s best (−1.940). Directionally the right sign but not a decisive win.

### 7. Does the combined feature set still help after changing labels?
**Near-wash.** Mean consistency: combined = −3.08, baseline = −3.16 (delta 0.08). Big feature interactions with mode + horizon: on some cells combined helps meaningfully (BTC ±2% 3c 24h combined has 17% positive folds vs baseline's 0%); on others it hurts (BTC ±1% 2c 24h combined does much worse). So the combined feature stack is not a reliable improvement across all modeling choices — its edge from Phase 5b.1 was specific to the original 3-class / 6h / ±1% setting.

### 8. Does any configuration pass the existing robustness gates?
**No. 0 of 32.** Nothing crosses `pct_folds_positive_cas ≥ 0.5`. Half the cells never even trade in a majority of folds, and the aggressive cells that do trade have unacceptable drawdowns (11–23%) and turnover (200–900x annually).

### 9. Should funding data be added next?
**Not yet — recommend deferring Phase 5b.2.** Reasoning: the current model has AUC-ROC 0.58–0.66 (real but small signal) and the strategy converts none of it into positive expectancy at any of the 32 cells sampled. Adding one more feature class is unlikely to fix a problem that survives changing model type, horizon, barriers, and 6 additional features simultaneously. Funding is a real signal in crypto perps but at these hourly horizons the funding rate itself moves too slowly (settles every 8h) to be the missing piece for hour-by-hour direction. The higher-leverage next experiment is elsewhere.

### 10. Does the evidence now justify moving toward Phase 5b.2?
**No.** The convergent negative evidence from 32 experiments — spanning every modeling knob the memo asked us to test — indicates the problem is not "one more feature away from working". The problem is fundamental: **hourly directional prediction on BTC/ETH perps with LightGBM + triple-barrier at ±1-2% over 6-24h is close to a fair coin after costs**. Fees + slippage at 10.5 bps round-trip are large relative to any move the model can consistently predict at these horizons.

## What the results indicate about the current modeling approach

Three consistent patterns across 80 experiments now run (24 baseline sweeps + 24 combined/regime sweeps + 32 matrix cells):

1. **Classifier quality is real but small.** AUC-ROC lands in 0.55–0.68 across every configuration. That's about 5–15% better than random on a 3-class problem where the flat class dominates. It's genuinely there.
2. **The signal disappears through the decision layer.** No threshold, feature set, model type, horizon, or barrier width converts that classifier edge into consistent post-cost expectancy. Turnover is the tell: high-turnover cells lose to costs; low-turnover cells rarely trade at all.
3. **We are trying to predict something the market prices out at this frequency.** Hourly directional moves on liquid crypto perps are dominated by market-maker inventory effects and news noise; the deterministic components that LightGBM can extract from OHLC/volume are tiny relative to costs.

This is what the platform is DESIGNED to reveal — the WFO + gates architecture is doing its job. It's showing us that a naive-directional-classifier-on-hourly-OHLC ansatz doesn't work on this asset class, and no amount of hyperparameter tuning within that ansatz will make it work.

## Recommended next highest-value experiment

**Move to daily bars — same platform, same matrix, different frequency.** Cheapest way to test the "wrong horizon" hypothesis before committing to funding data or a strategy pivot.

Concrete proposal (Phase 5b.1.7):

1. Local backfill: 5y of daily BTC + ETH klines (~1.8k bars per symbol, tiny — one CSV each, ~2 min of local time).
2. Adapt WFO schedule: train 3 years / test 6 months / step 6 months → ~4 folds. `bars_per_year` becomes 365.
3. Adapt labels: triple-barrier at ±5% / ±10% over 5–20 day horizons.
4. Re-run the same 16-cell matrix, same fixed threshold, same gates.

If the daily matrix produces at least one gate-passing configuration, we have a validated frequency and can then decide whether to add funding (Phase 5b.2) as an additive edge or ship what we have to paper trading.

If the daily matrix ALSO produces zero passers, the falsification is complete: directional prediction of BTC/ETH from OHLC alone is not a viable V1 for this platform. At that point the honest next moves are:

- **Change strategy type**: funding-arbitrage (short expensive funding, long cheap funding) — a genuinely different market inefficiency; requires funding backfill regardless.
- **Change data**: add cross-asset (SPY / DXY / TLT via daily), on-chain (Glassnode), order-book depth (needs live capture).
- **Change objective**: from "predict return direction" to "predict conditional variance" (short-vol strategies) or "predict basis" (perp-vs-spot arbitrage).

## What NOT to do next

- Do not run more matrix cells within the current hourly ansatz. Marginal information per experiment is now very low; we already know the answer to "does another knob within this framework fix it".
- Do not tune the threshold per cell to try to find one passer — that's leakage.
- Do not ship any current configuration to paper trading. The best cells either don't trade or lose meaningfully.
- Do not backfill funding data yet — that engineering time is better spent testing the daily-bars hypothesis first, because if daily also fails we know funding is a strategy-type change not a feature-add.

## Artifacts committed with this memo

- `configs/matrices/label_model_matrix_{btc,eth}_{1pct,2pct}.json` — the 4 matrix specs (32 cells total).
- `eval_reports/matrices/label_model_matrix_{btc,eth}_{1pct,2pct}.{json,csv,md}` — per-cell results, ranked CSV, and markdown summaries.
- All 32 cell fingerprints + reproducibility hashes stored in the `.json` files; re-running any single cell is `trade research run --spec-path <the spec block>` (or `trade research matrix` for the full matrix).
