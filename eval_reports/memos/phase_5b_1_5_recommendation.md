# Phase 5b.1.5 recommendation memo

## What was run

Two rounds of experiments on 2y hourly BTCUSDT + ETHUSDT (2024‑08‑06 → 2026‑08‑05, 6 rolling folds, 12mo train / 2mo test / 2mo step):

**Experiment B — Confidence-threshold sweep.** Same 5-feature baseline (`log_return@5`, `realized_vol@20`, `atr@14`, `macd_hist@12_26_9`, `rsi_close@14`), sweep θ ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80}. One model fit per fold, six strategy replays per fold.

**Experiment A — Combined winners.** BTC: baseline + `time_of_day@sin/cos` + `day_of_week@sin/cos` + `return_skew@20` + `return_kurtosis@20` + `vol_regime@20_120` (12 features). ETH: baseline + `vol_regime@20_120` only (6 features). Same threshold sweep.

Full data in `eval_reports/sweeps/*.csv`, `*.md`, `*.html`, `*.json`.

## The bottom line

**No configuration passed the robustness gates on either symbol.** The best "least-broken" configurations:

| Config | Symbol | θ | Mean CAS | pct pos folds | Max DD % | Ann TV | Cons | Gate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline sweep | BTC | 0.75 | −0.06 | 0.00 | 2.16 | 9.23 | **−0.16** | FAIL |
| baseline sweep | ETH | 0.70 | −1.20 | 0.17 | 1.03 | 10.13 | −3.12 | FAIL |
| combined winners | BTC | 0.80 | −0.88 | 0.17 | 1.37 | 5.04 | −2.60 | FAIL |
| regime‑only | ETH | 0.80 | −0.46 | 0.17 | 1.01 | 5.07 | −1.50 | FAIL |

*Cons = `mean_cas × pct_positive − std_cas`. Higher = better.*

## Signal quality: the model IS learning something

Per‑fold classifier diagnostics — measured on the horizon‑trimmed holdout, independent of the trading rule:

| Feature set | Symbol | Mean AUC‑ROC | Mean AUC‑PR | Mean ECE |
|---|:---:|:---:|:---:|:---:|
| baseline (5) | BTC | 0.580 | 0.400 | 0.091 |
| combined (12) | BTC | **0.663** | **0.463** | 0.078 |
| baseline (5) | ETH | 0.583 | 0.406 | 0.069 |
| regime‑only (6) | ETH | 0.588 | 0.408 | 0.065 |

**On BTC, adding the winning families lifted AUC‑ROC from 0.580 to 0.663 — a ~14% relative improvement in classifier quality that is real.** AUC‑PR moved from 0.400 to 0.463 (baseline for 3 balanced classes ≈ 0.33). ECE improved slightly. On ETH, adding just `vol_regime` barely moved the classifier (+0.5 AUC points).

## The paradox that determines the recommendation

Look at BTC combined‑winners vs BTC baseline at the same threshold:

| θ | BTC baseline (5 features) | BTC combined (12 features) |
|:---:|:---:|:---:|
| 0.55 | CAS −2.66, 0% folds+ | CAS −2.14, 0% folds+ |
| 0.75 | CAS −0.06, **0%** folds+ | CAS −1.42, **17%** folds+ |
| 0.80 | CAS −0.30, 0% folds+ | CAS −0.88, **17%** folds+ |

**More features → better classifier → some folds now cross into positive CAS territory (17% of them) — but average P&L got worse.** That's the classic "features add signal but also add variance" pattern: some folds win big, others lose big, mean drifts down.

Read: the classifier is learning genuine information; the decision layer (argmax + calibrated-probability threshold) is not converting it into robustly-profitable trades. That points at **two co‑equal bottlenecks: features and modeling approach.**

## Answering the three options

Framed as the memo requested:

**1. Is the current feature set sufficient?** **No.** Baseline AUC‑ROC 0.58 gives too small an edge relative to the 10.5 bps round‑trip cost. Combined‑winners AUC‑ROC 0.66 is meaningfully better but still insufficient to overcome cost + variance at any threshold.

**2. Is funding data likely to be the missing signal?** **Yes, partly — recommend proceeding with Phase 5b.2.** Funding rate is uncorrelated with price/volume/vol features (it's a market‑microstructure quantity determined by inventory imbalance) and the literature has clear evidence of predictive power against perpetual‑futures returns, especially at horizons that match our 6h barrier. Backfill effort is modest (one CSV per symbol via the existing `trade backfill funding bybit` CLI — same egress‑block situation as klines, so it's a local‑backfill job). Expected classifier bump: another ~2–5 AUC points, based on published cross‑asset funding studies.

**3. Is a different modeling approach required?** **Yes — and this is the one I would rank AHEAD of pure feature additions.** Three concrete changes, each testable with the existing framework and cheap:

   a. **Move from 3-class to 2-class classifier.** Right now the model burns probability mass on the dominant "flat" class (support 400–1000 out of 1434 test samples per fold). A 2-class "up-vs-not" + separate "down-vs-not" ensemble would let confidence on the trading classes grow instead of being suppressed. This alone might do more than funding.
   
   b. **Wider label barriers, longer horizons.** ±1% / 6h is close enough to noise on BTC/ETH that the model is fitting near-random labels for a chunk of the training set. Try ±2% / 24h barriers — the ablation framework can grid this trivially.
   
   c. **Isotonic recalibration on the holdout, not the tail slice.** Current pipeline carves the calibration set from the tail of TRAIN. That still overlaps distributionally with test. Consider a purged calibration slice.

## What I'd do next, in order

1. **Move the modeling change ahead of funding data.** Grid `{2-class vs 3-class} × {6h/24h horizons} × {±1%/±2% barriers} × {baseline vs combined winners}` — 16 experiments, all runnable via the existing `trade research sweep` + a small addition to support 2‑class labeling. Compute: ~90 min in this session.
2. **THEN backfill funding data (Phase 5b.2).** Backfill needs to happen locally (proxy blocks Bybit) — I'll re-issue the recipe. Wire `funding_rate@N`, `funding_zscore@N`, `funding_regime@N` features and run the ablation + sweep matrix again.
3. **Only after ONE configuration passes the gates on both symbols independently**, resume the Phase 4c paper-trading orchestrator work and validate the winner end-to-end.

**Do not** ship the current best-of-worst config to paper trading. Neither symbol is close to positive expectancy after costs; committing engineering effort to the paper loop with no ready strategy would be busy-work.

## Artifacts committed with this memo

- `configs/experiments/btc_combined_winners.json`, `configs/experiments/eth_regime_only.json` — Experiment A specs.
- `eval_reports/sweeps/{baseline_btc,baseline_eth,btc_combined_winners_v1,eth_regime_only_v1}_sweep.{json,md,html}` — per-run reports.
- `eval_reports/sweeps/*__cells.csv`, `*__thresholds.csv`, `*__classifier.csv` — dense tables for downstream analysis.
