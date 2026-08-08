# Phase 5b.2 decision memo — funding-features ablation

## What was run

The focused ablation you approved: no matrix expansion, no threshold retuning. Two symbols × two arms, everything else held identical to the Phase 5b.1.7 winners.

- **BTC arm A (baseline)** — winner_btc_daily_v1: 5 features (log_return@5, realized_vol@20, atr@14, macd_hist@12_26_9, rsi_close@14), 3c, 5-day horizon, ±5%.
- **BTC arm B (funded)** — same + `funding_rate@1`, `funding_zscore@21`, `funding_regime@9_63` (8 features total).
- **ETH arm A (baseline)** — winner_eth_daily_v1: 12 features (baseline_5 + time cyclic + higher-order + vol_regime), 2c, 10-day horizon, ±10%.
- **ETH arm B (funded)** — same + funding_3 (15 features).

5478 funding settlements per symbol (2021-08-08 → 2026-08-07, exactly 8h cadence, 18–19% negative regimes). Every WFO invocation used the same 3y-train / 6mo-test / 6mo-step schedule, θ=0.55, fee=5.5 bps, slippage=5 bps, gate spec unchanged. PIT-safe by construction: each funding feature runs `bisect_right(times, bar.event_time)` — no settlement with `event_time > bar.event_time` is visible.

Artifacts: `eval_reports/ablations/funding_ablation_v1.json` (per-arm dataclasses + deltas), `configs/experiments/winner_{btc,eth}_daily.json` (frozen baselines), `scripts/funding_ablation.py`, `scripts/user_supplied/fetch_funding.py`.

## Bottom line — funding HURTS on both symbols

**Decision: rule 3 fires. Remove funding features. Keep the daily winners as-is.**

Full metric-by-metric breakdown:

### BTCUSDT — winner_btc_daily_v1 (baseline_5)

| Metric | Baseline (PASS) | Funded (FAIL) | Δ |
|---|---:|---:|---:|
| Mean CAS | **+0.757** | **−0.752** | **−1.509** |
| Consistency score | −0.319 | −0.551 | −0.232 |
| pct folds positive CAS | **75%** | **0%** | −0.75 |
| Max fold DD % | 10.33 | 12.02 | +1.68 |
| Annualised turnover | **10.07** | **23.26** | **+13.19** |
| Total fills | 65 | 140 | +75 |
| Mean total return / fold | +2.12% | −2.75% | −4.87% |
| Mean Sharpe | +0.84 | −0.59 | −1.43 |
| Mean hit rate | 64.6% | 42.0% | −22.6pp |
| Mean win rate (trade-level) | 63.4% | 42.0% | −21.4pp |
| Mean expectancy / trade | −50.74 | +0.08 | +50.82 |
| Profit factor | 15.30 | 1.22 | −14.09 |
| **Mean AUC-ROC** | **0.521** | **0.528** | **+0.007** |
| Mean ECE | 0.127 | 0.130 | +0.003 |
| Oracle capture (long/short) | +0.001 | −0.001 | −0.002 |
| Gate | PASS | **FAIL** (pct_folds_positive_cas=0 < 0.5) | PASS→FAIL |

### ETHUSDT — winner_eth_daily_v1 (combined_12)

| Metric | Baseline (PASS) | Funded (FAIL) | Δ |
|---|---:|---:|---:|
| Mean CAS | **+0.341** | **−0.305** | **−0.646** |
| Consistency score | −0.374 | −1.603 | −1.230 |
| pct folds positive CAS | **50%** | **25%** | −0.25 |
| Max fold DD % | 12.91 | **21.49** (over 15% gate) | +8.58 |
| Annualised turnover | 19.92 | 20.13 | +0.21 |
| Total fills | 171 | 197 | +26 |
| Mean total return / fold | +3.92% | −4.20% | −8.12% |
| Mean Sharpe | +0.40 | −0.25 | −0.65 |
| Mean hit rate | 59.3% | 64.9% | +5.6pp |
| Mean win rate (trade-level) | 47.5% | 56.2% | +8.7pp |
| Mean expectancy / trade | +37.27 | +33.58 | −3.69 |
| Profit factor | 6.13 | 0.95 | −5.17 |
| **Mean AUC-ROC** | **0.547** | **0.541** | **−0.006** |
| Mean ECE | 0.085 | 0.096 | +0.012 |
| Oracle capture (long/short) | +0.001 | −0.000 | −0.001 |
| Gate | PASS | **FAIL** (DD 21.5% > 15%, pct+=0.25 < 0.5) | PASS→FAIL |

## Marginal contribution of funding features — the honest read

The single most diagnostic number in the whole ablation is **ΔAUC-ROC**:

- **BTC: +0.007 AUC.** Sixty-eight one-hundredths of a percent absolute lift on the classifier. That is floor noise for a WFO with only 4 folds of holdout data — well below what we'd need to see to conclude "funding adds signal".
- **ETH: −0.006 AUC.** The classifier got *slightly worse* with funding features added — again, floor noise, but the sign is the opposite of what a genuine signal would produce.

**Funding rates at daily bars do not carry additional predictive information for the direction of the next 5–10 day price move**, over and above what price/volume/vol already provide. That's the direct finding.

The strategy blowups (BTC CAS +0.76 → −0.75, ETH CAS +0.34 → −0.31) then have a clear mechanism: adding 3 uninformative features to a small-corpus LightGBM (~1090 training rows per fold) makes probability-mass distributions near the θ=0.55 threshold *shift* — differently in each fold — even when the classifier's ranking (measured by AUC) is essentially unchanged. Since the strategy's fate is decided by whether P(direction) crosses θ, those shifts change trade timing and turnover. BTC's turnover **more than doubled** (10.07 → 23.26 annually) with no matching return, so costs eat the strategy. ETH's DD **increased 66%** (12.9% → 21.5%) despite similar turnover — a specific fold got exposed to a large adverse move that the baseline model correctly stayed flat on.

None of that is funding "predicting the wrong thing" — it's funding adding *no* signal while adding noise to a model that has to allocate probability mass across more features.

## Rule application

Per your standing decision rules:

> 1. Funding improves both BTC and ETH: adopt funding features and proceed toward Phase 4c after a forward-test period.  
> 2. Funding is neutral: keep the simpler winner and proceed toward forward testing/paper trading.  
> **3. Funding hurts: remove it and do not force it into the model.**

**Rule 3 fires unambiguously on both symbols** (PASS → FAIL on both, ΔCAS clearly negative on both). The funding features are **not adopted**. The Phase 5b.1.7 winners remain the incumbent configurations.

- BTC winner (final): `configs/experiments/winner_btc_daily.json` — baseline_5 / 3c / 5d / ±5% / θ=0.55.
- ETH winner (final): `configs/experiments/winner_eth_daily.json` — combined_12 / 2c / 10d / ±10% / θ=0.55.

## Where does that leave us

- **Phase 5b research is done for now.** We have two gate-passing configurations that emerged from ~130 evaluated experiments (80 hourly + 48 daily matrix + 4 ablation-arm) plus honest infrastructure (WFO with PIT gates, robustness metrics, per-fold classifier diagnostics, oracle capture, reproducibility hashes). No configuration was engineered by tuning against gate results — the gate discipline held throughout.
- **Do not start paper trading yet, per your standing rule.** Before Phase 4c goes live, the memo you approved for 5b.1.7 said: "want funding-ablation + a forward-test window first." Funding is done and negative. Forward-test is the next thing.

## Forward-test window — my proposal

The Phase 5b.1.7 memo already predicted this moment. Concrete design:

**Objective**: verify the current daily winners keep working on data that was never seen during model selection.

**Cutoff date**: freeze the winner specs as of today (git sha `94a0089` or later). Any bar with `event_time > 2026-08-07` is out-of-sample forward-test data. The winners already trained through 2026-08-07; that's their last training bar.

**Window length**: at least 3 months of live daily bars — long enough that the ±5%/5d BTC winner sees ~15–20 potential trade decisions (its baseline turnover is ~10/year) and the ±10%/10d ETH winner sees ~10–15. Shorter than 3 months and 4-fold statistics don't inform anything.

**How it runs**:
1. When you're ready, backfill fresh daily klines: `2026-08-07 → today+3mo` (small — you already have the pipeline).
2. Freeze the trained artefacts from the 5b.1.7 winners' *last* fold and evaluate them (no retraining) against the new bars. This is the strictly cleanest test.
3. Report per-day equity curve, fill list, CAS, DD, expectancy, and gate result on the forward window as its own single-fold experiment.

**Decision criteria for greenlighting Phase 4c after forward-test**:
- CAS on the forward window > 0.
- Max DD on the forward window ≤ 15%.
- At least 5 fills.
- No unexplained regime break (i.e. classifier calibration on the forward window not egregiously worse than in-sample).

If the forward test also passes: greenlight Phase 4c on the winner. If it fails: the winners overfit to the training-through-2026-08-07 distribution and we go back to feature/model work — no live money.

## What NOT to do

- Do not re-run the whole matrix at daily now that we know it works. Diminishing returns — the current 12/48 passer rate is what we had 6 hours of compute ago; more knob-turning is data-snooping.
- Do not tune the ±5%/5d BTC winner further to try to beat itself. That's post-hoc curve-fitting to the training window.
- Do not skip the forward test. The winners passed a WFO gate on data that ended today. The one thing they haven't passed is *"the model doesn't know about tomorrow yet"*. Forward-test is precisely that check.
- Do not start paper trading tomorrow because the ablation was clean. Ablation only tells us funding was the wrong addition; it doesn't add out-of-sample evidence.

## Artifacts committed with this memo

- `eval_reports/ablations/funding_ablation_v1.json` — per-arm complete dataclass, every metric from the "please report" list, per-fold deltas.
- `configs/experiments/winner_{btc,eth}_daily.json` — frozen winners (5b.1.7 champions), also the baseline arms of this ablation. Any cell in these files can be re-run byte-for-byte via `trade research run --spec-path <file>`.
- `BTCUSDT_funding_5y.csv`, `ETHUSDT_funding_5y.csv` — the funding data, kept for later reference and possible use in a funding-arbitrage strategy (a genuinely different modeling problem from directional prediction).
- `scripts/funding_ablation.py` — the focused ablation runner, reproducible via `uv run python scripts/funding_ablation.py`.
- `scripts/user_supplied/fetch_funding.py` — your zero-dep script that produced the CSVs from Bybit v5 REST.
