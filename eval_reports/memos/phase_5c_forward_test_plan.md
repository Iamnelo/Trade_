# Phase 5c — forward-test protocol: freeze now, evaluate on genuinely-unseen data

You approved the forward-test window "exactly as proposed": freeze the winning
model artifacts + today's git SHA, then evaluate them (no retraining) on fresh
data that post-dates the freeze. This memo records what was frozen, the harness
that will run the test, and one hard constraint you need to know about before
expecting numbers.

## The one thing you need to know: there is no forward data yet

Our daily klines end **2026-08-07**. That is the winners' last training bar.
Today is **2026-08-08**. A forward test measures whether the edge survives on
bars the model has *never seen* — and right now that window is **one day long
and still open**.

I cannot manufacture three months of future bars, and I will not fake them. The
only genuine forward test is **real calendar time passing**. There is no
in-sample shortcut that produces an honest out-of-sample number today: every bar
we already hold was available when the winners were selected, so re-scoring the
recent tail would just be another backtest, not a forward test.

So Phase 5c splits cleanly into two halves:

- **The time-sensitive half — done now.** Freeze the artifacts + git SHA today,
  so the forward window provably post-dates the model. Build the evaluator with
  the gates hard-coded so the eventual run is push-button and un-tunable.
- **The half that must wait — you trigger it later.** Once ≥ 3 months of
  post-cutoff daily bars exist, backfill them and run one command. No decisions,
  no knobs.

## What was frozen (provenance)

`scripts/freeze_winners.py` trained each winner on **all** daily bars through
the cutoff and serialised the exact artifacts. Training on the full history
(rather than one WFO fold's slice) is deliberate — it is the model you would
actually deploy, and it uses strictly more data than any fold saw. The
unresolvable triple-barrier horizon tail is trimmed, so the last resolved label
sits ~horizon bars before the cutoff — PIT-safe by construction.

| | BTC winner | ETH winner |
|---|---|---|
| Spec | `winner_btc_daily_v1` | `winner_eth_daily_v1` |
| Label mode | 3-class | 2-class directional |
| Features | baseline_5 | combined_12 |
| Horizon / barriers | 5d / ±5% | 10d / ±10% |
| θ | 0.55 | 0.55 |
| Train labels | 1821 | 1141 |
| Repro hash | `85f468f6ead7…` | `a360dc02f40d…` |
| Cutoff | 2026-08-07 | 2026-08-07 |

Artifacts: `artifacts/frozen/winner_{btc,eth}_daily_v1/` (booster + isotonic
calibrator + manifest) and `artifacts/frozen/freeze_manifest.json` (cutoff, git
SHA, lockfile SHA, feature ids, θ, label params, repro hashes). The forward
harness verifies each loaded artifact's repro hash against the manifest before
using it — a frozen model that has been tampered with will refuse to run.

## How the forward test runs (when data exists)

```
uv run python scripts/forward_test.py \
    --btc-forward BTCUSDT_D_forward.csv \
    --eth-forward ETHUSDT_D_forward.csv
```

The harness **loads** the frozen model (no retraining — the strict form of your
"do not modify the models during the forward test") and replays it over the
post-cutoff slice, with the same fees (5.5 bps) + slippage (5 bps) as every
prior experiment. It stitches just enough pre-cutoff history to warm up the
longest-lookback feature, then counts only bars with `event_time > cutoff` as
the test window. Output: `eval_reports/forward/forward_test_v1.json` plus a
printed per-symbol verdict.

Until real post-cutoff bars exist, the script exits cleanly with a "no forward
data yet" message rather than a fake result. `--self-test` exercises the whole
pipeline against committed (in-sample) bars purely to prove the plumbing — it
prints a loud banner that those numbers are **not** a forward result.

## Gates — hard-coded, per-symbol, independent

A symbol PASSES iff **all four** hold on its forward window:

1. Cost-adjusted Sharpe (CAS) **> 0**
2. Max drawdown **≤ 15%** *(the same gate that correctly killed the funded ETH arm — not relaxed)*
3. **≥ 5** fills
4. Calibration stable: forward ECE **≤ 2×** the in-sample reference ECE

The thresholds live in the source as constants, not CLI flags — a forward test
cannot be quietly loosened to manufacture a pass. Verdicts are **per-symbol and
independent**: if BTC passes and ETH fails, BTC advances on its own. We do not
require both.

## What happens on pass vs. fail

- **Pass (a symbol):** that winner earns the right to begin **paper-trading
  validation** (Phase 4c) — *after your explicit go-ahead*, never automatically.
  Passing is **permission to start collecting a live sample, not proof of a
  profitable system.** Three months of daily bars is a small sample; the paper
  phase is where we gather a much larger live record with no money at risk.
- **Fail (a symbol):** **do not tune it.** The report emits a
  failure-attribution block that separates the likely causes — small sample /
  inactivity, model degradation or regime change (forward AUC collapse vs the
  in-sample reference), calibration drift (forward ECE blow-up), or costs
  (negative expectancy against turnover). We diagnose first; we do not curve-fit
  the winner to make the forward window pass.

## What NOT to do

- Do not fabricate or "simulate" the forward window. Wait for real bars.
- Do not retrain, re-threshold, or re-feature the frozen winners for the forward
  test. They are frozen; that is the point.
- Do not start paper/live trading on a pass without your explicit approval.
- Do not treat a 3-month pass as a green light to size up. It is a green light
  to begin paper validation and gather more sample.

## Housekeeping: CI was silently red — now green

While wiring this up I found CI has been failing on **every** push on this
branch at a single step: `ruff format --check`. Because that step runs before
mypy, pytest, and the feature-contract check, all three were being **skipped** —
our real safety checks weren't actually running in CI. The cause was formatting
drift in eight files from earlier phases (pure whitespace/wrapping, no logic).
This change formats them, so mypy + pytest (528 tests) + feature-contract
discipline now run and pass in CI. Worth knowing: the reproducibility and
robustness discipline we've been relying on is only as good as the CI that
enforces it, and that enforcement was dark until now.

## Status

- Funding ablation: done, rule 3 (funding rejected). Winners unchanged.
- Winners: **frozen** as of git SHA `4ff6057…`, cutoff 2026-08-07.
- Forward harness: **built, tested, ready.** Waiting on real post-cutoff data.
- Phase 4c (paper trading): **not started**, per your standing rule.

Next action is yours: when ≥ 3 months of fresh daily bars exist (~Nov 2026),
backfill them and run `scripts/forward_test.py`. I'll report per-symbol verdicts
and, on any pass, prepare Phase 4c and wait for your explicit go-ahead before
anything executes.
