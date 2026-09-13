# 12H strategy research — first pass

## Scope and isolation

This phase evaluates a separate 12-hour challenger. It does not modify the frozen Daily V1 artifacts, paper-trading configuration, Telegram notifier, or any running service.

The committed two-year hourly BTCUSDT and ETHUSDT datasets were resampled into 12-hour candles by `scripts/resample_hourly_to_12h.py`. Both outputs contain 1,460 complete candles from 2024-08-06 00:00 UTC through 2026-08-05 12:00 UTC. Validation found no gaps, duplicates, incomplete buckets, negative volume, or invalid OHLC rows. These are offline resamples, not independently downloaded native Bybit 720-minute candles.

Reproduce the input files before running the matrices:

```bash
uv run python scripts/resample_hourly_to_12h.py \
  --output-dir offline-12h \
  --report offline-12h/validation-report.json
```

## Predeclared matrix

Each symbol ran 12 cells: two label modes (3-class and 2-class directional), three horizons (2, 5, and 10 bars), and two feature sets (baseline 5 and combined 12). All cells used a fixed 0.55 confidence threshold, 2% symmetric triple barriers, one-year training windows, three-month test windows, realistic fees and slippage, and the existing unchanged robustness gates.

- BTC: all 12 cells failed. No BTC candidate had positive mean cost-adjusted Sharpe.
- ETH: all 12 cells failed at the fixed 0.55 threshold. The directional 10-bar/baseline-5 cell had the strongest raw return profile (mean CAS 0.457, 75% positive folds), but failed drawdown (28.22%) and annualized turnover (106.39).

## ETH threshold follow-up

| Threshold | Mean CAS | Positive folds | Max drawdown | Annualized turnover | Gate |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0.55 | 0.457 | 75% | 28.22% | 106.39 | FAIL |
| 0.60 | -0.398 | 75% | 27.86% | 91.03 | FAIL |
| 0.65 | -1.297 | 25% | 11.35% | 60.09 | FAIL |
| 0.70 | -0.575 | 0% | 8.35% | 50.82 | FAIL |
| 0.75 | -0.426 | 0% | 8.35% | 39.50 | FAIL |
| 0.80 | 0.031 | 50% | 6.74% | 37.68 | PASS |

## Decision

Treat the 0.80 ETH result as a **provisional challenger only**, not a paper or live strategy. The mean edge is nearly flat, one fold produced no trades, and classifier AUC varied sharply by fold. Passing the coarse gates is insufficient evidence for deployment.

The next valid step is to freeze this exact challenger specification and test it on genuinely unseen post-2026-08-05 12H candles without retraining or retuning. Daily V1 continues unchanged and independently.
