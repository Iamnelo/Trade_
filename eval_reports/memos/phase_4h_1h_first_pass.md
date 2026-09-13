# 4H and 1H strategy research — first pass

## Scope

This phase evaluated separate BTCUSDT and ETHUSDT challengers on 4-hour and 1-hour candles. It did not modify or restart Daily V1, the paper service, the Telegram notifier, or any VPS process.

## Data

- 4H: 4,380 candles per symbol, resampled from the committed two-year hourly data. Validation found no gaps, duplicates, incomplete buckets, negative volume, or invalid OHLC rows.
- 1H: the committed 17,520 hourly candles per symbol were used directly.

## Fixed matrices

Each symbol/timeframe ran 12 predeclared cells: two label modes, three label horizons, and two feature sets. All used the existing fee/slippage assumptions and the same drawdown, positive-fold, activity, and turnover gates.

## Results

| Timeframe | Symbol | Result | Decision |
| :---: | :---: | :---: | :--- |
| 4H | BTC | All 12 had negative mean CAS | Reject all |
| 4H | ETH | All 12 had negative mean CAS | Reject all |
| 1H | BTC | All 12 failed; mean CAS strongly negative and turnover extreme | Reject all |
| 1H | ETH | All 12 failed; mean CAS negative and turnover extreme | Reject all |

No 4H or 1H candidate qualifies for a confidence-threshold sweep, model freeze, paper deployment, or live deployment.

## Gate correction

The first 4H run exposed that the robustness gate could pass a model when half the folds were positive even if the mean cost-adjusted Sharpe was negative. A new hard gate now requires `mean_cost_adjusted_sharpe >= 0` by default. The 4H matrices were rerun under the corrected gate and all candidates failed.

## Current multi-timeframe decision

- Daily V1: continue unchanged.
- ETH 12H at threshold 0.80: retain as a provisional shadow challenger only; its edge is thin and still requires genuinely unseen validation.
- BTC 12H: rejected.
- BTC/ETH 4H: rejected for this specification.
- BTC/ETH 1H: rejected for this specification.
