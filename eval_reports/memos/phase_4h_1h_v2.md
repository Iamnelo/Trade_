# Revised ETH 4H and 1H research — V2 decision

## Scope

This phase evaluated revised, lower-turnover ETHUSDT model configurations on
4-hour and 1-hour candles. It did not modify, restart, or deploy any VPS paper
service. Daily V1 and the ETH 12H shadow service remain unchanged.

## Changes from the first pass

- longer label horizons;
- stronger confidence thresholds;
- smaller notional fractions;
- optional volatility-regime feature;
- larger minimum leaf sizes; and
- tighter turnover and positive-fold gates.

## Results

| Timeframe | Cells | Passed | Best useful observation | Decision |
| --- | ---: | ---: | --- | --- |
| ETH 4H | 18 | 0 | One cell reached mean CAS `0.041`, but only `25%` of folds were positive | Reject V2 |
| ETH 1H | 18 | 0 | Every cell had negative mean CAS; annualized turnover remained `29.36–83.89` | Reject V2 |

The 4H candidate with slightly positive mean CAS still failed consistency and
positive-fold requirements. The 1H failures show that simply raising the
threshold and extending the horizon did not overcome the cost/turnover problem.

## Decision

- Do not freeze or deploy either V2 matrix.
- Keep Daily V1 and ETH 12H running unchanged.
- Evaluate the predeclared rule-based Bollinger mean-reversion and Donchian
  breakout benchmarks on the same 4H and 1H data next.
- Do not begin a scalping paper service until a cost-stressed offline benchmark
  demonstrates an edge.
