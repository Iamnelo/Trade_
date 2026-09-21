# ETH 4H and 1H rule-strategy comparison

## Scope

This phase evaluated two deterministic strategy families on the same ETHUSDT
4-hour and 1-hour historical datasets used by the model research:

- Bollinger-band mean reversion, with windows `20`, `40`, and `80` and band
  widths `1.5`, `2.0`, and `2.5` standard deviations; and
- Donchian breakout, with windows `20`, `40`, and `80`.

All variants used a `0.25` notional fraction, allowed long and short targets,
and included `5.5` bps fees plus `5.0` bps slippage. Results were evaluated on
the same four chronological test folds and robustness gates as the revised ML
matrices.

## Implementation correction

The current decision bar is now excluded from the reference Bollinger and
Donchian windows. This prevents the signal candle from defining the boundary
it is being compared against. Dedicated tests verify that downside Bollinger
extremes and upside Donchian breakouts can create orders.

## Results

| Timeframe | Variants | Passed | Main failure |
| --- | ---: | ---: | --- |
| ETH 4H | 12 | 0 | Negative mean cost-adjusted Sharpe and turnover above the gate |
| ETH 1H | 12 | 0 | Strongly negative cost-adjusted Sharpe and extreme turnover |

The closest 4H result by mean cost-adjusted Sharpe was Bollinger `window=80`,
`std=2.0`: mean CAS `-0.314`, `25%` positive folds, maximum drawdown `5.06%`,
and annualized turnover `35.16` versus the `20.0` limit.

The strongest 1H mean CAS was still negative (`-2.755` for Bollinger
`window=80`, `std=2.0`) with annualized turnover `137.69` versus the `12.0`
limit.

## Decision

- Reject every tested rule-based 4H and 1H variant.
- Do not freeze, deploy, or start Telegram/VPS services for these candidates.
- Keep Daily V1 and ETH 12H unchanged.
- Do not deploy a scalper based on these signal families; the 1H results show
  that higher-frequency trading costs dominate the tested edges.
