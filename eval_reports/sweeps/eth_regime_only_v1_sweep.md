# Threshold sweep — eth_regime_only_v1

- **Symbol**: `ETHUSDT`  
- **Features**: `log_return@5, realized_vol@20, atr@14, macd_hist@12_26_9, rsi_close@14, vol_regime@20_120`  
- **WFO**: train=8640h  test=1440h  step=1440h  expanding=False  
- **Label**: triple-barrier, horizon=6, ±1.00% barriers  
- **Bars used**: 17,520  
- **Generated**: `2026-08-07T16:29:34.232225+00:00`  
- **git**: `d13beca4a0ca`  lock: `9750eaffca2c`

## Recommendation

> NO threshold passed the robustness gates; least-broken was 0.80 (cons=-1.500, gate reasons: pct_folds_positive_cas=0.17 < 0.50; n_folds_with_trades=2 < 3)

## Per-threshold aggregate

| threshold | mean_cas | pct_pos | max_dd% | ann_turnover | consistency | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.55 | -2.871 | 0.00 | 4.45 | 107.47 | -2.372 | FAIL |
| 0.60 | -2.037 | 0.17 | 3.40 | 54.50 | -2.820 | FAIL |
| 0.65 | -2.262 | 0.17 | 2.08 | 30.31 | -2.946 | FAIL |
| 0.70 | -1.796 | 0.33 | 1.29 | 22.25 | -3.083 | FAIL |
| 0.75 | -0.474 | 0.33 | 1.29 | 18.22 | -2.172 | FAIL |
| 0.80 | -0.458 | 0.17 | 1.01 | 5.07 | -1.500 | FAIL |

### Gate-failure detail
- `θ=0.55`: pct_folds_positive_cas=0.00 < 0.50; annualized_turnover=107.47 > 50.00
- `θ=0.60`: pct_folds_positive_cas=0.17 < 0.50; annualized_turnover=54.50 > 50.00
- `θ=0.65`: pct_folds_positive_cas=0.17 < 0.50
- `θ=0.70`: pct_folds_positive_cas=0.33 < 0.50
- `θ=0.75`: pct_folds_positive_cas=0.33 < 0.50
- `θ=0.80`: pct_folds_positive_cas=0.17 < 0.50; n_folds_with_trades=2 < 3

## Per-fold classifier diagnostics

These describe the trained model's predictive quality, independent of any trading threshold.

| fold | n_test | AUC-ROC | AUC-PR | ECE | support (down/flat/up) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1434 | 0.628 | 0.450 | 0.067 | 531 / 370 / 533 |
| 1 | 1434 | 0.582 | 0.396 | 0.050 | 631 / 237 / 566 |
| 2 | 1434 | 0.607 | 0.425 | 0.074 | 459 / 538 / 437 |
| 3 | 1434 | 0.571 | 0.384 | 0.070 | 649 / 243 / 542 |
| 4 | 1434 | 0.545 | 0.376 | 0.090 | 448 / 569 / 417 |
| 5 | 1434 | 0.594 | 0.417 | 0.046 | 519 / 462 / 453 |

## Trade-level metrics at the recommended threshold

_No gate-passing threshold to report on._