# Threshold sweep — baseline_eth_5f_hourly_v1

- **Symbol**: `ETHUSDT`  
- **Features**: `log_return@5, realized_vol@20, atr@14, macd_hist@12_26_9, rsi_close@14`  
- **WFO**: train=8640h  test=1440h  step=1440h  expanding=False  
- **Label**: triple-barrier, horizon=6, ±1.00% barriers  
- **Bars used**: 17,520  
- **Generated**: `2026-08-07T16:02:57.248578+00:00`  
- **git**: `05ebdb76b9d7`  lock: `9750eaffca2c`

## Recommendation

> NO threshold passed the robustness gates; least-broken was 0.60 (cons=-0.880, gate reasons: pct_folds_positive_cas=0.00 < 0.50; annualized_turnover=68.71 > 50.00)

## Per-threshold aggregate

| threshold | mean_cas | pct_pos | max_dd% | ann_turnover | consistency | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.55 | -5.553 | 0.00 | 5.33 | 124.07 | -2.063 | FAIL |
| 0.60 | -3.809 | 0.00 | 3.11 | 68.71 | -0.880 | FAIL |
| 0.65 | -1.373 | 0.33 | 1.37 | 20.20 | -3.550 | FAIL |
| 0.70 | -1.197 | 0.17 | 1.03 | 10.13 | -3.116 | FAIL |
| 0.75 | -1.497 | 0.17 | 0.48 | 5.06 | -2.831 | FAIL |
| 0.80 | -1.635 | 0.00 | 0.43 | 3.04 | -2.534 | FAIL |

### Gate-failure detail
- `θ=0.55`: pct_folds_positive_cas=0.00 < 0.50; annualized_turnover=124.07 > 50.00
- `θ=0.60`: pct_folds_positive_cas=0.00 < 0.50; annualized_turnover=68.71 > 50.00
- `θ=0.65`: pct_folds_positive_cas=0.33 < 0.50
- `θ=0.70`: pct_folds_positive_cas=0.17 < 0.50
- `θ=0.75`: pct_folds_positive_cas=0.17 < 0.50
- `θ=0.80`: pct_folds_positive_cas=0.00 < 0.50; n_folds_with_trades=2 < 3

## Per-fold classifier diagnostics

These describe the trained model's predictive quality, independent of any trading threshold.

| fold | n_test | AUC-ROC | AUC-PR | ECE | support (down/flat/up) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1434 | 0.621 | 0.439 | 0.075 | 531 / 370 / 533 |
| 1 | 1434 | 0.581 | 0.399 | 0.052 | 631 / 237 / 566 |
| 2 | 1434 | 0.612 | 0.441 | 0.078 | 459 / 538 / 437 |
| 3 | 1434 | 0.563 | 0.375 | 0.065 | 649 / 243 / 542 |
| 4 | 1434 | 0.544 | 0.375 | 0.090 | 448 / 569 / 417 |
| 5 | 1434 | 0.577 | 0.398 | 0.043 | 519 / 462 / 453 |

## Trade-level metrics at the recommended threshold

_No gate-passing threshold to report on._