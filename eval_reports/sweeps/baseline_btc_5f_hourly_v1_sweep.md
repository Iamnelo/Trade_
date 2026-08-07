# Threshold sweep — baseline_btc_5f_hourly_v1

- **Symbol**: `BTCUSDT`  
- **Features**: `log_return@5, realized_vol@20, atr@14, macd_hist@12_26_9, rsi_close@14`  
- **WFO**: train=8640h  test=1440h  step=1440h  expanding=False  
- **Label**: triple-barrier, horizon=6, ±1.00% barriers  
- **Bars used**: 17,520  
- **Generated**: `2026-08-07T16:02:38.529172+00:00`  
- **git**: `05ebdb76b9d7`  lock: `9750eaffca2c`

## Recommendation

> NO threshold passed the robustness gates; least-broken was 0.75 (cons=-0.157, gate reasons: pct_folds_positive_cas=0.00 < 0.50; n_folds_with_trades=1 < 3)

## Per-threshold aggregate

| threshold | mean_cas | pct_pos | max_dd% | ann_turnover | consistency | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.55 | -2.663 | 0.00 | 7.51 | 43.01 | -2.895 | FAIL |
| 0.60 | -1.398 | 0.00 | 5.80 | 29.25 | -2.011 | FAIL |
| 0.65 | -0.513 | 0.00 | 4.60 | 23.28 | -1.258 | FAIL |
| 0.70 | -0.288 | 0.00 | 2.32 | 15.29 | -0.706 | FAIL |
| 0.75 | -0.064 | 0.00 | 2.16 | 9.23 | -0.157 | FAIL |
| 0.80 | -0.302 | 0.00 | 0.49 | 4.05 | -0.740 | FAIL |

### Gate-failure detail
- `θ=0.55`: pct_folds_positive_cas=0.00 < 0.50
- `θ=0.60`: pct_folds_positive_cas=0.00 < 0.50
- `θ=0.65`: pct_folds_positive_cas=0.00 < 0.50; n_folds_with_trades=1 < 3
- `θ=0.70`: pct_folds_positive_cas=0.00 < 0.50; n_folds_with_trades=1 < 3
- `θ=0.75`: pct_folds_positive_cas=0.00 < 0.50; n_folds_with_trades=1 < 3
- `θ=0.80`: pct_folds_positive_cas=0.00 < 0.50; n_folds_with_trades=1 < 3

## Per-fold classifier diagnostics

These describe the trained model's predictive quality, independent of any trading threshold.

| fold | n_test | AUC-ROC | AUC-PR | ECE | support (down/flat/up) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1434 | 0.530 | 0.360 | 0.060 | 237 / 992 / 205 |
| 1 | 1434 | 0.593 | 0.409 | 0.152 | 468 / 570 / 396 |
| 2 | 1434 | 0.589 | 0.403 | 0.055 | 339 / 777 / 318 |
| 3 | 1434 | 0.579 | 0.400 | 0.115 | 571 / 380 / 483 |
| 4 | 1434 | 0.560 | 0.384 | 0.110 | 303 / 805 / 326 |
| 5 | 1434 | 0.628 | 0.442 | 0.056 | 424 / 704 / 306 |

## Trade-level metrics at the recommended threshold

_No gate-passing threshold to report on._