# Threshold sweep — btc_combined_winners_v1

- **Symbol**: `BTCUSDT`  
- **Features**: `log_return@5, realized_vol@20, atr@14, macd_hist@12_26_9, rsi_close@14, time_of_day@sin, time_of_day@cos, day_of_week@sin, day_of_week@cos, return_skew@20, return_kurtosis@20, vol_regime@20_120`  
- **WFO**: train=8640h  test=1440h  step=1440h  expanding=False  
- **Label**: triple-barrier, horizon=6, ±1.00% barriers  
- **Bars used**: 17,520  
- **Generated**: `2026-08-07T16:31:18.889719+00:00`  
- **git**: `d13beca4a0ca`  lock: `9750eaffca2c`

## Recommendation

> NO threshold passed the robustness gates; least-broken was 0.65 (cons=-2.215, gate reasons: pct_folds_positive_cas=0.00 < 0.50)

## Per-threshold aggregate

| threshold | mean_cas | pct_pos | max_dd% | ann_turnover | consistency | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.55 | -2.137 | 0.00 | 3.71 | 62.50 | -2.409 | FAIL |
| 0.60 | -1.629 | 0.00 | 2.35 | 32.29 | -2.256 | FAIL |
| 0.65 | -1.957 | 0.00 | 1.94 | 21.20 | -2.215 | FAIL |
| 0.70 | -1.866 | 0.00 | 1.68 | 13.13 | -2.340 | FAIL |
| 0.75 | -1.418 | 0.17 | 1.31 | 8.09 | -2.773 | FAIL |
| 0.80 | -0.881 | 0.17 | 1.37 | 5.04 | -2.596 | FAIL |

### Gate-failure detail
- `θ=0.55`: pct_folds_positive_cas=0.00 < 0.50; annualized_turnover=62.50 > 50.00
- `θ=0.60`: pct_folds_positive_cas=0.00 < 0.50
- `θ=0.65`: pct_folds_positive_cas=0.00 < 0.50
- `θ=0.70`: pct_folds_positive_cas=0.00 < 0.50
- `θ=0.75`: pct_folds_positive_cas=0.17 < 0.50
- `θ=0.80`: pct_folds_positive_cas=0.17 < 0.50; n_folds_with_trades=2 < 3

## Per-fold classifier diagnostics

These describe the trained model's predictive quality, independent of any trading threshold.

| fold | n_test | AUC-ROC | AUC-PR | ECE | support (down/flat/up) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1434 | 0.639 | 0.413 | 0.034 | 237 / 992 / 205 |
| 1 | 1434 | 0.671 | 0.487 | 0.122 | 468 / 570 / 396 |
| 2 | 1434 | 0.705 | 0.492 | 0.047 | 339 / 777 / 318 |
| 3 | 1434 | 0.638 | 0.452 | 0.088 | 571 / 380 / 483 |
| 4 | 1434 | 0.635 | 0.439 | 0.088 | 303 / 805 / 326 |
| 5 | 1434 | 0.687 | 0.494 | 0.053 | 424 / 704 / 306 |

## Trade-level metrics at the recommended threshold

_No gate-passing threshold to report on._