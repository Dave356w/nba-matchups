# WP to MOV calibration: nested walk-forward

Research only. Final home points minus away points, including overtime. No market data.
Probability sample: 9,509 games across 9 seasons. Mapping warm-up: 2016-17 (1,074 games).
Matched evaluation: 8,435 games across 8 seasons. Games before both teams have 10 prior games are excluded.

Every WP uses composite weights and logits fitted on earlier seasons only. For season Y, margin coefficients use ONLY out-of-sample WP/outcome pairs from seasons < Y. No same-season or in-sample WP is used to fit a mapping. The earliest WP season is excluded equally for all methods.

Mappings, z = inverse standard normal CDF(WP): fixed = 13.5*z; scale_only = b*z; affine = a+b*z (ordinary least squares). The fitted b is a conditional-mean conversion scale, not an independently estimated residual SD. Bias means prediction minus actual; negative comparison deltas favor the fitted mapping.

## Matched out-of-sample results

| Method | Games | MAE | RMSE | Bias | Actual-on-pred intercept | Actual-on-pred slope |
|---|---:|---:|---:|---:|---:|---:|
| affine | 8435 | 10.6436 | 13.6389 | +0.0979 | -0.4435 | 1.1695 |
| fixed_13_5 | 8435 | 10.6568 | 13.6516 | +0.8901 | -0.6212 | 0.9050 |
| scale_only | 8435 | 10.6400 | 13.6367 | +0.1848 | -0.5874 | 1.1895 |

## Paired changes versus fixed 13.5

95% percentile CIs from paired date-cluster bootstrap (2,000 draws, seed 20261021). All games on a sampled date stay together; cross-date dependence and feature-selection risk are not captured.

| Method | Delta MAE [95% CI] | Delta RMSE [95% CI] |
|---|---:|---:|
| scale_only | -0.0168 [-0.0548, +0.0271] | -0.0149 [-0.0553, +0.0281] |
| affine | -0.0133 [-0.0518, +0.0308] | -0.0127 [-0.0538, +0.0314] |

## By season

| Group | Method | Games | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| 2017-18 | affine | 1075 | 9.677 | 12.477 | +0.835 |
| 2018-19 | affine | 1074 | 10.108 | 13.033 | -0.244 |
| 2020-21 | affine | 924 | 10.939 | 13.859 | +1.507 |
| 2021-22 | affine | 1073 | 11.084 | 14.146 | +0.487 |
| 2022-23 | affine | 1073 | 10.072 | 12.715 | -0.883 |
| 2023-24 | affine | 1074 | 11.053 | 14.195 | -0.174 |
| 2024-25 | affine | 1071 | 10.968 | 14.122 | -0.266 |
| 2025-26 | affine | 1071 | 11.293 | 14.450 | -0.285 |
| 2017-18 | fixed_13_5 | 1075 | 9.836 | 12.600 | +1.268 |
| 2018-19 | fixed_13_5 | 1074 | 10.077 | 13.045 | +0.524 |
| 2020-21 | fixed_13_5 | 924 | 11.010 | 13.923 | +2.225 |
| 2021-22 | fixed_13_5 | 1073 | 11.130 | 14.201 | +1.490 |
| 2022-23 | fixed_13_5 | 1073 | 10.058 | 12.730 | +0.140 |
| 2023-24 | fixed_13_5 | 1074 | 11.011 | 14.114 | +0.691 |
| 2024-25 | fixed_13_5 | 1071 | 10.947 | 14.106 | +0.516 |
| 2025-26 | fixed_13_5 | 1071 | 11.239 | 14.400 | +0.450 |
| 2017-18 | scale_only | 1075 | 9.644 | 12.432 | +0.120 |
| 2018-19 | scale_only | 1074 | 10.108 | 13.031 | -0.442 |
| 2020-21 | scale_only | 924 | 10.927 | 13.841 | +1.377 |
| 2021-22 | scale_only | 1073 | 11.098 | 14.164 | +0.736 |
| 2022-23 | scale_only | 1073 | 10.053 | 12.696 | -0.471 |
| 2023-24 | scale_only | 1074 | 11.074 | 14.210 | +0.115 |
| 2024-25 | scale_only | 1071 | 10.967 | 14.132 | +0.103 |
| 2025-26 | scale_only | 1071 | 11.292 | 14.463 | +0.107 |

## By model route

| Group | Method | Games | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| avail | affine | 3216 | 11.105 | 14.256 | -0.242 |
| base | affine | 5076 | 10.367 | 13.253 | +0.285 |
| v4 | affine | 143 | 10.101 | 12.926 | +1.081 |
| avail | fixed_13_5 | 3216 | 11.066 | 14.207 | +0.553 |
| base | fixed_13_5 | 5076 | 10.417 | 13.314 | +1.088 |
| v4 | fixed_13_5 | 143 | 9.993 | 12.685 | +1.439 |
| avail | scale_only | 3216 | 11.111 | 14.269 | +0.108 |
| base | scale_only | 5076 | 10.358 | 13.243 | +0.228 |
| v4 | scale_only | 143 | 10.040 | 12.838 | +0.358 |

## By season phase

| Group | Method | Games | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| opening third | affine | 2081 | 10.475 | 13.465 | -0.196 |
| middle third | affine | 3325 | 10.671 | 13.672 | +0.236 |
| last third | affine | 3029 | 10.729 | 13.721 | +0.148 |
| opening third | fixed_13_5 | 2081 | 10.515 | 13.478 | +0.593 |
| middle third | fixed_13_5 | 3325 | 10.694 | 13.728 | +1.050 |
| last third | fixed_13_5 | 3029 | 10.713 | 13.686 | +0.918 |
| opening third | scale_only | 2081 | 10.465 | 13.453 | -0.116 |
| middle third | scale_only | 3325 | 10.673 | 13.669 | +0.321 |
| last third | scale_only | 3029 | 10.725 | 13.726 | +0.242 |

## By favorite probability

| Group | Method | Games | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| 0.5-0.6 | affine | 2654 | 10.977 | 13.866 | +0.297 |
| 0.6-0.7 | affine | 2478 | 10.790 | 13.668 | +0.497 |
| 0.7-0.8 | affine | 1940 | 10.288 | 13.396 | +0.171 |
| 0.8-0.9 | affine | 1201 | 10.197 | 13.433 | -1.054 |
| 0.9-1.0 | affine | 162 | 10.508 | 13.842 | -1.597 |
| 0.5-0.6 | fixed_13_5 | 2654 | 10.955 | 13.850 | +0.454 |
| 0.6-0.7 | fixed_13_5 | 2478 | 10.740 | 13.695 | +0.963 |
| 0.7-0.8 | fixed_13_5 | 1940 | 10.273 | 13.393 | +1.149 |
| 0.8-0.9 | fixed_13_5 | 1201 | 10.355 | 13.455 | +1.061 |
| 0.9-1.0 | fixed_13_5 | 162 | 11.343 | 14.200 | +2.550 |
| 0.5-0.6 | scale_only | 2654 | 10.980 | 13.860 | +0.403 |
| 0.6-0.7 | scale_only | 2478 | 10.781 | 13.665 | +0.581 |
| 0.7-0.8 | scale_only | 1940 | 10.289 | 13.402 | +0.248 |
| 0.8-0.9 | scale_only | 1201 | 10.184 | 13.428 | -0.981 |
| 0.9-1.0 | scale_only | 162 | 10.497 | 13.841 | -1.571 |

## Mapping coefficients fitted before each test season

| Test season | Method | Training games | Last training season (BBR year) | Intercept | Scale |
|---|---|---:|---:|---:|---:|
| 2017-18 | fixed_13_5 | 1074 | 2017 | +0.0000 | 13.5000 |
| 2017-18 | scale_only | 1074 | 2017 | +0.0000 | 9.1141 |
| 2017-18 | affine | 1074 | 2017 | +0.8486 | 8.6037 |
| 2018-19 | fixed_13_5 | 2149 | 2018 | +0.0000 | 13.5000 |
| 2018-19 | scale_only | 2149 | 2018 | +0.0000 | 9.4935 |
| 2018-19 | affine | 2149 | 2018 | +0.2356 | 9.3382 |
| 2020-21 | fixed_13_5 | 3223 | 2019 | +0.0000 | 13.5000 |
| 2020-21 | scale_only | 3223 | 2019 | +0.0000 | 10.0685 |
| 2020-21 | affine | 3223 | 2019 | +0.1556 | 9.9652 |
| 2021-22 | fixed_13_5 | 4147 | 2021 | +0.0000 | 13.5000 |
| 2021-22 | scale_only | 4147 | 2021 | +0.0000 | 10.2140 |
| 2021-22 | affine | 4147 | 2021 | -0.2946 | 10.4137 |
| 2022-23 | fixed_13_5 | 5220 | 2022 | +0.0000 | 13.5000 |
| 2022-23 | scale_only | 5220 | 2022 | +0.0000 | 10.4186 |
| 2022-23 | affine | 5220 | 2022 | -0.4778 | 10.7461 |
| 2023-24 | fixed_13_5 | 6293 | 2023 | +0.0000 | 13.5000 |
| 2023-24 | scale_only | 6293 | 2023 | +0.0000 | 10.5436 |
| 2023-24 | affine | 6293 | 2023 | -0.3346 | 10.7769 |
| 2024-25 | fixed_13_5 | 7367 | 2024 | +0.0000 | 13.5000 |
| 2024-25 | scale_only | 7367 | 2024 | +0.0000 | 10.9412 |
| 2024-25 | affine | 7367 | 2024 | -0.4142 | 11.2250 |
| 2025-26 | fixed_13_5 | 8438 | 2025 | +0.0000 | 13.5000 |
| 2025-26 | scale_only | 8438 | 2025 | +0.0000 | 11.1796 |
| 2025-26 | affine | 8438 | 2025 | -0.4331 | 11.4588 |

## Reliability by home win probability

Descriptive averages on the identical held-out games; not used to fit or select a mapping.

| WP band | Method | Games | Predicted home margin | Actual home margin | Bias |
|---|---|---:|---:|---:|---:|
| 0.1-0.2 | affine | 249 | -10.667 | -11.863 | +1.196 |
| 0.2-0.3 | affine | 584 | -6.953 | -9.127 | +2.174 |
| 0.3-0.4 | affine | 891 | -4.039 | -5.278 | +1.240 |
| 0.4-0.5 | affine | 1231 | -1.325 | -2.102 | +0.777 |
| 0.5-0.6 | affine | 1423 | +1.241 | +1.359 | -0.118 |
| 0.6-0.7 | affine | 1587 | +3.910 | +3.830 | +0.079 |
| 0.7-0.8 | affine | 1356 | +6.803 | +7.494 | -0.691 |
| 0.8-0.9 | affine | 952 | +10.362 | +12.004 | -1.643 |
| 0.9-1.0 | affine | 153 | +14.542 | +16.176 | -1.634 |
| 0.1-0.2 | fixed_13_5 | 249 | -13.265 | -11.863 | -1.402 |
| 0.2-0.3 | fixed_13_5 | 584 | -8.879 | -9.127 | +0.248 |
| 0.3-0.4 | fixed_13_5 | 891 | -5.125 | -5.278 | +0.153 |
| 0.4-0.5 | fixed_13_5 | 1231 | -1.583 | -2.102 | +0.519 |
| 0.5-0.6 | fixed_13_5 | 1423 | +1.757 | +1.359 | +0.398 |
| 0.6-0.7 | fixed_13_5 | 1587 | +5.248 | +3.830 | +1.417 |
| 0.7-0.8 | fixed_13_5 | 1356 | +9.032 | +7.494 | +1.538 |
| 0.8-0.9 | fixed_13_5 | 952 | +13.709 | +12.004 | +1.705 |
| 0.9-1.0 | fixed_13_5 | 153 | +19.140 | +16.176 | +2.963 |
| 0.1-0.2 | scale_only | 249 | -10.341 | -11.863 | +1.522 |
| 0.2-0.3 | scale_only | 584 | -6.778 | -9.127 | +2.348 |
| 0.3-0.4 | scale_only | 891 | -3.907 | -5.278 | +1.371 |
| 0.4-0.5 | scale_only | 1231 | -1.205 | -2.102 | +0.897 |
| 0.5-0.6 | scale_only | 1423 | +1.335 | +1.359 | -0.025 |
| 0.6-0.7 | scale_only | 1587 | +3.968 | +3.830 | +0.138 |
| 0.7-0.8 | scale_only | 1356 | +6.838 | +7.494 | -0.656 |
| 0.8-0.9 | scale_only | 952 | +10.369 | +12.004 | -1.635 |
| 0.9-1.0 | scale_only | 153 | +14.533 | +16.176 | -1.643 |

## Full-sample research fit (not an out-of-sample score)

- scale_only: a=+0.0000, b=11.4470; all 9509 OOS WP/outcome pairs. Not deployed and not used to score any season above.
- affine: a=-0.4593, b=11.7280; all 9509 OOS WP/outcome pairs. Not deployed and not used to score any season above.

## Limits

This tests expected final margin, not a complete score/margin distribution, cover probabilities, or financial returns. A calibrated WP alone does not identify the mean margin. An affine mean mapping need not correspond to a Normal distribution whose win probability equals WP. Model choices predate this test but were developed retrospectively; walk-forward fitting is not an untouched prospective holdout. Availability routing is concentrated in recent seasons, so route comparisons are also era comparisons. The 2019-20 bubble season is excluded, as in the existing history backtest. Subgroup tables are exploratory, not selection rules.
