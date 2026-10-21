# Median margin vs DraftKings: 2025–26 forecast benchmark

Frozen training: **8,438** earlier-season OOS probability/outcome pairs, last BBR year 2025. Held-out common sample: **964 games**, 2025-11-25 to 2026-04-12.

Two fixed 10-percentage-point lookup candidates: actual home-margin medians by home WP, and favorite-margin medians by favorite WP (converted back to home margin). Both lookups are fitted only on seasons before 2025–26. No bands, thresholds, or candidates are selected using holdout performance. This is retrospective research: prior analysis had already inspected this season, so chronological training exclusion is not an untouched prospective holdout.

## Held-out forecast accuracy

MAE is the primary median-forecast score (equivalent to twice the 0.5-quantile loss). RMSE and average bias are secondary diagnostics. All values are final home-minus-away points, including overtime. Market forecast margin = minus the quoted home handicap.

| Forecast | Games | MAE | RMSE | Mean bias | Median error |
|---|---:|---:|---:|---:|---:|
| dk_close | 964 | 11.0560 | 14.1410 | +0.3205 | -0.5000 |
| dk_open | 964 | 11.3019 | 14.4319 | +0.2842 | +0.5000 |
| favorite_band_median | 964 | 11.4979 | 14.6895 | +0.0228 | +0.0000 |
| fixed_13_5 | 964 | 11.3934 | 14.5812 | +0.4304 | -0.2503 |
| home_band_median | 964 | 11.5166 | 14.7134 | -0.0934 | +0.0000 |

## Paired uncertainty

Candidate minus benchmark; negative favors the candidate. Paired date-cluster bootstrap: 2,000 draws, seed 20261021; 95% percentile intervals. Intervals do not capture cross-date serial dependence, model-development selection, or multiple-comparison effects.

| Candidate | Benchmark | Delta MAE [95% CI] | Delta RMSE [95% CI] |
|---|---|---:|---:|
| home_band_median | fixed_13_5 | +0.1232 [-0.0073, +0.2614] | +0.1322 [-0.0038, +0.2782] |
| home_band_median | dk_open | +0.2147 [+0.0021, +0.4343] | +0.2815 [+0.0539, +0.5105] |
| home_band_median | dk_close | +0.4606 [+0.2447, +0.6801] | +0.5724 [+0.3394, +0.8129] |
| favorite_band_median | fixed_13_5 | +0.1045 [-0.0113, +0.2223] | +0.1084 [-0.0164, +0.2335] |
| favorite_band_median | dk_open | +0.1961 [-0.0032, +0.4009] | +0.2576 [+0.0369, +0.4883] |
| favorite_band_median | dk_close | +0.4419 [+0.2456, +0.6532] | +0.5485 [+0.3237, +0.7813] |

## Marginal median calibration

For a calibrated median with discrete scores, P(actual < forecast) ≤ 50% ≤ P(actual ≤ forecast). A 50% value inside that interval is a necessary aggregate diagnostic, not proof of conditional calibration. Exact equality is reported separately.

| Forecast | Actual below | Equal | Actual at or below |
|---|---:|---:|---:|
| dk_close | 49.07% | 0.00% | 49.07% |
| dk_open | 50.10% | 0.00% | 50.10% |
| favorite_band_median | 47.61% | 3.01% | 50.62% |
| fixed_13_5 | 49.69% | 0.00% | 49.69% |
| home_band_median | 46.89% | 3.42% | 50.31% |

## Frozen earlier-season lookup

Rare bands are left visible, not silently combined. Empty-band fallback uses the nearest nonempty training band; no evaluated game needed it in this experiment.

| Perspective | WP band | Training games | Training median margin |
|---|---|---:|---:|
| home | 0–10% | 16 | -10.50 |
| home | 10–20% | 212 | -10.00 |
| home | 20–30% | 562 | -9.00 |
| home | 30–40% | 877 | -6.00 |
| home | 40–50% | 1216 | -2.00 |
| home | 50–60% | 1421 | +3.00 |
| home | 60–70% | 1625 | +5.00 |
| home | 70–80% | 1358 | +7.00 |
| home | 80–90% | 969 | +11.00 |
| home | 90–100% | 182 | +13.50 |
| favorite | 50–60% | 2637 | +2.00 |
| favorite | 60–70% | 2502 | +5.00 |
| favorite | 70–80% | 1920 | +8.00 |
| favorite | 80–90% | 1181 | +11.00 |
| favorite | 90–100% | 198 | +13.00 |

## Descriptive holdout by home WP band

Holdout outcomes below are diagnostics, not refitted forecast values. Market medians summarize varying game-specific lines inside each band; the accuracy table scores every original line.

| Home WP | Games | Frozen home-band forecast | Actual median | DK opening median | DK closing median |
|---|---:|---:|---:|---:|---:|
| 0–10% | 1 | -10.50 | -21.00 | -21.50 | -21.50 |
| 10–20% | 55 | -10.00 | -12.00 | -12.50 | -13.50 |
| 20–30% | 80 | -9.00 | -9.00 | -8.50 | -8.50 |
| 30–40% | 113 | -6.00 | -7.00 | -4.50 | -5.50 |
| 40–50% | 136 | -2.00 | -4.00 | -2.50 | -2.50 |
| 50–60% | 157 | +3.00 | +2.00 | +1.50 | +2.50 |
| 60–70% | 149 | +5.00 | +6.00 | +5.50 | +5.50 |
| 70–80% | 145 | +7.00 | +9.00 | +8.50 | +8.50 |
| 80–90% | 105 | +11.00 | +14.00 | +13.50 | +13.50 |
| 90–100% | 23 | +13.50 | +19.00 | +16.50 | +17.50 |

## Provenance and limitations

- DraftKings reconstructed rows: 964; complete DK opening/closing lines and scores: 964; matched OOS rows: 964; unmatched complete DK rows: 0.
- Probability-model fits and composite weights precede their own seasons. The median lookup's latest training year is 2025, before test year 2026.
- Final scores agree exactly across sources. Maximum difference between reconstructed CSV WP and archived OOS WP: 0.00004879, within four-decimal rounding.
- Matched route counts: {'avail': 964}. Empty-band fallback games: home=0, favorite=0.
- Actual reconstructed snapshot timestamps: 0. The repo archives opening and closing endpoints, not a timestamp-aligned historical model prediction at each opening line. Model WP can include pregame information unavailable at the opening endpoint; the opening comparison is not an equal-information-time experiment.
- This is forecast accuracy and midpoint calibration only. It does not estimate a full conditional margin distribution or validate predictions at arbitrary margin thresholds.
- Games before both teams have 10 prior games, the first month without DK coverage, and 2019–20 remain outside the relevant source samples. Training mixes historical model routes and eras; the target season's routing mix may differ.
