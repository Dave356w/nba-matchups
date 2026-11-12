# Rolling-50 retrospective research

- Exploratory retrospective replay, separate from immutable forward record.
- Prices are historical closes, not first valid snapshots within five hours.
- Base models are leave-one-season-out; earlier test years can use later training seasons.
- No prior-season same-book seed exists for either primary cohort; cold-start cells abstain.
- 2024-25 ESPN BET is a separate bookmaker sensitivity, not DraftKings evidence.
- Replay does not verify execution, publication or visitor views. No thresholds/windows optimized.

| Cohort | Covered / available | Estimator | Brier | Log loss | +EV bets | Units | ROI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-25 espnbet | 1187 / 1225 | model | 0.2033 | 0.5919 | 886 | -33.95 | -3.8% |
| 2024-25 espnbet | 1187 / 1225 | market | 0.1996 | 0.5825 | 0 | 0 | — |
| 2024-25 espnbet | 1187 / 1225 | realized v1 (raw cell rate) | 0.2128 | infinity | 907 | -89.07 | -9.8% |
| 2024-25 espnbet | 1187 / 1225 | realized v2 (band-adjusted) | 0.2118 | infinity | 871 | -44.91 | -5.2% |

2024-25 espnbet: contradicted 0%/100% estimates v1 13, v2 8; median cell size 50.
Slate bootstrap 95% intervals, v1: {"brier_delta_vs_market_95": [0.007542908660347688, 0.019359971352555738], "realized_roi_95": [-0.17145287142461677, -0.02661583577484822]}
Slate bootstrap 95% intervals, v2: {"brier_delta_vs_market_95": [0.006684120798610118, 0.017932184729620482], "realized_roi_95": [-0.13129052416271994, 0.02976894110970463]}

| Cohort | Covered / available | Estimator | Brier | Log loss | +EV bets | Units | ROI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-26 dk | 941 / 964 | model | 0.2009 | 0.5861 | 739 | -85.40 | -11.6% |
| 2025-26 dk | 941 / 964 | market | 0.1951 | 0.5712 | 0 | 0 | — |
| 2025-26 dk | 941 / 964 | realized v1 (raw cell rate) | 0.2125 | infinity | 771 | -46.60 | -6.0% |
| 2025-26 dk | 941 / 964 | realized v2 (band-adjusted) | 0.2091 | infinity | 775 | -38.26 | -4.9% |

2025-26 dk: contradicted 0%/100% estimates v1 17, v2 8; median cell size 42.
Slate bootstrap 95% intervals, v1: {"brier_delta_vs_market_95": [0.008443344225708865, 0.026827993326419573], "realized_roi_95": [-0.15770559721525876, 0.0441699761848448]}
Slate bootstrap 95% intervals, v2: {"brier_delta_vs_market_95": [0.0054201359459077805, 0.023228596127394696], "realized_roi_95": [-0.15143746649030967, 0.06050411262808631]}

No historical observation is added to the shadow forward files or public records.
Reproduce: `python research/realized_wp_replay.py`. Causal evidence selection and input-order checks passed.
