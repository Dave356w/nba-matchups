# Direct v6 iterations — 2026-10-11

Built and tested 41 standalone alternatives to v6 in two passes. No market price is a predictor. The strongest leads improve v6 slightly but do not demonstrate better log loss than the market or a repeatable profitable betting rule. Keep production v6 unchanged.

## Same-row results

Lower log loss is better. Each season/market below contains 1,209 reconstructed regular-season games. 2025 means 2024–25; 2026 means 2025–26. The v6 comparator is chronological walk-forward, not the ledger’s leave-one-season-out probability.

| Candidate | 2024–25 ESPN BET LL | 2025–26 Kalshi LL | Kalshi gap |
|---|---:|---:|---:|
| archived_v6 | 0.595505 | 0.583660 | +0.015920 |
| projected_r10 | 0.593699 | 0.582639 | +0.014899 |
| phase_projected_r10 | 0.592862 | 0.582260 | +0.014520 |
| phase_stale_r10 | 0.593164 | 0.582115 | +0.014375 |
| stage_stale_r10 | 0.595310 | 0.581449 | +0.013709 |
| Market | 0.582108 | 0.567740 | — |

`projected_r10` was selected using only the first-pass 2024–25 log losses. Its 2025–26 gain versus v6 is −0.001021 (paired weekly bootstrap 95% CI −0.003354 to +0.001267). This selector preserves the season boundary, but the research menu was designed retrospectively; it is not prospective evidence.

The second pass combines the observed first-pass leads and therefore reuses both evaluation seasons. Its best 2025–26 model, `stage_stale_r10`, gains −0.002211 versus v6 (95% CI −0.004504 to +0.000047). Its 2024–25 gain is only −0.000196. `phase_projected_r10` has the larger two-season point improvements: −0.002644 / −0.001400, both unresolved. None is a confirmed upgrade.

## ROI at closing quotes

One unit per bet. Lean bets every game on P ≥ 0.5; EV rules choose the side with the larger P × posted payout − 1 and bet only when that EV exceeds the threshold. ROI is net units / bets. Kalshi posted odds include the repository’s fee approximation. The market-null ROI is the same bets’ expected profit under the no-vig market probability. The favourite baseline uses exactly those selected games and their posted prices.

| 2025–26 Kalshi candidate | Rule | Bets | ROI | 95% CI | Market null | Favourite ROI |
|---|---|---:|---:|---:|---:|---:|
| archived_v6 | lean | 1209 | -0.88% | -5.59% to +3.99% | -3.18% | -3.30% |
| archived_v6 | ev0 | 952 | -12.04% | -20.23% to -3.34% | -6.04% | -3.71% |
| projected_r10 | lean | 1209 | -0.18% | -4.52% to +4.26% | -3.17% | -3.30% |
| projected_r10 | ev0 | 933 | -11.92% | -20.64% to -2.70% | -6.16% | -2.64% |
| phase_projected_r10 | lean | 1209 | +0.72% | -3.99% to +5.54% | -3.16% | -3.30% |
| phase_projected_r10 | ev0 | 932 | -11.59% | -20.33% to -3.15% | -6.18% | -3.00% |
| phase_stale_r10 | lean | 1209 | +0.36% | -4.35% to +5.22% | -3.17% | -3.30% |
| phase_stale_r10 | ev0 | 941 | -11.37% | -20.12% to -2.69% | -6.16% | -2.73% |
| stage_stale_r10 | lean | 1209 | -0.13% | -5.01% to +4.59% | -3.16% | -3.30% |
| stage_stale_r10 | ev0 | 934 | -10.86% | -19.15% to -2.69% | -6.12% | -2.04% |

No candidate/rule produced positive ROI in both seasons at the respective ESPN BET / Kalshi prices. All tested Kalshi EV-filtered rules lost money in 2025–26. The strongest blanket-lean point ROI is +0.72% for `phase_projected_r10`, but its interval spans losses and its 2024–25 lean ROI is negative. A lower global log loss does not guarantee calibration of the model–market disagreements that drive selective bets.

## Sportsbook cross-check

| 2025–26 book | Games | v6 LL | Selected projected_r10 LL | Market LL |
|---|---:|---:|---:|---:|
| dk | 964 | 0.588548 | 0.587435 | 0.570186 |
| espnbet | 245 | 0.564426 | 0.563770 | 0.563087 |

The small ESPN BET early-calendar segment is unresolved; isolated tiny advantages there do not establish an edge across the main season. Kalshi and sportsbook rows often represent the same games and are not independent replications.

## Build and validation

- First pass: nine mixed shooting/possession/FT decay profiles (15/25/40 game half-lives), four standardized ridge strengths, three seasonal recency half-lives, phase/stage interactions, injury-adjusted talent, separate stale-talent terms, injury interactions, and three carryover weights.
- Second pass: eight higher-penalty fits plus four combinations of temporal and injury/talent changes.
- Every fold fits on seasons strictly before the evaluated season. Train-only standard deviations are used for ridge. Features and outcome labels for the evaluated season never enter its fit. Raw team prefixes exclude the current game; the feature builder checks date boundaries.
- Games 1–9 keep original carryover forecasts for every midseason variant. Carryover variants change only that early route. Missing feature rows retain the archived production fallback.
- Baseline probability reproduction: maximum error below 9e−16 for both seasons and early routes. Raw feature recomputation matches all exported training/test tables within 6.1e−13, including finite/missing masks.
- First-pass selection is saved in selection.json. All fits, predictions, book/season/early-versus-midseason scores, all ROI thresholds (0%, 2.5%, 5%, 10%), nulls, favourite baselines and weekly 95% intervals are saved beside this report.

## Limits and next decision

These are reconstructed, retrospective results. No native or preseason rows are included. The 95% intervals are descriptive and unadjusted for testing many variants; selected winners are optimistic. Two seasons cannot settle small improvements. Closing quote ROI does not reconstruct decision-time price, executable liquidity, slippage or exact contract-count fee rounding.

Mixed-decay variants keep the fold’s original prior-season composite factor weights; they refit the base/availability logits rather than relearning factor weights for each decay menu. Availability and roster windows remain at 25 games. The projected-talent arm approximates pregame roster changes using the archived injury report, not actual participation. This is a bounded screen, not an exhaustive search.

All 1,071 midseason evaluation games in each exported fold were on the availability route. Base-route logits were trained but have no independent held-out route evidence in this sample.

Next: freeze one injury/talent + phase challenger for shadow forecasts on native games, and broaden outer-season evaluation before promotion. The larger unresolved limitation is predicting the market’s team-strength correction; none of these variants closes most of that gap.

## Reproduce

```bash
python research/export_v6_inputs.py
python research/v6_direct_iterations.py --inputs research/output/v6_inputs
python research/v6_direct_grading.py
python validate_data_files.py
python -m pytest tests/ -q
```

The research-only Actions workflow supplies historical caches and runs the same pipeline. Initial portable-input export: [run 38123453021](https://github.com/Dave356w/nba-matchups/actions/runs/38123453021), artifact 11695898849. Provenance and checksums are in protocol.json. No production model, model tag, bot ledger, or site files are changed.
