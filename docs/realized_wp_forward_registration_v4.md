# Realized WP shrunk cell + drift shadow forward record — v4

Registered 2026-11-13T20:03:04Z, on the owner's instruction ("v4 patch please") after asking
whether Realized WP can stay on the latest 200 cell results and still respond
to the live locks, committed on its own before any v4 code or capture.
Algorithm version: `realized_wp_shrunk200_drift_v4`. It supersedes
`realized_wp_rolling200_bandadj_v3` (docs/realized_wp_forward_registration_v3.md)
for new captures and for the card display. Everything not restated here is
unchanged from v3: cells, the latest-200 cell window, seeding, pooling of
early and 10+ evidence, exclusions, ordering, capture, the official
observation, frozen fields, scoring, the +EV flat 1u paper rule, benchmarks,
scope and persistence.

## Why

The owner observed that a 200-result cell is 86 to 95 percent last season's
replay, so the live five-hour locks barely move it. The shrunk cell + drift
backtest (`research/shrunk_drift_backtest.py`,
diagnostics/shrunk_drift/RESULTS.txt; same walk-forward and cohorts as the
matrix window backtest) scored the candidate below against v3 and the price:

| Cohort | v3 − price | v4 − price | v4 − v3, paired |
| --- | --- | --- | --- |
| 2025-26 DraftKings replay, 963 games (primary) | +0.0024 ± 0.0022 | −0.0009 ± 0.0012 | −0.0033 ± 0.0019 |
| 2024-25 ESPN BET cold start, 1,187 games | +0.0113 ± 0.0061 | +0.0007 ± 0.0007 | −0.0106 ± 0.0058 |
| 2026-27 native locks, 98 games (information only) | −0.0046 ± 0.0074 | −0.0070 ± 0.0072 | −0.0024 ± 0.0060 |

Brier differences, ± 95%, negative = better. v4 is better than v3 on both
replay seasons and beats the price on neither. Nearly all of the gain is the
shrinkage; the drift alone is −0.0012 ± 0.0009 on 2025-26 and +0.0004 ±
0.0004 on 2024-25. The largest single adjustment falls from 32.1 points to
6.4 (71.7 to 4.7 on the cold start). The constants below were picked on
2026-11-13 from a small grid scored on these same replay seasons, and the
question was raised after a 75–23 live run, so this selection is in-sample
and outcome-prompted; the v4 forward record is its out-of-sample test.

Tried the same day and not adopted (hindsight, same seasons, exploratory
runs whose scripts are not in the repository): a latest-50 window, per-cell
exponential weighting, season-phase-matched cells, cells shrunk toward the
previous season's same half, and current-season-only cells with a
phase-matched market term. None beat the price at 95%, and none scored
better than v4.

## Estimate

    cell  = (wins/n − mean(q_member)) × n / (n + 200)
    drift = Σ wᵢ (wonᵢ − qᵢ) / (Σ wᵢ + 200),   wᵢ = 0.5 ^ (ageᵢ / 100)
    Realized WP = clip( q_game + cell + drift, 0, 1 )

- **cell**: v3's band adjustment over the latest 200 results in the joint
  model-WP × market-WP cell (n = results held, all of them when fewer than
  200), multiplied by n/(n + 200): 200 pseudo-games at zero. An empty cell
  has no estimate, as before.
- **drift**: one number per slate, the same for every game. It reads the
  latest 600 picks of **every cell and every stage** (openers included)
  from the same evidence the cells draw on: valid graded native v11
  five-hour DraftKings locks of the current season and the immediately
  previous season's replay at the DraftKings close, strictly earlier slates
  only, each oriented to its model's selected team. Order is slate date,
  then game ID; ageᵢ is 0 for the most recent pick and counts picks, not
  days. 200 pseudo-games at zero.
- A pick's result enters both its own cell and the drift. That overlap is
  intended and was present in the backtest.
- No other shrinkage, thresholds or tuning. The four constants (200, 100,
  200, 600) are fixed; changing any of them needs a new version.

## Reading, fixed now

Primary: official graded v4 observations of 2026-27, all stages pooled, the
paired Brier difference between Realized WP v4 and the saved no-vig
DraftKings probability on identical games, with a game-level 95% interval,
read once at the end of the 2026-27 regular season. v4 "beats the price" only
if that interval lies below zero. Otherwise the reading is that the estimate
adds no predictive value to the price, whatever its paper return. Interim
figures are descriptive. The comparison with v3 cannot be made forward (v3
stops capturing) and rests on the backtest above.

## Record

- v3 closes when v4 is deployed (production keeps saving v3 until then). At
  registration, on `c485f64`, the shadow file holds 27 v3, 60 v2 and 4 v1
  captures. They stay unchanged, graded under their own versions and never
  pooled with v4. Nothing is rescored or converted.
- v4 captures start with the first production build after deployment. No
  backfill, including for games that already hold a v3 observation; such a
  game may also receive its own v4 observation from a valid capture before tip.
- The snapshot schema is unchanged. Each v4 cell proof additionally records
  the shrinkage (200), the drift value and the ID of a content-addressed
  proof holding the drift's terms and pooled picks (date, game ID, season,
  source, result, no-vig probability); that proof is saved in the same
  evidence file, once per distinct pool. Validation recomputes the drift
  from it and fails closed on a mismatch or on non-causal picks.

Shadow research only. The public card and guide show the v4 estimate under
the same name, with the shrunk cell adjustment and the drift stated on the
card. A positive drift raises every card of a slate together; it is a
description of recent results against prices, not an edge claim. The ROI
matrix and its window switch, the model, locks, prices, grades and public
records are unchanged.
