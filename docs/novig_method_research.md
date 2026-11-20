# No-vig method comparison — research terms

Owner request, 2026-11-20: compare the production proportional no-vig method
with Shin and power across the records and Realized WP, before any decision
to change the method. Terms fixed here and committed on their own before the
comparison script exists or any of its figures are computed.

Disclosed beforehand: on 2026-11-20, in chat, the cell-level ROI null under
the three methods was computed for the card cells (latest 200, games 1+).
Power and Shin lowered the favourite cells' ΔROI by about 1–3 points. No log
loss, Brier, record or Realized WP figure under Shin or power has been seen.

Research only. Nothing here changes `market.devig`, any stored `*_q_home`
column, a record, a rule's membership, Realized WP, a lock, a grade or a
registration. Any adoption needs its own owner amendment.

## Methods, fixed now

For one two-way moneyline pair with raw implied probabilities a (home) and
b (away), overround O = a + b:

- **Proportional** (production, `market.devig`): q = a / O.
- **Power:** q = a^k, with k solved so that a^k + b^k = 1.
- **Shin:** q = (√(z² + 4(1 − z)a²/O) − z) / (2(1 − z)), with z in [0, 0.4)
  solved so the two sides sum to 1.

Each is solved per game; nothing is fitted to outcomes. A pair with O ≤ 1
(no hold) takes the proportional value under every method and is counted.

## Primary reading, fixed now

DraftKings 2025-26 closing moneylines (`data/nba_reconstructed.csv`,
basis reconstructed, close book DraftKings), every graded regular-season game
with a usable pair, scored on the home team: paired log loss, Shin minus
proportional and power minus proportional, with a game-level 95% interval.
A method counts as better calibrated than proportional only if its interval
lies wholly below zero. Two comparisons are read; neither interval is
adjusted for the other, and that is stated with the result.

## Secondary, descriptive (±95%)

- The same paired log loss and Brier on ESPN BET 2024-25 and 2025-26 closes
  and on 2026-27 native DraftKings five-hour locks and closes (graded,
  voided locks out). Book × season × basis × price timing stay separate.
- Favourite calibration: favourites by proportional no-vig band, win rate
  against each method's mean probability, primary cohort.
- Records: the public games 1+ model-prediction ML record (saved live locks;
  2025-26 historical replay separately), ROI, its no-vig null and z under
  each method; the model-minus-market log loss on the same games.
- Rule membership, information only: how many H2·F·5 and O/U lock bets to
  date would enter or leave under each method. Membership stays as
  registered; nothing is regraded.
- Realized WP: the shrunk-drift backtest harness
  (`research/shrunk_drift_backtest.py`) rerun with each method's q for the
  cells, the drift and the base price; the public estimate (shrunk cell, no
  drift) and v4 (with drift) minus each method's own market, and paired
  Brier of the public estimate under each method minus proportional on
  identical games. Cohorts as that script: 2025-26 DraftKings replay
  (primary there), 2024-25 ESPN BET, 2026-27 native locks.
- Card cells: the latest-200 games 1+ cell ROI null under each method.

Script `research/novig_compare.py`; output
`diagnostics/novig_compare/RESULTS.txt`.
