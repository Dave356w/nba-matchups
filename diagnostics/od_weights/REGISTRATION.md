# Offence/defence weights refitted on game margins: registration

Owner's request, 2026-11-06. Registered before any code for this test or any
result from it. Research only: no model, rule, threshold, ledger or page changes.

## Why

In the interaction-weighting study ([PR #150](https://github.com/Dave356w/nba-matchups/pull/150),
`diagnostics/interaction_weight/`), the OD arm, which adds the offensive half of the
composite gap (`delta_off`) to the production margin fit, had a negative coefficient
in all 9 walk-forward folds. The game margin wanted less weight on offence than the
composite's team-season ridge gives it. OD improved on both priced book-seasons
(DraftKings 2025-26 games 10+ −0.0014, ESPN BET 2024-25 −0.0015, Brier agreeing)
but failed G7 on outcomes (games 1+ −0.0004 ± 0.0005, better in 6 of 9).
Those seasons (2016-17 .. 2025-26) have now been seen, so they cannot be this
test's primary evidence.

## Arms (fixed)

The composite C = O + D splits exactly into an offensive half O (off eFG%, TOV,
ORB%, FTr at the production weights) and a defensive half D (the four opponent
factors).

- **base**: production terms (v10h25 config: ORB%, half-life 25, the v8 fade).
- **OD2** (the candidate): base + `delta_off` = O_h − O_a and `doff_phase` =
  delta_off × phase. Offence and defence each get their own slope and their own
  growth through the season, fitted on the final margin by least squares with the
  other production terms. Implied offence share = (s + k) / (2s + k) at a phase,
  where s is the slope of delta and k the slope of delta_off at that phase.
- **GW8** (description only, not in the rule): the eight factor gaps
  (f_h − f_a) / sd fitted directly on the margin, in place of delta, plus
  delta × phase and the other terms. It shows which factors move.

## Evidence

1. **Primary, out of era: 2006-07 .. 2013-14 (8 test seasons).** No fit or design
   choice has examined the offence/defence split on these seasons (History backtest
   Part B scored only the frozen core there). Walk-forward: before each test season
   Y, the composite weights are refitted (ridge, ORB%) on the era's team-seasons
   2004-05 .. Y−1 and the margin OLS on the era's game seasons 2005-06 .. Y−1.
   Base route, games 1+ (both teams past game 0; game 0 excluded). No box scores or
   injury reports exist there, so talent_diff and the report route are absent for
   every arm alike; luck_def and ft_diff come from the game logs.
2. **Secondary, in era: 2016-17 .. 2025-26**, production routing as PR #150
   (already seen; description, and the G7 market check).

## Decision rule (fixed)

OD2 is a candidate for a new model tag only if all of these hold:

- Out of era, games 1+: OD2 − base log loss below zero beyond the 95% interval,
  and better in at least 6 of the 8 seasons.
- Out of era: the fitted offence share is below 0.5 at phase 1 in at least 6 of
  the 8 folds (the direction replicates).
- In era: OD2 is not worse than base on reconstructed DraftKings 2025-26 and ESPN
  BET 2024-25, games 10+, with Brier agreeing.
- No season gains more than 0.008 (leakage tripwire).

Passing makes OD2 a candidate, the owner's decision; adoption would need its own
amendment and the usual post-change checks (rescore, rebuild, build, model review,
DraftKings and History backtests). Failing closes the question unless new evidence
appears. Every result is reported. ± is 95%, game-level, paired.

## Amendment (owner's decision, 2026-11-06, before any code ran or any result of this test)

The owner directs the test onto seen data: **2016-17 through today, minus the COVID
seasons, is the new era of the game.** This replaces the out-of-era evidence above.

- **Era seasons:** 2016-17, 2017-18, 2018-19, 2021-22, 2022-23, 2023-24, 2024-25,
  2025-26 and 2026-27 to date (BBR years 2017, 2018, 2019, 2022–2027). 2019-20 (the
  bubble) and 2020-21 (mostly empty arenas) are left out of every game-level fit
  and every test season.
- **Walk-forward:** each test season Y (2018, 2019, 2022 .. 2027) is scored by fits
  on the era seasons before Y only: the base-route margin OLS on the era's game
  seasons, the report route (refitted before each slate) on the report seasons
  before Y plus Y's covered games so far, as in production. Composite weights keep
  the production protocol (ridge on WEIGHT_YEARS before Y), so only the game-level
  weighting differs between arms. Production routing, v10h25 config, talent included.
- **These seasons were seen** in PR #150, where OD (delta_off alone) was better on
  both priced books but failed G7 on outcomes. The owner chose this knowingly; the
  result is not independent of that one.
- **Decision rule (replaces the one above):** OD2 is a candidate only if
  1. games 1+ over the completed test seasons (2018, 2019, 2022–2026): OD2 − base
     log loss below zero beyond the 95% interval, and better in at least 5 of those 7
     seasons;
  2. the fitted offence share at phase 1 is below 0.5 in at least 5 of the 7
     completed folds (base route);
  3. not worse than base on reconstructed DraftKings 2025-26 and ESPN BET 2024-25,
     games 10+, with Brier agreeing;
  4. no season gains more than 0.008 (leakage tripwire).
  2026-27 to date is reported, apart, and not counted. GW8 stays description only.
