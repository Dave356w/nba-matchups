# Day of the week in the totals model (ou2): registration

Owner's request, 2026-11-06. Registered before any code for this test or any further
look at the data. Research only: no model, rule, threshold, ledger or page changes.

## Why, and what has been seen

A descriptive look at ou2's walk-forward predictions (from the totals O/D test,
`diagnostics/totals_od/`, games 1+, 2017-18 .. 2025-26 minus the COVID seasons) found
totals above the model on Wednesday and Thursday and below it on Sunday and Monday.
Wed–Thu minus Sun–Mon was positive in all 7 seasons (+0.8 to +2.2 points, mean about
+1.6). **That contrast was chosen after seeing the pooled table**, so these seasons are
not independent evidence for it. Against both teams' days of rest, Wed and Thu stayed
about +1.2 and +1.5 points above Monday, but the 7-day joint test weakened (F = 1.23).
Against the closing total, ESPN BET 2024-25 showed the same direction and DraftKings
2025-26 mostly did not.

## Arms (fixed)

All share ou2's pipeline in `research/totals_od.py` (T* stage by least squares on
total − league total, games 1+, then the report stage refitted on each arm's residual).
Days of rest are computed from each team's schedule within the season (days since its
previous game minus 1, capped at 3; a team's first game counts 3). The day is the ET
slate date.

- **base**: ou2 (rel, form, b2b, phase).
- **REST** (the control): base + `rest_tot` = home rest + away rest.
- **DOW** (the candidate): REST + `midweek` (1 on Wednesday or Thursday) + `sunmon`
  (1 on Sunday or Monday).

The rule compares **DOW with REST**, so it measures the day beyond rest. REST against
base is reported as information.

## Evidence

The owner's new era: 2016-17 .. today minus 2019-20 and 2020-21. Walk-forward: each
test season Y (2018, 2019, 2022 .. 2026) is predicted from fits on era seasons before Y
only. 2026-27 to date is reported apart and not counted.

**Forward (information, reported with the result and as it accumulates):** the native
2026-27 O/U locks (`data/nba_ledger.csv`, the tracked OU rows), results by Wed–Thu,
Sun–Mon and other days, against the lock line. This is the only source the original
look did not touch.

## Decision rule (fixed)

DOW is a candidate for a new totals tag only if all hold:

1. **Squared error on the final total, games 1+, the 7 completed seasons:** DOW − REST
   below zero, one-sided p < 0.05, and better in at least 5 of the 7 seasons.
2. **P(over) log loss at the close, main lines, games 10+,** reconstructed DraftKings
   2025-26 and ESPN BET 2024-25: DOW not worse than REST on either, point estimate.
3. **Leakage tripwire:** no season improves by more than 10 points².

Power is reported (minimum effect detectable with 80% power, 2.49 × SE). Fitted
`midweek` and `sunmon` coefficients are reported per fold. Passing makes DOW a
candidate, the owner's decision; adoption would need its own amendment, a new totals
tag and the usual post-change checks. Because the contrast was chosen on these
seasons, a pass is weaker evidence than its p-value suggests; the forward 2026-27 locks
carry the independent weight. Every result is reported.
