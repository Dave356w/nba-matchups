# Offence/defence weights in the totals model (ou2): registration

Owner's request, 2026-11-06. Registered before any code for this test or any result
from it. Research only: no model, rule, threshold, ledger or page changes.

## Why

The margin test (`diagnostics/od_weights/`) found defence predicting margins better
than the composite assumes (offence share 0.25–0.33 at phase 1 in 6 of 7 folds), but
OD2 failed its registered rule: −0.0008 ± 0.0009, about 44% power at that effect.
The totals model has never been tested for an unequal split. Its expected total
(`totals.py`, `Season.features`) is

  naive = (pace_h × pace_a / league pace) × (ORtg_h × DRtg_a + ORtg_a × DRtg_h) / league rtg / 100

so a team's offence and its opponent's defence move the total by the same proportion.

## Decomposition (exact)

With r the league rating to date, a = ORtg − r (offence above league) and
d = DRtg − r (points allowed above league; positive = weaker defence),
P = pace_h × pace_a / league pace:

  naive = P/100 × [2r + (a_h + a_a) + (d_h + d_a) + (a_h d_a + a_a d_h) / r]

- `off_t` = P/100 × (a_h + a_a): both offences against league-average defence.
- `def_t` = P/100 × (d_h + d_a): both defences against league-average offence.

The production term `rel` = naive − league total contains both with equal weight.

## Arms (fixed)

- **base**: ou2 as in production. T* = league total + a + b·rel + c·form + d·b2b +
  e·phase by least squares (games 1+), then ou2's report stage (av_o, av_d, av_m from
  the last NBA report ≥ 300 min before tip, v11's Questionable weighting) fitted on
  T*'s residual.
- **ODT** (the candidate): base + `off_t` + `off_t × phase` in the T* stage. Offence
  and defence each get their own weight and their own growth through the season. The
  report stage is refitted on ODT's own residual. Implied offence share at a phase =
  (b + k) / (2b + k), with b the rel slope and k the off_t slope at that phase.

## Evidence

The owner's new era: 2016-17 .. today minus 2019-20 and 2020-21 (BBR years 2017,
2018, 2019, 2022–2027). Walk-forward: each test season Y (2018, 2019, 2022 .. 2026;
2026-27 to date apart, not counted) is predicted from fits on era seasons before Y
only. The report stage, for Y ≥ 2024, is fitted on the report seasons before Y
(2022-23 on); without an earlier report season both arms use T* alone. Inputs read
games strictly before each slate. These seasons were used to fit and select ou1/ou2;
ODT itself has never been run on them.

## Decision rule (fixed)

ODT is a candidate for a new totals tag only if all hold:

1. **Squared error on the final total, games 1+, the 7 completed seasons:** ODT − base
   below zero, **one-sided** (the 90% two-sided upper bound below zero, i.e.
   one-sided p < 0.05), and better in at least 5 of the 7 seasons.
2. **The O/U test's measure:** P(over) log loss against the book's no-vig over price
   at the close, main lines (both prices −135..+115), games 10+, reconstructed
   DraftKings 2025-26 and ESPN BET 2024-25: ODT not worse than base on either, point
   estimate.
3. **Leakage tripwire:** no season's squared error improves by more than 10 points²
   (about 3% of the total's variance).

The fitted offence share is reported per fold (direction only, not in the rule).
**Power** is reported with the result: the minimum effect detectable with 80% power
at the observed standard error (2.49 × SE, one-sided). Passing makes ODT a candidate,
the owner's decision; adoption would need its own amendment, a new totals tag and the
usual post-change checks. Every result is reported. ± is 95% unless labelled.
