# Early-season talent weight: walk-forward test registration

**Status: DRAFT, not registered.** It becomes the registration when the owner
approves it. The approving commit records the time; it changes only this file
and comes before any test code or run. Test version: `talent_early_v1`.

State of the data when drafted (2026-11-17): `main` at `0ac87e4`. The native
ledger there has 210 rows, its last lock at 2026-11-16T21:02:35Z. The
motivating audit is `diagnostics/loss_feature_audit/RESULTS.txt` (PR
Dave356w/nba-matchups#185).

## Question

When teams have played fewer than 10 games, should roster talent
(`talent_diff`) carry more weight than the single weight the model now uses
all season?

This is a research test only. A pass does not change the model. Adopting the
term needs its own amendment in CLAUDE.md, a new model tag, and the
post-change checks.

## Why now (selection data; none of it enters the test)

These readings come from 2026-27 native v11 five-hour locks, games 1+
(n = 129, picks 95–34; ± 95%).

- **Losses vs wins.** In losses, `talent_diff` pushed toward the pick by
  +0.67 points, against +1.70 in wins (difference −1.02 ± 0.59). Games 1–9
  alone: −1.05 ± 0.68.
- **Against results.** One feature at a time, results ask for 2.3× its
  current weight (+1.30 ± 1.02 extra multiples).
- **Against the market.** Relative to the model, the lock market weights
  talent 1.7× as heavily (+0.69 ± 0.18).
- **Calibration.** The model is too close to 50% on games 1–9:
  calibration slope 1.72 ± 0.87
  (`diagnostics/model_calibration/RESULTS.txt`).

These readings chose the hypothesis, so they cannot also test it. They are
excluded from every reading below.

### Earlier research this does not repeat (G8)

- **v7 early roster.** This added the games-10+ `talent_diff` to the old
  early logit. That logit had no player term, and the test was −0.0105 ±
  0.0087 on games 1–9.
- **v8 unified.** This fitted one talent weight across all games.
- **"talent × last season's share adds nothing"** (v8 unified).
- **In-season talent terms** (`talent_cur`, `curphase`, on/off, in-season
  BPM). These replaced or supplemented the frozen value with this season's
  play.

None of these let the frozen `talent_diff` weight differ by season phase on
the current faded-rating model. That is the one change tested here.

## Arms (fixed)

- **prod**: production v11 routing and features, refitted per fold.
- **tE**: prod plus one feature, `talent_early = talent_diff × e`, where
  e = max(0, 1 − g / 10) and g = min(gp_home, gp_away). The 10 is
  production's `fade_games`, not a tuned value. At game 0, e = 1; from game
  10 on, e = 0. The feature enters both routes' least-squares margin fits
  next to every existing feature. Nothing else changes:
  - other features, their construction and the 30-minute report lead;
  - the od rule, Questionable weighting, rookies, ORB% and half-life;
  - sigma, which is each fold's residual SD, as in production.

There are no other arms, and no multiple of the talent weight is chosen by
hand.

## Data and folds (fixed)

- **Inputs.** As production builds them: `nc.fade_v8_games` for the base
  features and `pav.season_rows` / `model/avail_train.csv` for the report
  route. Game logs and box scores are strictly before each date. The report
  matcher is the one the deployed files use; if the suffix fix is active by
  then, both arms use it.
- **Base route.** Each test season Y is fitted on the base fit's seasons
  before Y (from 2015-16; 2019-20 is not used). Test seasons: 2016-17 ..
  2025-26 games 1+, excluding 2019-20 and 2020-21.
- **Report route.** Each test season Y is fitted on the report seasons
  before Y (from 2022-23). Test seasons: 2023-24 .. 2025-26, covered games
  1+.
- **Production routing.** Each game is scored by the route production would
  use (report route when covered and g ≥ 1, otherwise base), for both arms
  on identical games. Game 0 is reported separately and is not in the pass
  rule.
- **Not used.** 2026-27 is not a test season and is never used in a fit.

## Readings (± 95%, paired by game, arm minus prod; negative = tE better)

1. **Primary.** Log loss against outcomes, games 1–9, pooled over the test
   seasons. Also by season, with the count of seasons where tE is better.
2. **Brier**, same games.
3. **Games 10+ against outcomes.** tE changes these only through the
   refitted shared coefficients, so this is a check that they don't get
   worse.
4. **G7 against the close, games 10+.** Reconstructed DraftKings 2025-26 and
   ESPN BET 2024-25.
5. **Games 1–9 against the close.** ESPN BET 2024-25 and 2025-26 (n ≈ 138
   each), for each arm minus the close. Information only, since these
   samples are too small for a rule.
6. **Fitted coefficients per fold**: `talent_early`, `talent_diff`, `delta`
   and `d_phase`, with each feature's correlation to `talent_early`.
7. **Information only, not in the pass rule.** tE replayed on the 2026-27
   native locks. These are the selection data.

## Pass rule (fixed before the run)

tE passes only if all of these hold:

- **(a)** Reading 1 is below 0, with the upper 95% bound below 0.
- **(b)** Brier (reading 2) has the same sign.
- **(c)** tE is better in at least half of the test seasons on games 1–9.
- **(d)** Games 10+ against outcomes (reading 3) and both G7 book-seasons
  (reading 4) are not worse. "Not worse" means a point estimate ≤ 0, or the
  lower 95% bound below 0 with |estimate| < 0.0005.
- **(e)** No test season's games 1–9 gain exceeds 0.008 (G7 leakage
  tripwire). If one does, the run is checked for leakage before any reading
  is quoted.

Every outcome is reported, pass or fail. A fail closes this question at the
pre-game view of talent; it is not followed by a tuned variant (another fade
length, a cap or a multiplier) without a new reason and a new registration.

## Outputs

- `research/talent_early.py` (walk-forward; nothing touches `data/`, `public/`
  or `model/*.json`).
- `diagnostics/talent_early/RESULTS.txt`, with the readings above in this
  order, the pass rule's verdict line by line, and the commit of this
  registration.
