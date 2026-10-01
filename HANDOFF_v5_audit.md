# Handoff: v5 audit checks (NBA team composite)

**For:** Claude Code, working in David's repo.
**Context:** A review of v5 (live since 2026-09-30, described in `MODEL.md`) raised five issues. Two of them (decision-time price, home-court drift) could make edges at the open look bigger than they are. That matters because H2–H4 rest on edges at the open and there are no native ledger rows yet. This file says what to check, in what order, and what not to touch. Items marked **David decides** need his sign-off.

## Ground rules

1. **v5 is live. Don't change it.** No edits to `model/*.json`, the routing in `build_site.score_game`, existing ledger fields, frozen rows, or the text of H2–H4.
2. **One production change is in scope.** Task A adds ledger columns. Show David the diff before merging.
3. **Where work goes:** `diagnostics/v5_audit/` (or the repo's existing diagnostics location). Write one script per task, plus `diagnostics/v5_audit/RESULTS.md`.
4. **Time order only.** Evaluate walk-forward by season: fit on earlier seasons, score the next. Features use only games before the slate date. In-sample numbers are not evidence.
5. **Candidates, not changes.** Anything that changes predictions is a v6 candidate. Fit it walk-forward and report per-season log loss as (candidate − v5) with a paired per-game SE. Never overwrite v5 files.
6. **Leakage tripwire.** Assume leakage, and audit before reporting, if a candidate does either of these:
   - beats the closing line;
   - improves on v5 by more than ~0.008 log loss in a season (about half the v5-to-close gap).
7. **Stop and report** if the code contradicts this file or a needed data source doesn't exist. Don't improvise around it.

## Task 0: Orientation (read-only)

Read `MODEL.md`, `build_site.py`, `ledger.py`, `model/*.json`, the H2–H4 pre-registration, and the walk-forward evaluation code. Answer these at the top of `RESULTS.md`:

a. Where the walk-forward predictions for 2024-25 (ESPN BET) and 2025-26 (DK) live, with their open and close prices. Can they be regenerated per season?
b. Is `model/weights.json` refit inside each walk-forward fold, or fit once on all seasons? If once, flag it: the composite has seen the seasons it is scored on.
c. The free-throw factor as coded: FTA/FGA or FTM/FGA.
d. How `av_min` and `av_bpm` compute each player's "usual share of games": the window and the weighting.
e. Snapshot and price timing:
   - when the first pregame snapshot runs relative to the opening price;
   - which source and book the grader uses for open and close;
   - whether that source can return a current price at snapshot time.
f. The intercept in `model/logit_avail.json`, next to `model/logit.json`'s 0.297. The availability logit was fit on report seasons only, so a lower intercept is an early sign of Task B.
g. The exact wording of H2–H4: which price they're scored against, the edge threshold, and any price-band conditions.
h. Opening night of 2026-27 from the repo's schedule data. It sets Task A's deadline.

## Task A: Log the decision-time price (deadline: before the first native ledger row)

**Why.** The ledger keeps open and close but not the price available when the first snapshot was written. That snapshot may use something published after the open, most obviously a later injury report that routes the game to the availability logit. If it does, "edge vs open" credits the model with news the open didn't have. Live prices can't be backfilled.

**Do.**
1. At the first pregame snapshot, record per game:
   - book;
   - UTC timestamp;
   - home and away American odds;
   - no-vig P(home), using the grader's de-vig method;
   - the timestamp of the injury report the snapshot used;
   - the route taken (early / avail / base / v4), if it isn't already stored.
2. Apply the same write-once rule as the existing first snapshot. Add new columns only.
3. Use the grader's book where possible. If the live source differs, record which.
4. If no price exists yet at snapshot time, write null plus a reason. Don't fill it from a later fetch.
5. If the pipeline has no live price source, stop and report options. Don't add a dependency on your own.

**David decides** whether to register an amendment so H2–H4 are also scored against the snapshot price. This has to happen before the first native row. Don't edit the hypothesis text.

**Done when:**
- a dry run on a test slate writes the new fields once;
- a second run leaves them unchanged;
- grading still writes only results and open/close.

## Task B: Home-court drift

**Hypothesis.** `logit.json` pools 2016–19, when home teams won roughly 58–59%, with 2021–26. σ(0.297) = 57.4% home at even strength. An earlier analysis found a 54.6% home-only baseline on recent seasons. If the pooled intercept is 2–3 points high for current seasons, home-side edges are inflated by about what H2–H4 need.

**Do.**
1. For every season, games 10+, split by route, report:
   - n;
   - actual home win rate;
   - mean predicted P(home);
   - the difference and its binomial SE.
2. Fit these candidates walk-forward. Season dummies can't predict a new season, so use time weighting:
   - refit only the intercept on the last k ∈ {1, 2, 3} seasons, holding the other coefficients at their pooled values as an offset;
   - refit the whole base logit with game weights 0.5^(seasons ago / h), h ∈ {1, 2, 3}.

   Report all variants. Don't select one on the walk-forward test seasons.
3. For both walk-forward seasons, count the games that cross the H2–H4 edge threshold under v5 and under each candidate, split home/away.

**Decision rule.** Recommend the best variant as a v6 candidate if both of these hold, pooled over the last 2–3 seasons:
- predicted − actual exceeds about +1.5 points;
- |z| > 2.

Do this even if the log-loss gain is tiny (~0.001). The gain that matters is in which bets qualify.

## Task C: Flat favourites and edge by price band

**Hypothesis.** "Heavy favourites are too flat" may be measured against the close. A model with noisier inputs should be flatter than the market, and that is correct calibration. It's a defect only if the model is flat against outcomes.

**Do.**
1. Build a reliability table from walk-forward predictions, games 10+.
   - Take the favourite's side: fold each game to max(P, 1 − P).
   - Use bins of 0.05 from 0.50.
   - Report n, mean predicted, actual and a Wilson 95% CI per bin.

   Build the same table for the no-vig close.
2. Fit the calibration slope logit(y) ~ a + b·logit(p_model), per season and pooled. If b > 1 with a CI excluding 1, the model is flat against outcomes.
3. Only if it is flat against outcomes: refit the base logit with a probit link (same features), walk-forward. Compare log loss overall and in the top bins. The ATS display already assumes normal margins (σ = 13.5), so probit would make the two consistent.
4. Using the H2–H4 edge definition, break qualifying edges down by:
   - market price band: ≤ −300, −300 to −150, −150 to +150, +150 to +300, ≥ +300;
   - side: favourite or underdog.

   For each cell, report the count, ROI at the open price, and CLV (open-to-close movement toward the model's side).

**Decision rule.** Probit becomes a v6 candidate only if both of these hold:
- b > 1 in both walk-forward seasons;
- the probit refit improves the top bins without hurting overall log loss.

**David decides** whether H2–H4 should report by price band if qualifying edges concentrate on big underdogs. That also needs an amendment before native rows.

## Task D: What the persistent team-level gap is made of

**Hypothesis.** `weights.json` is fit descriptively, from season factors to season win%. So all eight factors shrink together through Δ·phase, even though rebounding and turnover rates stabilise much faster than opponents' shooting. `luck_def` corrects only opponents' 3P%. If the FT factor is FTA/FGA, the team's own FT% skill isn't in the model at all.

**Do.**
1. For walk-forward games 10+ with a close, compute r = logit(no-vig P_close) − logit(P_model), from the home side.
2. Regress r on the eight standardised factor gaps (home − away, with the same recency weighting and as-of date as Δ).
   - Then add the gap in each team's own FT% (FTM/FTA).
   - Separately, try a variant with own-side FTM/FGA.
   - Use cluster-robust SEs by team, or a block bootstrap by week.
   - Run it per season.
3. Test team persistence within each season.
   - Take each team's mean r, signed so that positive means the market rates the team above the model.
   - Compare first half with second half and report the correlation.
   - Repeat on r minus the step-2 fit.

   The headline number is how much of the persistent team-level variance the factor gaps explain.
4. Optional: split-half reliability of each factor at 10, 20 and 40 games.
5. If any factor coefficients replicate across both seasons, build a v6 candidate whose game logit uses the eight factor gaps directly. Give each its own coefficient plus a phase interaction, ridge-penalised, and fit on outcomes walk-forward. The residual regression is a diagnostic only. **David decides** whether the model should ever be fit to market prices.

## Task E: Stale talent in the availability logit

**Hypothesis.** `talent_diff` is built from the previous box score. That causes two problems:
- a player who played last game but is Out or Doubtful today still counts;
- a player returning from an absence counts nowhere.

A continuing absence has already left `talent_diff` and also takes the av_* hit. A fresh absence takes only the av_* hit, and fresh absences are the case lines move on. One av_* coefficient averages the two, so fresh absences are probably under-penalised.

**Do.** Use the four report seasons and the same fitting protocol as `logit_avail.json`.
1. Per team-game, compute these two sums, using the same per-player value as `talent_diff`:
   - `newly_out`: players on the previous box score who are Out or Doubtful on the same report v5's availability fit uses.
   - `returning`: players who meet all of these: they appeared for the team earlier this season, aren't on the previous box score, aren't Out or Doubtful on that report, and are still on the roster at the slate date. If roster data isn't available, document the approximation.

   Use pre-tip information only. Never use the game's own box score to decide who returned.
2. Refit the availability logit with `newly_out_diff` and `returning_diff` added. Report coefficients with SEs, and walk-forward log loss against v5's availability logit.
3. If they come out near −/+ the talent coefficient (0.041), build a candidate that computes talent from the projected roster for covered games: previous box score, minus Out/Doubtful, plus returning. Test it walk-forward.
4. Check the Task 0d answer. If "usual share of games" isn't weighted with Δ's 0.5^(games ago / 25) decay, test the decayed version, so the adjustment matches the window Δ actually reflects.

## Out of scope

- Travel and altitude.
- Post-report lineup news.
- The early (games 1–9) model.
- Anything already tested and not adopted: schedule strength, assists, in-season on/off, alternative minutes assignment, and minutes-averaging rules.

## RESULTS.md format

1. Open with the Task 0 answers and Task A's status.
2. Then write one section per task with:
   - what ran;
   - the key table;
   - a verdict: v6 candidate, no change, or needs data or a decision from David.
3. Give log-loss differences as (candidate − v5) per season, with paired SEs.
4. End with a list of anything that contradicted this file.

## Reference: v5 as described in MODEL.md (verify against code)

- **Routing.** g = the fewer games either team has played before the slate date.
  - g = 0: abstain.
  - g = 1–9: early logit.
  - g ≥ 10: the availability logit if the report covers the game, otherwise the base logit.
  - Frozen v4 base logit if a v5 term can't be computed.
- **Base logit.** P = σ(0.297 + 0.0160·Δ + 0.315·b2b + 0.0212·Δ·phase + 0.0125·luck_def + 0.0578·talent_diff). Fit on 2016–19 + 2021–26, n = 10,192.
- **Availability logit.** The base terms plus 0.136·av_min + 0.0366·av_bpm. Fit on the four report seasons, n = 4,289. Talent coefficient 0.041.
- **Early logit.** σ(0.392 + 0.0348·Δ + 0.306·b2b), with last season chained at weight 0.25.
- **Δ.** Eight four-factor features. Each prior game's 16 totals are weighted 0.5^(games ago / 25). Ridge weights are fit to team-season win% (`model/weights.json`). Units are win-% points.
- **phase.** Days since opening night / 175, capped at 1.
- **Against the close** (games 10+, walk-forward): +0.016 log loss for 2024-25 (ESPN BET), +0.019 for 2025-26 (DK).
- **ATS display.** P is mapped to a margin with fixed σ = 13.5. Display only.
