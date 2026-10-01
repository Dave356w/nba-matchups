# v5 audit: results

Handoff: `HANDOFF_v5_audit.md`. Scripts: `diagnostics/v5_audit/` (one per
task, shared folds in `common.py`). Workflow: "v5 audit"
(`.github/workflows/v5-audit.yml`). Research only: no change to
`model/*.json`, routing, frozen rows or H1–H4.

Conventions: ± is **1 SE** unless marked 95%. Log-loss differences are
(candidate − v5) per season with a paired per-game SE; negative means the
candidate is better.

## Task 0: orientation

**a. Walk-forward predictions.** `research/walk_forward.py --avail --v5`
writes them per game to `research/output/walk_forward.csv` (a workflow
artifact, not committed). `backfill_history.reconstruct_season(...,
walk_forward=True, v5=True)` regenerates any season on demand.

- Open and close are not in that CSV. They are in
  `data/nba_reconstructed.csv` (`open_*`, `close_*`, `close_book`; every
  one of the 2,418 rows has both), joined on date and teams.
- Rows by season and book: 2024-25 ESPN BET 1,209; 2025-26 DK 964;
  2025-26 ESPN BET 245.
- This audit rebuilds the predictions in `common.build_folds`. They are
  checked against `nc.build_games` and against the §6.1 log loss in the
  workflow log.

**b. `weights.json` inside the walk-forward.** It is refit inside each
fold. `walk_forward.walk_forward` and `reconstruct_season(walk_forward=True)`
both call `nc.fit_weights(WEIGHT_YEARS < Y)`. The *committed*
`weights.json` was fit once, on 2015–19 and 2021–26, so the
leave-one-season-out reconstructed rows and the live model have seen the
scored seasons. The walk-forward numbers (MODEL.md §6.1) have not.

**c. Free-throw factor.** FTA/FGA, for both own and opponent
(`nc.features_from_totals`: `a["FTA"] / a["FGA"] * 100`). FT% (FTM/FTA) is
not in the model; FTM only enters through eFG's points. Own FT% skill is
missing (tested in Task D).

**d. "Usual share" in `av_min` / `av_bpm`.**
`player_availability.team_availability`:

- a_i = Σ w·[played] / Σ w over the team's games before this one, with
  w = 0.5^(games ago / 25). This is the same decay as Δ. Games are the
  team's own games this season, and a game counts as played when minutes > 0.
- role_i is the same decayed sum of minutes divided by the decayed games
  played, over 48.

So a_i is already decayed like Δ, and the decayed variant asked for in
Task E.4 is already what ships. By contrast, `talent_diff`'s minutes share
(`arrival_roles`) is an *undecayed* season-to-date mean (else last
season's MP/G). Its averaging rules were tested in PR #39 and are out of
scope.

**e. Snapshot and price timing.**

- **First snapshot.** The build runs hourly at :07 from 10:07 to 22:07 ET
  (`build.yml`) and scores only that day's slate. A game's first snapshot
  is normally the 10:07 ET run on game day.
- **Open.** The open is ESPN's `open` field for the book, and the time it
  was posted is not exposed. It is posted before game day, so the first
  snapshot always comes after the open.
- **Grader's source and book.** `market.pick_close(market.book_odds(gid),
  prefer=pre_book)` reads ESPN core odds `open`/`close` from one book:
  the row's latest `pre_book`, else DraftKings, then ESPN BET. Provider 59
  (live) is ignored.
- **Book mismatch.** `prefer` is the *latest* pregame book, not
  `first_book`. If the book changed between the first and last snapshot,
  open/close come from a different book than `first_*`. The `first_book`
  column records the book; the grader does not compare the two.
- **Current price at snapshot time.** Yes. The snapshot reads `current`
  from the same ESPN feed and book order (`market.pick_pregame`).
- **Injury-report timing.** The live build reads the latest report at or
  before *now*, so the first snapshot uses the morning edition. The
  availability logit was fit on the last report **≥ 30 min before tip**.
  The first-snapshot P therefore comes from an earlier report than the one
  its coefficients were trained on. This is relevant to H4.

**f. Intercepts.**

| Logit | Intercept | σ(intercept) |
|---|---:|---:|
| `logit_avail.json` | 0.247 | 56.1% |
| `logit.json` | 0.297 | 57.4% |
| v4 fallback | 0.291 | 57.2% |
| early | 0.392 | 59.7% |

The availability logit is fit on 2023–26 only. Every games-10+ test game
was covered, so it scores essentially all games 10+, and its home edge is
1.3 points lower than the pooled base. That is consistent with Task B's
hypothesis, but the intercepts are at zero features (mean av_min is not
exactly 0), so Task B measures it directly.

**g. H2–H4 as coded** (`analysis.HYPOTHESES`, `hypothesis_picks`). The
value side is the side where model P > no-vig q, both read at the scoring
price.

| | Rule | Native price | Hindsight price |
|---|---|---|---|
| H2 | games 10+, value side **and** favourite (q > 0.5), edge > 0 | `pre` (latest pregame snapshot) | open |
| H3 | games 10+, value side, P − q ≥ 0.12 | `pre` | open |
| H4 | H3's rule | `first` (`first_*` with `first_p_home`) | open |

No price-band conditions. (H1: games 1–9, value side, edge ≥ 0.08, close.)

**h. Opening night 2026-27.** This is **not in the repository's data**:
there are no schedule files, and `nc.season_opening` reads it from the
game logs once games exist. The tests' 2026-10-21 is a fixture, not
schedule data. Task A's deadline is the first regular-season slate on
ESPN's scoreboard (`build_site.score_slate` skips preseason). The
first native row is written by the 10:07 ET build on opening day. Listed
under contradictions below.

## Task A: decision-time price (status)

**Implemented in PR #41** (`claude/v5-audit-decision-price`), awaiting
David's review. The pipeline already had a live price source (ESPN core
odds `current`, same book order as the grader), and `first_*` already
stores the book, UTC time, both American prices and no-vig q (devigged
with `market.devig`, as the grader does).

| New column (write-once) | Content |
|---|---|
| `first_report_utc` | injury-report edition the first snapshot's build read (ET slot → UTC) |
| `first_route` | early / avail / base / v4 |
| `seen_utc` | the row's earliest snapshot, written at creation |
| `seen_note` | why `first_*` was not filled then: `no price (<reason>)` / `no model P`; blank if it was |

Done-when checks are covered by tests:

- a dry run of the build writes the fields once;
- a second run leaves them unchanged;
- grading leaves them unchanged and still writes only `GRADE_COLUMNS`.

**David decides:** whether to register an amendment so H2–H4 are also
scored at the first-snapshot price. This must happen before the first
native row.

## Run and reproduction check

The "v5 audit" workflow ran on runs 36939293871 and 36939891686 (the
second adds D.5b), both on branch `claude/v5-audit-research`.

- **Fast builder vs `nc.build_games`.** On 2025-26 with the production
  weights, Δ, luck_def, talent_diff, d_phase and b2b_net match to
  ≤ 2e-13 on all 1,071 games.
- **Folds vs MODEL.md §6.1.** The folds reproduce §6.1 to four decimals:
  v5 log loss 0.5939 (2024-25 ESPN BET, n = 1,071), 0.5896 (2025-26 DK,
  n = 964), 0.5481 (2025-26 ESPN BET, n = 107). The closes are 0.5776,
  0.5702 and 0.5551.
- **Coverage.** All games 10+ of 2023-24 to 2025-26 are covered and go
  through the availability logit. 2016-17 and 2017-18 send about 140
  games each to the v4 fallback, because a v5 term is missing there.

## Task B: home-court drift

**B.1: v5 walk-forward, games 10+, predicted − actual home win %, ± 1 SE.**

| Season | Route | n | Actual | Predicted | Pred − actual | z |
|---|---|---:|---:|---:|---:|---:|
| 2016-17 | all | 1,074 | 58.4 | 59.2 | +0.8 ± 1.4 | 0.6 |
| 2017-18 | all | 1,075 | 58.3 | 59.2 | +0.9 ± 1.4 | 0.6 |
| 2018-19 | base | 1,074 | 59.0 | 58.4 | −0.6 ± 1.4 | −0.4 |
| 2020-21 | base | 924 | 55.4 | 58.8 | +3.3 ± 1.5 | 2.2 |
| 2021-22 | base | 1,073 | 54.2 | 58.2 | +3.9 ± 1.4 | 2.8 |
| 2022-23 | base | 1,073 | 57.6 | 57.1 | −0.5 ± 1.4 | −0.3 |
| 2023-24 | avail | 1,074 | 53.7 | 56.9 | +3.2 ± 1.4 | 2.3 |
| 2024-25 | avail | 1,071 | 54.3 | 55.6 | +1.3 ± 1.4 | 0.9 |
| 2025-26 | avail | 1,071 | 54.9 | 55.1 | +0.2 ± 1.4 | 0.1 |
| **pooled last 2** | avail | 2,142 | 54.6 | 55.3 | **+0.75 ± 0.97** | 0.8 |
| **pooled last 3** | avail | 3,216 | 54.3 | 55.9 | **+1.56 ± 0.79** | 1.97 |

The model leaned home in 4 of the 6 seasons from 2020-21 on. The
availability logit, fit on report seasons only, has already absorbed most
of the drop: the gap fell from +3.2 (2023-24, trained on one season) to
+1.3 and +0.2.

**B.2: candidate − v5 log loss, games 10+, ± 1 SE.** These are
walk-forward and use the production routing.

| Candidate | 2024-25 | 2025-26 | Pred − actual 2024-25 / 2025-26 |
|---|---:|---:|---:|
| int1 (intercept on last season) | −0.0004 ± 0.0011 | +0.0001 ± 0.0006 | −0.2 / −0.6 |
| int2 | 0 (= v5: 2 training seasons) | +0.0001 ± 0.0006 | +1.3 / −0.8 |
| int3 | 0 | 0 (= v5) | +1.3 / +0.2 |
| wt1 (h = 1) | +0.0005 ± 0.0007 | +0.0003 ± 0.0009 | +0.8 / −0.3 |
| wt2 | +0.0002 ± 0.0004 | +0.0001 ± 0.0005 | +1.0 / −0.1 |
| wt3 | +0.0001 ± 0.0002 | +0.0001 ± 0.0003 | +1.1 / −0.0 |

- The availability logit has at most 3 training seasons, so int2 and int3
  equal v5 wherever k covers all of them.
- On the earlier, base-routed seasons the variants help in 2021-22 (int1
  −0.0033 ± 0.0018) and hurt in 2022-23 (int1 +0.0031 ± 0.0021). The
  recency signal does not persist from one season to the next.
- Against the close every variant is within 0.0005 of v5 (+0.016 /
  +0.019).

**B.3: which bets qualify (value side at the open; ROI % beside its null
of about −4.1).**

| Rows | Candidate | H2 n (home/away) | H2 ROI | H3/H4-open n (home/away) | H3 ROI |
|---|---|---:|---:|---:|---:|
| 2024-25 ESPN BET | v5 | 508 (314/194) | +0.2 | 132 (68/64) | +19.0 |
| | int1 | 496 (266/230) | +0.8 | 147 (49/98) | +10.4 |
| | wt1 | 522 (306/216) | −0.5 | 146 (65/81) | +10.9 |
| 2025-26 DK | v5 | 405 (248/157) | −0.9 | 128 (67/61) | +15.1 |
| | int1 | 399 (227/172) | −0.8 | 121 (49/72) | +18.1 |
| | wt1 | 417 (240/177) | −1.4 | 133 (55/78) | +19.2 |

A one-season intercept moves H3's picks sharply toward away teams (68 →
49 home in 2024-25), but H3's ROI does not depend on it in a consistent
direction (−8.6 and +3.0 points in the two seasons). Full table: workflow
summary.

**Verdict: no change.** The decision rule (pooled > +1.5 pts **and** |z| > 2)
is **not met**. Last 3 seasons: +1.56 ± 0.79, z = 1.97, at the threshold.
Last 2: +0.75 ± 0.97. The inflation the handoff feared is real in the
seasons when the availability logit was young (2023-24: +3.2) and has
mostly fit itself out of the current one. Recheck with the native rows,
reporting home and away separately, as CLAUDE.md already asks for H1.

## Task C: flat favourites and edge by price band

**C.1: reliability, favourite's side, pooled 2024-25 + 2025-26 (n = 2,142),
actual with Wilson 95% CI.**

| Bin | v5 n | v5 mean P | v5 actual | Close n | Close mean q | Close actual |
|---|---:|---:|---:|---:|---:|---:|
| 0.70–0.75 | 268 | 0.724 | 0.757 (0.70–0.81) | 273 | 0.724 | 0.780 (0.73–0.83) |
| 0.75–0.80 | 223 | 0.773 | 0.807 (0.75–0.85) | 221 | 0.773 | 0.742 (0.68–0.80) |
| 0.80–0.85 | 238 | 0.823 | 0.819 (0.77–0.86) | 225 | 0.824 | 0.787 (0.73–0.84) |
| 0.85–0.90 | 146 | 0.874 | 0.842 (0.78–0.89) | 186 | 0.873 | 0.909 (0.86–0.94) |
| 0.90–0.95 | 53 | 0.920 | 0.925 (0.82–0.97) | 75 | 0.917 | 0.987 (0.93–1.00) |

The full 0.50–1.00 table per season is in the workflow summary.

**C.2: calibration slope b (95% CI).**

| Series | 2024-25 | 2025-26 | Pooled |
|---|---|---|---|
| v5 | 0.99 (0.84–1.14) | 1.05 (0.89–1.20) | 1.02 (0.91–1.13) |
| close | 1.06 (0.90–1.21) | 1.08 (0.93–1.23) | 1.07 (0.96–1.18) |

**C.3: probit − v5 log loss.** Overall: +0.0004 ± 0.0003 (2024-25),
+0.0003 ± 0.0003 (2025-26). For favourites ≥ 0.75: +0.0011 ± 0.0009 and
+0.0007 ± 0.0007. Probit is slightly worse everywhere.

**C.4: H2 and H3/H4-open qualifying edges by the picked side's open price.**
ROI % ± 1 SE; ROI null about −4. CLV is the movement in no-vig pp from the
open to the close toward the pick.

| Rows | Rule | Band | Side | n | ROI | CLV |
|---|---|---|---|---:|---:|---:|
| 2024-25 ESPN BET | H2 | ≤ −300 | fav | 153 | −6.1 ± 4.0 | +0.8 |
| | H2 | −300 to −150 | fav | 225 | −2.1 ± 4.7 | +2.4 |
| | H2 | −150 to +150 | fav | 130 | +11.7 ± 7.5 | +2.3 |
| | H3 | −150 to +150 | fav / dog | 21 / 28 | +50.8 ± 13.8 / +8.6 ± 21.0 | +7.0 / +4.1 |
| | H3 | +150 to +300 | dog | 33 | +12.3 ± 26.7 | +3.7 |
| | H3 | ≥ +300 | dog | 29 | +26.0 ± 44.9 | +5.7 |
| 2025-26 DK | H2 | ≤ −300 | fav | 117 | −5.6 ± 4.5 | +1.0 |
| | H2 | −300 to −150 | fav | 185 | −3.2 ± 5.2 | +2.0 |
| | H2 | −150 to +150 | fav | 103 | +8.8 ± 8.6 | +1.5 |
| | H3 | −300 to −150 | fav | 21 | +31.4 ± 12.2 | +4.4 |
| | H3 | −150 to +150 | dog / fav | 29 / 14 | +29.5 ± 19.3 / +13.8 ± 23.6 | +3.3 / +3.3 |
| | H3 | +150 to +300 | dog | 39 | −10.0 ± 22.1 | +4.7 |
| | H3 | ≥ +300 | dog | 24 | +24.4 ± 45.2 | +2.9 |

**Verdict: no change; the premise does not hold.** Against outcomes, v5 is
not flat: b = 1.02 (0.91–1.13) pooled, and the CI includes 1 in both
seasons. "Heavy favourites too flat" (MODEL.md §7) was measured against
the close or on an earlier model; v5 walk-forward has no such defect.
Probit fails its gate.

On C.4:

- **H3's qualifying edges spread across all bands.** About half are
  underdogs at +150 or longer. The big-dog cells have SEs of 22–45 points
  and resolve nothing.
- **H2's heavy favourites** (≤ −300) run −6 / −6 against a null of −4.7 /
  −4.0, with small positive CLV. That is about the null.
- **H2's near pick'em favourites** (−150 to +150) are the positive cell in
  both seasons: +11.7 ± 7.5 and +8.8 ± 8.6, about 2.1 and 1.5 SE above
  the null. This is an in-sample band, so it is a hypothesis for native
  rows, not a filter (CLAUDE.md rule 4).
- **CLV is positive in every cell.** Lines move toward the model from the
  open, as in the earlier open-vs-close study.

**David decides:** whether H2–H4 should *report* by price band. The data
do not show H3's edges concentrating on big underdogs.

## Task D: what the persistent team-level gap is made of

r = logit(no-vig close) − logit(v5 walk-forward P), home side, games 10+.
2025-26 ESPN BET (n = 107) is skipped (< 300).

**D.2: r on the eight factor gaps.** Coefficients are per sample SD of
the gap, with week block-bootstrap SEs (1,000 draws). Only terms with
|t| > 2 in at least one season are shown.

| Term | 2024-25 ESPN BET | 2025-26 DK |
|---|---:|---:|
| off eFG | −0.034 ± 0.027 | **−0.085 ± 0.010** |
| −opp eFG | +0.030 ± 0.022 | **+0.120 ± 0.019** |
| opp TOV forced | **+0.058 ± 0.022** | −0.006 ± 0.014 |
| −opp ORB | **+0.031 ± 0.013** | **+0.045 ± 0.012** |
| −opp FTA/FGA | **−0.029 ± 0.013** | +0.001 ± 0.016 |
| off FTA/FGA | −0.001 ± 0.012 | **+0.040 ± 0.017** |
| *+ own FT% (FTM/FTA)* | **+0.042 ± 0.014** (t 3.0) | **+0.095 ± 0.010** (t 9.7) |
| R²: 8 factors / + FT% | 0.089 / 0.098 | 0.135 / 0.180 |

- Positive means the market rates the team above the model as that gap
  grows.
- The own FTM/FGA variant (+0.12 / +0.34) mostly re-expresses FT%: off
  FTA/FGA flips to −0.11 / −0.29 beside it.
- Replicated in both seasons (same sign, |t| > 2): **own FT%** and
  **−opp ORB**.
- Offensive eFG and opponents' eFG pull in opposite directions in 2025-26
  (t −8 and +6). The market discounts shooting-driven offence and credits
  shooting-driven defence beyond the model, after luck_def. This does not
  replicate in 2024-25.

**D.3: team persistence.** Each team's mean signed r, first half of the
season vs second half, across 30 teams.

| Rows | Split-half corr, raw | After the fit | Split-half cov, raw → after | Share explained |
|---|---:|---:|---:|---:|
| 2024-25 ESPN BET | 0.33 | 0.16 | 0.0072 → 0.0027 | **63%** |
| 2025-26 DK | 0.30 | 0.25 | 0.0103 → 0.0055 | **46%** |

The eight factor gaps plus own FT% explain about half of the persistent
team-level disagreement with the market (46–63%). The rest is not in the
box-score factors.

**D.5: candidates fit on outcomes, walk-forward, production routing.**
± 1 SE.

| Candidate | 2024-25 ESPN BET | 2025-26 DK | 2025-26 ESPN BET (107) |
|---|---:|---:|---:|
| 8 factor gaps + gap × phase in place of Δ, Δ·phase (ridge λ = 30 by training LOSO) | +0.0004 ± 0.0019 | +0.0001 ± 0.0019 | −0.0009 ± 0.0099 |
| **v5 + own FT% gap** (D.5b) | **−0.0003 ± 0.0004** | **−0.0010 ± 0.0006** | −0.0018 ± 0.0022 |
| v5 + own FTM/FGA gap | +0.0003 ± 0.0007 | −0.0007 ± 0.0008 | −0.0027 ± 0.0031 |

On all games 10+, v5 + FT% is −0.0003 ± 0.0004 (2024-25) and
−0.0011 ± 0.0005 (2025-26). Its gap to the close moves from +0.0163 to
+0.0160 and from +0.0194 to +0.0184. The availability-logit coefficient is
+0.89 / +1.04 per unit FT% (about 0.01 logit per FT% point). It is stable
across folds and has the sign the market residual predicts.

**Verdict.**

- **v5 + own FT% is a (small) v6 candidate.** It is better in both
  seasons and in the same direction as the replicated residual term.
  Resolved in 2025-26 (~2 SE), not in 2024-25. Its size, about −0.001, is
  far from the 0.008 tripwire. Next step: the v5 gate protocol
  (`research/walk_forward.py --avail --v5` with the term), then a
  decision. FT% is not a market-fit term: it is fit on outcomes, and the
  residual regression only pointed at it. **David decides.**
- **Replacing Δ with the eight free factor coefficients: no change.** The
  ridge refit adds nothing (±0.0019). The composite already uses the
  factors about as well as outcomes can identify them. The persistent
  market view is FT% plus something outside the box score.
- D.4 (split-half factor reliability) was not run; it is optional.


## Task E: stale talent in the availability logit

`newly_out` counts players on the previous box score who are Out or
Doubtful on the report the fit uses (≥ 30 min before tip). `returning`
counts players who played for the team earlier this season, are missing
from the previous box, are not Out or Doubtful, and whose latest
appearance was for this team.

**Roster approximation.** No roster data exists, so a waived or demoted
player with no later appearance still counts. `returning_listed` adds the
requirement that the player is listed on the report with a playable
status. `returning` is non-zero in almost every game (DNP-CD bench players
who played earlier count), with mean |diff| 1.6–2.0 against talent_diff's
5–6.

**E.2: coefficients (± 1 SE).** talent_diff in the same fit is
0.056–0.076.

| Fit | newly_out | returning | av_bpm |
|---|---:|---:|---:|
| In-sample 2023-26 (n = 4,289) | −0.062 ± 0.022 | +0.019 ± 0.013 | +0.014 ± 0.017 |
| In-sample, returning_listed | −0.070 ± 0.022 | +0.035 ± 0.022 | +0.006 ± 0.017 |
| Walk-forward for 2024-25 (n = 2,147) | −0.054 ± 0.031 | +0.057 ± 0.019 | +0.011 ± 0.025 |
| Walk-forward for 2025-26 (n = 3,218) | −0.049 ± 0.025 | +0.036 ± 0.015 | +0.033 ± 0.019 |

newly_out comes out at about −1× the talent coefficient: the stale-roster
double count is real. Adding it pulls av_bpm from 0.037 toward 0.01, so
av_bpm was mostly absorbing fresh absences. returning is positive but
smaller than talent and unstable (+0.019 to +0.057).

**E.2 / E.3: walk-forward log loss vs v5's availability logit, same
covered games, ± 1 SE.**

| Candidate | 2024-25 ESPN BET | 2025-26 DK | 2025-26 ESPN BET (107) |
|---|---:|---:|---:|
| + newly_out + returning | +0.0022 ± 0.0025 | +0.0005 ± 0.0020 | +0.0013 ± 0.0056 |
| + newly_out + returning_listed | −0.0006 ± 0.0017 | −0.0010 ± 0.0015 | −0.0032 ± 0.0044 |
| projected-roster talent (E.3) | +0.0027 ± 0.0034 | +0.0018 ± 0.0028 | +0.0029 ± 0.0070 |

**Verdict: needs more data; no v6 candidate yet.**

- The mechanism is confirmed: newly_out ≈ −talent, t ≈ −2 to −3 in every
  fit.
- The log-loss gain is unresolved. The only arm that is better in both
  seasons, returning_listed, gains about 0.001 ± 0.0015.
- The projected-roster talent (E.3) is worse, because the loose
  `returning` set adds noise.
- The newly_out-only variant was not run separately. It is the natural
  next arm, along with a real roster source (the `data/nba_injuries.csv`
  snapshots), which would replace the approximation.
- E.4: not needed. a_i is already decayed (Task 0d).

## Leakage tripwire

No candidate beats the close, and none improves on v5 by more than 0.008
in a season. The largest gain is −0.0011 (v5 + own FT%, 2025-26).


## Contradictions with the handoff

1. **Task A item 4 vs H4.** The handoff says that when there is no price at
   snapshot time, null + a reason should be written and never filled from a
   later fetch. H4's frozen definition is "the earliest pregame snapshot
   *with a model P and a price*", so `first_*` must wait for a price.
   Resolved without editing H4: `first_*` keeps its definition (P, price
   and report from one snapshot, never mixed), and the new `seen_utc` /
   `seen_note` record the earlier no-price snapshot with its reason.
2. **Opening night (0h)** is not in the repo's data; see above.
3. **D.5** builds its candidate from the eight factors only, but the
   strongest replicated residual term (own FT%) is not one of them. D.5b
   adds it as an extra arm, fit on outcomes.
4. **E.4** asks for a decayed "usual share" if it isn't decayed. It already
   is (0d), so nothing was tested.
