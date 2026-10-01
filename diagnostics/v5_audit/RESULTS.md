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

<!-- TASKS B-E: filled from the "v5 audit" workflow run -->

## Contradictions with the handoff

1. **Task A item 4 vs H4.** The handoff says that when there is no price at
   snapshot time, null + a reason should be written and never filled from a
   later fetch. H4's frozen definition is "the earliest pregame snapshot
   *with a model P and a price*", so `first_*` must wait for a price.
   Resolved without editing H4: `first_*` keeps its definition (P, price
   and report from one snapshot, never mixed), and the new `seen_utc` /
   `seen_note` record the earlier no-price snapshot with its reason.
2. **Opening night (0h)** is not in the repo's data; see above.
3. **E.4** asks for a decayed "usual share" if it isn't decayed. It already
   is (0d), so nothing was tested.
