# Claude — working agreement for NBA Matchups

The owner directs the product. Be rigorous, inspect the code and current data,
state uncertainty, and turn findings into concrete tests.

## Live model and forward tests

Public UI approval, 2026-10-23: publish only Today, Results, Performance and
How it works. Public records contain the three registered native tests,
with test × stage × season separate. Forecasts and frozen tracked picks
are explicitly distinct. No legacy lean headline, adjusted edge/EV,
reconstructed results, retired rules or exploratory price bands appear in
the public UI. Keep the complete workbench under `diagnostics/site/`,
outside the Pages artifact, and maintain `data/ledger_report.txt` as before.
The original registration's Ledger/Retired references below now refer to
that internal workbench. `public/` is a generated, allowlisted directory;
remove stale contents and verify it before upload. The repository itself
remains public. This changes presentation only, not prediction maths,
thresholds, native snapshot locks or registered rule membership.

Production uses v9 everywhere: the base margin model in `model/logit.json`
and its injury-report arm in `model/logit_avail.json`, with the same faded
rating from game 0. These are one model's covered/uncovered routes. Model
P = Φ(predicted home margin / fitted residual sigma). Missing optional
inputs contribute zero and carry distinct `margin_neutral_v9` /
`margin_neutral_avail_v9` tags; missing core ratings abstain. Last season's
rating fades out by game 10. No carryover/v4 fallback is loaded. Frozen
legacy fits and scoring are research-only under `research/legacy_*`.

There are three active first-snapshot tests, each with two separate rows:

| Test | Games 10+ | Games 1–9 | Frozen rule |
|---|---|---|---|
| H5 spread | H5 | H5·E | 13.5 × Φ⁻¹(P); edge ≥ 1 point; both spread prices −135..+115 |
| H2·F moneyline | H2·F | H2·F·E | Value side when no-vig q > 0.5 |
| H4 moneyline | H4 | H4·E | Value side when P − no-vig q ≥ 0.12 |

The games-10+ registrations are unchanged. Games 1–9 and 10+ are never
pooled, at least until a season of each exists; any pooling needs an explicit
amendment. Game 0 is in no rule. First prices, probabilities and tags are
immutable, including any recorded before the v9 rollout. Early rows use
only snapshots tagged v9: native rules read `first_model_tag`, never a
newer pregame tag in place of the frozen model. Older snapshots remain
history and do not enter the new early segments. Early v9 uses
injury reports, so every early segment has the injury-timing caveat. The
first snapshot is later than the open, not an exact opening-price replica.

H1, H6 and late-pregame H2/H3 are retired. Their registered hindsight
figures and ungraded native bets are frozen under internal Ledger → Retired and in
`diagnostics/first_snapshot_registration/retired.json`; no new grades enter
an active record. The ESPN BET early-segment hindsight (276 v9 rows) is
information only, separate from DraftKings and native evidence.

## Registration (unchanged timestamp and terms)

- **First-snapshot amendment (owner's plan, registered 2026-10-23T19:46:58Z).**
  Registered before CHI @ MIL (2026-10-23) is final: the fetched native
  ledger at `b4fcb67` has no result for game 401909848 (tip 20:00Z).
  This registration is committed on its own before implementation or
  computing the new segments' hindsight figures.
  - Keep exactly three forward tests, each at the frozen **first snapshot**:
    **H5** (spread), **H2·F** (moneyline value favourite), **H4** (moneyline
    value side with P − no-vig q ≥ 0.12). H5 keeps sigma 13.5 × Φ⁻¹(P),
    edge ≥ 1 point and both main-line prices −135..+115. No thresholds,
    price filters or games-10+ membership change.
  - Extend each to games **1+** by adding its own **games 1–9 segment**,
    using the primary model P (v9), never the carryover model P. Games
    10+ keep their existing registered rows. Show games 1–9 and games
    10+ separately, never pooled, until at least one season of each exists
    (any later pooling requires an explicit amendment). Game 0 remains
    outside all hypotheses. Existing first snapshots stay immutable,
    including their recorded model tags and probabilities.
  - Retire **H1**, **H6**, and **H2/H3 at the late pregame price** as of
    this timestamp. Keep all four on the Ledger under **Retired**, with
    their registered hindsight figures and their native bets to date;
    none has a graded native bet at registration. Do not admit later
    snapshots or grade these bets into any active forward record.
    H1/H6 retain their historical carryover P for archival reproduction.
  - The early segments replace H1/H6. Compute their hindsight on the
    **276 ESPN BET early games** using v9 rows, for information only;
    label this hindsight explicitly, apart from DraftKings and the native
    forward test. No threshold selection or new forward evidence.
  - Early v9 uses injury-report inputs: these new segments inherit the
    injury-timing caveat of H2–H5. The first snapshot is later than the
    open and is not an exact replica of the opening-price backtest.
  - Ledger scoreboard, report, Results/Performance spread tiles and Picks
    badges use these same active rules and separate stage rows. Test
    membership by stage, retirement retention/exclusion, and frozen
    thresholds. Run data validation and the full suite; capture page
    screenshots. This amendment supersedes conflicting active-rule
    descriptions below; the original registrations remain as history.

- **Model v10 amendment (owner's decision, registered 2026-10-28T18:16:46Z).**
  Committed on its own before any v10 code or snapshot.
  - v10 = v9 plus three changes, from
    [v10 candidate](diagnostics/v10_candidate/RESULTS.txt) (arm v10h25):
    the composite's rebounding factors as ORB% (ORB / (ORB + opp DRB));
    drafted rookies valued by pick (BPM = a + b log pick, fitted on
    earlier seasons) in talent_diff and av_bpm; and the injury-report
    route refitted before each slate on the report seasons plus this
    season's covered games before the slate. Half-life stays 25. New tags
    end `_v10`; neutral-input variants keep their own tags.
  - Native first snapshots, probabilities and tags recorded before
    activation are immutable and never rescored. Rows keep refreshing only
    before tip; only rows first snapshotted after activation carry v10.
  - The three tests continue in their existing rows (test × stage ×
    season), with unchanged rules, thresholds and prices. Each bet reads its
    own frozen first P and tag. The games 1–9 rows accept first tags ending
    `_v9` or `_v10`; older tags still never enter them. A by-tag split is
    kept internally.
  - Reconstructed and rebuilt (hindsight) rows are rescored under v10 by
    the usual post-change checks. Pages describe the current model with
    one dated line that earlier picks used the previous version; no
    hindsight figures are added to public pages. Activation waits on the
    G7–G10 checks.

- **Five-hour lock amendment (owner's decision, registered 2026-10-28T21:30:29Z).**
  Committed on its own before any lock code or snapshot. Evidence:
  [Questionable weighting and lock lead](diagnostics/questionable_weight/RESULTS.txt).
  At the overnight first snapshot (~18 h before tip) the NBA report has
  flagged 23–26% of eventual absences on back-to-backs and 71–72% on
  other games; by 5 h, 76–84% of both.
  - Each tracked bet is frozen at the **lock snapshot**: the first pregame
    snapshot taken within 5 hours of tip (snapshot time ≥ tip − 5 h).
    Its prices, model P, model tag, report time and route are write-once
    `lock_*` columns. A game with no snapshot in that window has no
    tracked bet; missed locks are counted internally. First snapshots keep
    being recorded, write-once, as history and forecasts.
  - The three tests keep their rules, thresholds and prices (H5 13.5 ×
    Φ⁻¹(P), edge ≥ 1, both spread prices −135..+115; H2·F q > 0.5; H4
    P − q ≥ 0.12), read at the lock price with the lock P and tag. Stage
    membership is unchanged (games 10+; games 1–9 with v9-or-later tags;
    game 0 never).
  - **New rows from activation.** Each test × stage × season gets
    separate 5 h-lock rows (H5·L, H5·E·L, H2·F·L, H2·F·E·L, H4·L, H4·E·L).
    The first-snapshot rows close with the last slate before activation
    (the first slate with a lock snapshot); from that slate on, games enter
    only the lock rows. Closed rows stay visible and graded, never pooled
    with the lock rows. CLV is the lock price against the close.
  - Builds are attempted every 30 minutes through each slate's afternoon
    and evening, since scheduled runs are dropped or delayed. Pages show
    forecasts until a game locks and its tracked pick after. This changes
    the bet's timing only, not prediction maths.

- **Model v11 amendment (owner's decision, registered 2026-10-28T21:30:46Z).**
  Committed on its own before any v11 code or snapshot.
  - v11 = v10 plus Questionable weighting on the injury-report route (arm
    q of [Questionable weighting](diagnostics/questionable_weight/RESULTS.txt)):
    Doubtful, Questionable and Probable players count as the share of
    listed players with that status who logged minutes in the report
    seasons, read at the report's lead before tip (the nearest of 18, 12,
    8, 5 and 2 h and 30 min); Out counts 0; Available and unlisted
    previous-box players count 1. The route's fit is unchanged (Out/Doubtful
    on the last report ≥ 30 min before tip, refitted before each slate).
    New tags end `_v11`; neutral-input variants keep their own tags.
  - Adopted by the owner although arm q failed the G7 rule set before its
    run: games 1+ −0.0012 ± 0.0010 at 18 h and −0.0006 ± 0.0012 at 5 h,
    but DraftKings 2025-26 games 10+ +0.0007 at 5 h (± 95%).
  - Native snapshots, probabilities and tags recorded before activation
    are immutable and never rescored. The games 1–9 rows also accept tags
    ending `_v11`; a by-tag split is kept internally.
  - Reconstructed and rebuilt rows are rescored under v11 by the usual
    post-change checks (rescore, rebuild, build, model review, DraftKings
    and History backtests), reviewed before activation. Pages carry the
    one dated model-version line; no hindsight figures are added.

- **Model-family reset amendment (owner's decision, registered 2026-10-29T13:58:40Z).**
  Committed on its own before any reset code and before any v11 lock: the
  native ledger at `f1d3ed8` has no lock snapshot on the 2026-10-29 slate
  (first tip 2026-10-30T00:00Z) and no first or lock tag ending `_v11`.
  - v11, live from the 2026-10-29 refit, starts a new model family, and
    the public records restart with it. From the **2026-10-29 slate**, each
    test × stage × season gets new v11-family rows (H5·L·v11, H5·E·L·v11,
    H2·F·L·v11, H2·F·E·L·v11, H4·L·v11, H4·E·L·v11), read at the 5 h lock
    with the lock price, P and tag. Rules, thresholds, prices, lock timing
    and stages are unchanged (H5 13.5 × Φ⁻¹(P), edge ≥ 1, both spread
    prices −135..+115; H2·F q > 0.5; H4 P − q ≥ 0.12; games 1–9 and 10+
    separate; game 0 never), except that both stages require a lock tag
    ending `_v11` (neutral-input variants included).
  - The 5 h-lock rows (·L) close with the 2026-10-28 slate; the
    first-snapshot rows stay closed. Closed rows keep their bets, grades
    and CLV in the internal workbench and `data/ledger_report.txt`, never
    pooled with the v11-family rows. At registration they stand: first
    snapshot H5·E 8–14 (−6.72u), H2·F·E 5–4 (−1.64u), H4·E 3–2 (+1.55u);
    5 h lock, v10, 2026-10-28 slate H5·E·L 1–0 (+0.91u), H2·F·E·L 0–2
    (−2.00u), H4·E·L 0–1 (−1.00u); no games-10+ bets. A game from the reset
    slate on whose lock tag does not end `_v11` has no tracked bet and is
    counted internally.
  - Public Today, Results and Performance show only the v11-family rows,
    with one dated line that the record restarted on October 29, 2026 with
    the current model and that earlier picks are kept separately. No
    earlier record appears on the public pages.
  - A later model version enters these rows only through its own amendment
    saying whether it continues this family. Prediction maths, thresholds,
    native first and lock snapshots and their tags are unchanged; this
    changes which rows the records read and what the pages show.

## Sources of truth and engineering rules

- Read `docs/model_guidelines.md` before a model or rule change. The model
  is `MODEL.md` / `nba_composite.py`; prices are `market.py`; pregame locks
  and schema are `ledger.py`; grading/statistics are `analysis.py`.
- Quote current numbers from `data/ledger_report.txt`, the pages or current
  diagnostics, never an old PR. Native (`data/nba_ledger.csv`), reconstructed
  (`data/nba_reconstructed.csv`) and rebuilt current-season rows
  (`data/nba_rebuilt.csv`) are separate evidence. Never pool them in a fit,
  rule, record or display. Rebuilt is hindsight, never a pregame snapshot.
- No lookahead: features use games strictly before the slate date. Refresh
  only before tip; freeze after. Grading writes result/close columns only.
  First snapshots and their model P/tags are write-once. Native history is
  never rescored. A prediction-maths change needs a new model tag.
- Do not hand-commit bot-generated `data/` or `public/` changes. Existing
  Kalshi columns, carryover columns and `data/nba_preseason.csv` are history:
  keep and validate them, preserve values, stop new writes. No exhibitions
  are scored or published. Preseason box scores remain model inputs for
  regular-season game 0; do not confuse those with exhibition predictions.
- DraftKings is the live/display market. ESPN BET stays in the historical
  data, apart from the explicitly labeled early hindsight diagnostic. Keep
  book × season × basis × stage separate. Read ROI against its no-vig null
  and same-game favourite. Price bands are descriptive, never filters.
- Pages and report use ±1 SE, labeled. Research uses ±95% unless explicitly
  labeled otherwise. Colour signals a gap ≥2 SE from its null, not sign.
  Pages use Pacific time; slate dates follow ET and data timestamps use UTC.
- Public cards show one primary probability and margin. Shrunk edge/EV (w=0.40)
  and descriptive price windows remain internal research; graded rules use the raw
  frozen first P. The display margin never replaces H5's frozen 13.5 conversion.
- Keep site-build serialized with cancel-in-progress false and its data
  validate/commit on !cancelled(), even when rendering fails. Tests gate PRs,
  never block a pregame snapshot. Other data-writing workflows use their own
  queue and retry pushes.
- `data/nba_box_<season>.csv` and preseason boxes are model inputs. Injury
  lists (`data/nba_injuries.csv`) shown on cards and Sleeper projections
  (`data/nba_sleeper.csv`, first/latest snapshots) stay research-only. The ESPN
  injury lists are retained internally rather than presented as the NBA
  injury-report inputs. Their
  failures never stop a build or alter its probability.

Before every PR: `python validate_data_files.py`, `python -m pytest tests/ -q`,
and screenshots for page changes. After prediction changes: historical
rescore, current-season rebuild, build, model review, DraftKings backtest and
History backtest (G7/G10). No activation until these checks are reviewed.

## Open questions

- Native H5/H2·F/H4 first-snapshot ROI and CLV, with early/established rows
  separate and every result reported. One season is not a general verdict.
- A truly pregame roster and newly-out-player adjustment, once injury/Sleeper
  snapshots cover enough games; walk-forward gate before any model use.
- Early-season calibration and home lean; watch without fitting to thin data.
- Re-measure the shrinkage weight at the first snapshot when native coverage
  supports it. Rookies by draft slot remain a small candidate for a future
  tagged change, not part of this simplification.

## Settled

- Availability/roster alternatives, calibration and box-score ablations:
  [v5 audit](diagnostics/v5_audit/RESULTS.md),
  [History backtest](diagnostics/history_backtest/RESULTS.txt).
- Carryover/early roster and unified model:
  [early roster](diagnostics/early_roster/RESULTS.txt),
  [unified](diagnostics/unified/RESULTS.txt).
- Opening-price hypotheses and shrinkage:
  [DraftKings backtest](diagnostics/dk_hypotheses/RESULTS.txt),
  [ATS signal](diagnostics/ats_signal/RESULTS.txt),
  [shrinkage](diagnostics/open_shrinkage/RESULTS.txt).
- Margin/calibration alternatives: [margin model](diagnostics/margin_model/RESULTS.txt),
  [model review](diagnostics/model_review/RESULTS.txt).
- Kalshi and live trading: [benchmark](diagnostics/kalshi_benchmark/RESULTS.txt),
  [strength vs reaction](diagnostics/strength_vs_reaction/RESULTS.txt).
- Fantasy and Sleeper signals: [fantasy](diagnostics/fantasy_signal/RESULTS.txt),
  [Sleeper](diagnostics/sleeper_signal/RESULTS.txt).
- Pace, totals (O/U) and the picks' timing under v11:
  [v11 review](diagnostics/v11_review/RESULTS.txt),
  [totals fit](diagnostics/totals_model/RESULTS.txt).

Detailed dated entries, original registrations and decisions remain in
[the research history](docs/research_history.md). Scripts and results stay;
closed workflows can be recovered from git history.

## Workflows

Exactly eight remain: `build`, `tests`, `fit-model`, `backfill`, `rebuild`,
`box-history`, `history-backtest`, `margin-model`. Fifteen closed research
workflows are retired. Fit model already fits availability.
