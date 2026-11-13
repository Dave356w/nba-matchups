# Claude — working agreement for NBA Matchups

The owner directs the product. Be rigorous, inspect the code and current data,
state uncertainty, and turn findings into concrete tests.

## Live model and forward tests

### Realized WP shrunk cell + drift (v4) — owner instruction, November 13, 2026

The owner asked whether Realized WP can stay on the latest 200 cell results
and still respond to the live locks, then asked for the v4 patch.
**Realized WP = this game's no-vig DraftKings WP + (cell win rate − cell mean
no-vig WP) × n/(n+200) + drift**, clipped to 0–100%. The cell is v3's latest
200; the drift is one number per slate, an exponentially weighted mean of
(won − no-vig WP) over the latest 600 picks of every cell and stage,
half-life 100 picks, 200 pseudo-games at zero. The four constants are fixed.
`research/shrunk_drift_backtest.py`: better than v3 on both replay seasons
(2025-26 paired Brier −0.0033 ± 0.0019), level with the price, not better
than it; the constants were picked in-sample the same day. Cards state the
shrunk cell adjustment and the drift; the guide explains both. The shadow
record continues as v4 with the drift's pooled picks saved as their own
content-addressed proof; v3 closes, kept and never pooled. Fixed terms and
the primary reading are in `docs/realized_wp_forward_registration_v4.md`,
committed before implementation. A positive drift raises every card of a
slate together: describe it as recent results against prices, never as an
edge. The ROI matrix and its switch, model, locks, grades and records are
unchanged.

### Realized WP window 200 (v3) — owner instruction, November 13, 2026

The owner asked to update Realized WP with the best researched window. The
matrix window backtest's pre-set reading picks **200** (lowest walk-forward
Brier against the price on 2025-26 DraftKings replay; no window beats the
price). Cards and the guide show the band-adjusted estimate over the latest
200 cell results; everything else in v2 is unchanged. The shadow record
continues as v3; v2 closes, kept and never pooled. Fixed terms are in
`docs/realized_wp_forward_registration_v3.md`, committed before
implementation. The ROI matrix switch (Latest 50 default), model, locks,
grades and records are unchanged.

### ROI matrix window switch — owner approval, November 13, 2026

ML stats' ROI matrix gets a window switch: **Latest 50** per cell (default,
the owner's earlier selection), **Latest 100** per cell, and **All games** in
the selected source and season. The switch applies to every panel and keeps
the existing filters, ordering (slate date, game ID), stake, price timing and
exclusions. Card Realized WP keeps its fixed latest-50 window. Presentation
only: no rule, record, lock, grade or data change.

### Public games 1+ and source simplification — owner approval, November 13, 2026

The owner asked to combine games 1–9 and 10+ and to drop the mixed source on
the public pages, for clarity. Today, Results, ML stats and How it works show
one public record of every native v11 five-hour lock (and, under its own
source, eligible v11 replay) from **game 1 on**: early and established games
together. Opening games (game 0) and unknown stages stay out of that record;
openers still appear on Today as forecasts. Sources are **Saved live locks**
(default) and **Historical replay**, never pooled: the Mixed option and the
mixed cumulative headline leave the public pages. Running ROI on Results is
per season × source over games 1+. This supersedes, for public pages only,
the ML focus amendment's mixed headline and separate public stages, the
public model-prediction record's stage split and the public clarity
amendment's mixed source filter. The frozen 10+ ML rule, its internal
records, `data/ledger_report.txt`, the tracked tests, Realized WP, locks,
grades and data are unchanged. Pages say once that the public record
includes early-season games and that the registered rule counts 10+ only.

### Model–market blend shadow record — owner approval, November 13, 2026

The owner registered a fixed blend of lock model P and lock no-vig DraftKings
q as internal shadow research: P_blend = σ(logit q + 0.40 · (logit P − logit
q)), `analysis.shrink_p` at the existing SHRINK_W, never refitted. Native v11
five-hour locks (`model_predictions`, voids out, both lock prices) from the
2026-11-13 slate; games 1–9 and 10+ separate; no backfill. Primary reading:
games 10+, 2026-27, blend minus lock-price log loss at 95%. Internal only: no
card, page or guide display. Fixed terms are in
`docs/blend_wp_forward_registration.md`, committed before implementation.
The model, locks, Realized WP, ML record and tracked tests are unchanged.

### Realized WP band adjustment (v2) — owner instruction, November 12, 2026

After a sanity check showed the raw cell rate is biased by where a game sits in
its market band, the owner sent the proposed fix back as the instruction:
Realized WP = this game's no-vig DraftKings WP + (cell wins/games − the cell's
mean no-vig market WP), clipped to 0–100%. Same cells, latest-50 window,
seeding, pooling and exclusions; no shrinkage. Cards and the guide show this
value under the same name with the cell's W/L and market average. The shadow
record continues as a new version; v1 closes, kept and never pooled. Fixed
terms are in `docs/realized_wp_forward_registration_v2.md`, committed before
implementation. The ROI matrix, model, locks, grades and records are unchanged.

### Realized WP shadow forward record — owner approval, November 12, 2026

The owner explicitly requests immutable pre-tip Realized WP capture and matched
model/market accuracy and ROI tracking, shadow research only. Fixed terms are
in `docs/realized_wp_forward_registration.md`, committed before implementation.
Use the first saved valid capture within five hours of tip; retain all pre-tip
captures; +EV flat 1u, either side, at most one per game. No historical backfill,
public forward section, model change, or alteration of existing ML records.

### Break-even headline — owner approval, November 12, 2026

Display "Price required break even" as a metric label with its calculated
percentage in the same prominent style as Realized WP, above Actual result.
This replaces the sentence format only; pricing and calculations are unchanged.

### ML stats ROI matrix — owner approval, November 12, 2026

Replace the Winner accuracy by stated win chance table with a joint model-WP
(row) × no-vig market-WP (column) ROI matrix. The owner selected latest 50 per
cell. Apply the existing stage/source/season filters before the cap. Each result
stakes 1 unit on the saved model winner at its source price; show ROI, count,
W/L and net units. Include market-underdog/disagreement cells. Pending and
unpriced games do not enter. Use lock P/prices for live and historical model
P/closing prices for replay. Label filter/window scope; card Realized WP pools
its own previous/current-season calibration population. No rule or data changes.

### Card layout refinement — owner approval, November 12, 2026

Name the display Realized WP (owner follow-up). Keep Model probability at top left; replace the large DraftKings moneyline
with DraftKings no-vig probability at top right. Replace the duplicated market
readout with Rolling calibrated WP. Keep its pooled W/L and season composition,
put break-even text above Actual result, and remove the old duplicate calibration
block, latest-50/pooling boilerplate, inline How calculated link and "likely
winner may still be poor value" sentence from the summary. The guide retains
the method and all calculation and tracking rules remain unchanged.

### Rolling matrix amendment — owner approval, November 12, 2026

The owner selected the latest 50 results per joint model-WP × market-WP cell,
seeded with the immediately previous season and updated with completed earlier
current-season slates. This supersedes the 100-market-prior display below.
Use actual cell wins/games without shrinkage; an empty cell is unavailable.
Current-season evidence remains valid native v11 five-hour locks; historical
seeding uses eligible v11 replay at its DraftKings close. Exclude same-season
replay, older seasons, pending/void games and same/later slates. Preserve native
precedence and one observation per game. Order by date and game ID. Pool early
and 10+ calibration evidence across all teams; openers remain separate. Include
model/market disagreement and market-underdog cells. Show actual W/L, window
size and season composition, and explain the historical close/live-lock timing
difference. No production model, tracked rule, lock, grade or record changes.

### In-season calibrated WP — owner approval, November 12, 2026

The owner requested the exact joint-cell estimate on each game card:
(wins + 100 × current no-vig DraftKings WP) / (games + 100).
Use only graded native v11 five-hour locks on strictly earlier slates in the
same season, pooling early and 10+ calibration evidence (owner follow-up),
with the model's selected team defining
both probabilities and the observed result. Cells use model 50/60/70/80/100%
and market 0/50/60/70/80/100% boundaries, lower inclusive (100% in the last).
Opening games remain separate. No replay or prior-season seeding. Display count and W/L, explain empty cells
and smoothing, and label the estimate experimental in the guide. This is a
separate pregame display: it does not change the model, saved picks, grades,
record membership, or pricing. Preserve historical and frozen data.

### Public clarity amendment — owner approval, November 12, 2026

The owner approved all six site-review suggestions and merge after validation.
This supersedes earlier presentation defaults: Performance now defaults to saved
live locks, with live/replay summaries side by side and the mixed cumulative
record retained in the source filter. Today distinguishes model probability,
DraftKings price and raw break-even probability, consolidates the saved pick,
combines separate stage/status filters, and uses a compact mobile introduction.
Per-game model and no-vig market probabilities refer to the same team beside
its actual Win/Loss; market timing is explicit. Realized win rates remain
aggregate measures, never invented probabilities for individual final games.
Explain accuracy, ROI and probability quality in plain language. No selection,
model, saved price, lock, record membership, grading or data workflow changes.


### Public ML-only amendment — owner approval, November 10, 2026

The owner explicitly requested removing tracked betting tests from every
public page. This supersedes the expandable-paper-section instructions below.
Today, Results, Performance and How it works show ML forecasts/records only:
no spread, value-favorite, totals or retired-test selections, rules, tables,
charts, links or notes, including inside collapsed details. These tests keep
all existing internal data, registered rules, ongoing grading, workbench and
report records. No model, snapshot, mixed-ML membership or grading changes.

### ML focus amendment — owner approval, 2026-11-10T19:37:57Z

This amendment supersedes conflicting public-presentation instructions below.
The owner approved making ML forecasts the public focus, moving spread and
total selections to paper tracking, preserving every existing result, and
freezing the ML rule, with the UI centered on the games-10+ model.

- **Frozen ML rule.** For each native
  v11-family prediction at its existing valid five-hour lock, select the team
  with the higher saved win probability (home at exactly 50%). Stake a flat
  1 unit at the saved DraftKings moneyline, with both lock prices available.
  Both teams must have completed at least 10 games before the slate. No edge,
  favorite, price-band or confidence filter; missing locks/prices abstain.
  Existing stale-log voids still apply. Later forecasts and closing prices
  cannot replace a lock. Future model families need their own amendment.
- **Owner amendment during implementation:** pool historical and new games
  within the same v11 model family in one cumulative ML record. Label the
  historical rule selection honestly; do not describe the combined record
  as a pre-registered forward test. The owner explicitly includes reconstructed
  and rebuilt v11 replays, labeled as a mixed record. Pool seasons in the
  headline with season and live-only filters available; keep stages separate.
  Replays use their DraftKings closing prices, native picks their saved lock
  prices, and the UI names both timings. Only graded v11 replays with a usable
  DraftKings pair qualify. Native IDs take precedence, including missed/void
  locks; rebuilt takes precedence over a duplicate reconstructed replay.
  Pooling is a presentation/monitoring choice, not evidence that replay is live.
  Early/opening/unknown-stage forecasts
  remain available separately and never enter the games-10+ forward test.
- Today defaults to games 10+, with one primary ML forecast and its locked
  selection/price when available. Early games remain accessible. Results
  defaults to games 10+; Performance leads with cumulative v11 ML return,
  W/L, uncertainty and matched market benchmarks. Keep historical ML and
  other stages accessible with explicit labels; do not imply proven profit.
- Spread H5·3, value-favorite H2·F·5 and totals OU retain their unchanged
  rules, snapshots, grades and future paper tracking, accessible in secondary
  paper-record views. They are no longer public betting recommendations.
  H4 and all previously retired rules stay retired under their existing
  terms. No data, old results or models are deleted or rescored.
- No prediction maths, fitting, tags, ledger schema, lock timing, data-write
  workflow or pricing changes. Validate data, run the full test suite and
  desktop/mobile browser checks before publication through GitHub Pages.

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

- **Edge floors and O/U test amendment (owner's decision, registered 2026-10-29T16:13:38Z).**
  Committed on its own before any floor or O/U code and before any v11
  lock: the native ledger at `699e65e` has no lock on the 2026-10-29 slate
  (first lock window opens 2026-10-29T19:00Z) and no lock tag ending `_v11`.
  Evidence: [v11 review](diagnostics/v11_review/RESULTS.txt),
  [totals fit](diagnostics/totals_model/RESULTS.txt),
  [totals improvements](diagnostics/totals_improve/RESULTS.txt).
  - **Floors.** The public tracked tests, from the 2026-10-29 slate at the
    5 h lock, v11 family (lock tag ending `_v11`), test × stage × season:
    **H5·3** spread, H5's rule (13.5 × Φ⁻¹(P), both spread prices
    −135..+115) with edge **≥ 3 points**; **H2·F·5** value favourite, q > 0.5
    with P − q **≥ 0.05**; **H4** unchanged (P − q ≥ 0.12); each with its
    games 1–9 (·E) and 10+ rows (keys H5·3·L·v11, H5·3·E·L·v11,
    H2·F·5·L·v11, H2·F·5·E·L·v11). The floors follow cost, not a
    return scan: the model's disagreement is ~0.3–0.5 right at the open, so
    a gap needs ~0.05 to clear half the hold; spread line moves appear only
    from ~3 points. The v11-family rows at the old floors (H5·L·v11 edge ≥ 1,
    H2·F·L·v11 any gap) continue internally as their supersets, never on
    public pages, never pooled.
  - **O/U test (OU).** Model: the totals fit T* of the research (pregame
    decayed pace and points per 100 at half-life 25 plus a 5-game form
    term; prior = last season regressed 40%, weight 10 games; league level
    to date with last season as 30 games; back-to-backs; season phase;
    fitted by least squares relative to the league's level on completed
    seasons, refitted by Fit model). P(over) = 1 − Φ((line − T) / σ), σ the
    fit's residual sd for games 1–9 or 10+. Inputs use games strictly
    before the slate; a missing input abstains. Tag
    `totals_hl25_form5_ou1`. Market: the pregame book's main-line total
    (both prices −135..+115), no-vig P(over) q. Rule: at the 5 h lock (the
    first pregame snapshot within 5 hours of tip with a model total and a
    main-line total; write-once `lock_ou_*`), track the model's side when
    |P(over) − q| ≥ 0.05; games 1–9 (OU·E·L) and 10+ (OU·L) separate,
    game 0 never, per season; DraftKings prices; a push returns the stake;
    CLV = lock total against the close, in points toward the bet. First
    snapshots of the total are recorded write-once (`first_ou_*`) as history.
  - The O/U test is its own record, never pooled with H5·3, H2·F·5 or H4.
    No hindsight O/U rule beat its null at the close; the signal found was
    at the open, and the owner chose the 5 h lock knowingly.
  - Public Today shows each game's model total, the book's total and the
    lean as a forecast, and the tracked O/U pick once it locks; Results and
    Performance show the four tests (H5·3, H2·F·5, H4, OU) by stage and
    season; How it works states the floors and the O/U rule. No hindsight
    figures on public pages. v11's prediction maths, native snapshots and
    tags are unchanged.

- **DFS lineup page (owner's decision, 2026-10-29T21:29Z).** Committed on
  its own before any DFS code. A fifth public page, `dfs.html` ("DFS
  lineups"), joins the four approved pages: a Yahoo daily-fantasy lineup
  optimizer for the day's slate, a separate tool, never a tracked test.
  - Inputs: Yahoo's salaries, positions and cap (its public NBA players
    feed, snapshot to `data/nba_yahoo_dfs.csv`) and Sleeper's latest
    pregame projections (`data/nba_sleeper.csv`), converted to Yahoo
    scoring (PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK − TO; it reproduces
    Yahoo's season FPPG exactly for 55 of 74 players, the rest within a stat
    correction). Sleeper stays out of the model, the tests and the records;
    this page is its one public use. A player Sleeper lists at 0 minutes or
    Out projects 0; an unmatched player has no projection.
  - Version 1, classic multi-game contests: PG, SG, G, SF, PF, F, C, UTIL
    under the feed's cap, players from at least 3 teams and at most 6 from
    one (Yahoo's rules). The visitor's browser finds the best lineups exactly
    (lock, exclude, edit projections, exposure caps); players whose game has
    started are out.
  - No model probability, tracked pick or record appears on it, and it
    feeds nothing back. Yahoo or Sleeper failures never stop a build or
    alter any forecast. The page says the projections are Sleeper's.

- **Stale game-log locks amendment (owner's decision, registered 2026-11-03T07:12:59Z).**
  Committed on its own before any void or lock code: the native ledger at
  `2ef14d0` has no lock after 2026-11-02T22:02:34Z and no slate after
  2026-11-02. Evidence: [stale game-log locks](diagnostics/stale_log_locks/RESULTS.txt).
  - Eight 5 h locks were taken while a team's saved game log was behind its
    graded finals (its previous game missing, b2b unset): 2026-10-31 PHI @ CHO
    (401909897); 2026-11-02 ATL @ IND (401909909), BRK @ NYK (401909910),
    WAS @ ORL (401909911), BOS @ HOU (401909916), PHO @ GSW (401909919),
    TOR @ LAC (401909920) and MIA @ LAL (401909921). The list comes from that
    rule alone (games played at each lock commit against the saved box
    scores; no result, price or probability read) and covers every lock to date.
  - **Void.** These eight locks leave every tracked test (H5·3, H2·F·5, H4, OU,
    each stage and internal superset), its CLV and the public model-prediction
    record. Their first and lock values stay in `data/nba_ledger.csv`,
    unchanged, and are listed internally with the bets they made. No game is
    re-locked or re-priced; the complete-log replay is hindsight, information
    only. At registration the void removes H5·3·E·L·v11 3–1 (+1.80u),
    H2·F·5·E·L·v11 1–0 (+0.74u), H4·E·L·v11 0–1 (−1.00u), OU·E·L 0–3 (−3.00u)
    and model predictions 5–3.
  - **Lock condition.** From the next slate, the 5 h lock and the O/U lock are
    each the first pregame snapshot within 5 hours of tip, with its existing
    requirements, whose game logs are current: neither team's log ends before
    its latest graded final strictly before the slate. A snapshot with a stale
    log still refreshes the forecast and may be the first snapshot; it does not
    lock, and later builds retry. A game with no such snapshot before tip has
    no tracked bet and counts internally as a missed lock.
  - Rules, thresholds, prices, stages, model maths and tags are unchanged.
    Pages carry one dated line that eight locks on October 31 and November 2,
    2026 were voided for incomplete game logs.

- **O/U model ou2 amendment (owner's decision, registered 2026-11-03T23:05:00Z).**
  Committed on its own before any ou2 code or snapshot: the native ledger at
  `3503c29` (containing `main` at `473d97f`) has 39 O/U locks, the last at
  2026-11-02T22:02:34Z, every lock tag `totals_hl25_form5_ou1`. Evidence:
  [totals report availability](diagnostics/totals_report/RESULTS.txt) (arm
  rep5h: walk-forward squared error −2.54 ± 2.42 on games 1+ of 2023-24 ..
  2025-26, the selection fixed before its run; at the close the model still
  trails both books), [totals talent](diagnostics/totals_talent/RESULTS.txt).
  - **Model ou2** = the totals model T* plus injury-report availability:
    av_o / av_d = both teams' summed av_bpm valued by last season's OBPM /
    DBPM, av_m = their summed av_min, from the latest NBA report at the
    snapshot with v11's Questionable weighting at its lead (Out 0;
    Doubtful, Questionable, Probable their play rates; Available and
    unlisted previous-box players 1). The terms enter by least squares on
    T*'s residual (total − T), fitted on the report seasons 2022-23 ..
    2025-26 at the 300-minute lead, σ by stage its residual sd; refitted by
    Fit model. A game the report does not cover, or a missing term,
    contributes zero under its own neutral tag. Tags
    `totals_hl25_form5_rep_ou2` and `totals_hl25_form5_rep_neutral_ou2`.
  - The O/U rule is unchanged: at the 5 h lock with current game logs, the
    model's side when |P(over) − q| ≥ 0.05, main-line DraftKings totals
    (both prices −135..+115), games 1–9 and 10+ separate, game 0 never, per
    season, a push returns the stake, CLV = lock total against the close.
  - **Record.** The O/U test restarts with ou2 in new rows OU·L·ou2 and
    OU·E·L·ou2, from the 2026-10-29 slate. Locks taken before activation
    (tag ending `_ou1`) are rescored once with ou2 at their recorded lock:
    the lock's book, line, prices and T* total, the latest NBA report at or
    before the lock snapshot, box scores and game logs strictly before the
    slate, and last season's BPM; nothing after the lock is read. These
    values are written once to `data/nba_ou_rescore.csv` (keyed by game id);
    the native first_ou_* and lock_ou_* columns stay unchanged. From
    activation, locks carry ou2 natively. The voided locks stay void. The
    ou1 rows OU·L and OU·E·L close with the last slate before activation and
    stay graded in the internal workbench and `data/ledger_report.txt`,
    never pooled with the ou2 rows.
  - Public Today, Results and Performance show only the ou2 rows, with one
    dated line that O/U picks before activation were scored with the current
    totals model from the information available at each lock. How it works
    describes ou2. The margin model, its tests and their records are
    unchanged.
  - Presentation (owner's decision, 2026-11-03, before activation): that
    dated line appears once, as a clause in How it works' O/U rule, not on
    Results or Performance.

- **O/U model ou3 amendment (owner's decision, registered 2026-11-06T21:10:26Z).**
  Committed on its own before any ou3 code or snapshot: the native ledger at
  `e683b28` (containing `main` at `a249715`) has 62 O/U locks, the last at
  2026-11-06T20:02:31Z, every lock tag ending `_ou1` or `_ou2`, none `_ou3`.
  Evidence: [totals day-of-week and rest](diagnostics/totals_dow/RESULTS.txt)
  (REST arm: walk-forward squared error on the total −0.25 ± 0.71 points², games
  1+, new era 2017-18 .. 2025-26 minus the COVID seasons, better in 4 of 7; rest
  slope +0.8 to +1.1 points per day of rest in every fold from 2018-19). The
  registered rest-days forward test was not run: the owner adopted the term
  knowingly, its expected gain (~0.03 points of RMSE) being far below what a
  season of forward games can detect.
  - **Model ou3** = ou2 plus `rest_tot`, both teams' days of rest (days since the
    team's previous regular-season game minus 1, capped at 3; a team's first game
    of the season counts 3), from game logs strictly before the slate, in T*'s
    least-squares stage with rel, form, b2b and phase, on Fit model's seasons; the
    report stage (ou2's terms and lead) is then refitted on ou3's T* residual,
    σ by stage as before. A game without report terms keeps T* under its own
    neutral tag. Tags `totals_hl25_form5_rep_rest_ou3` and
    `totals_hl25_form5_rep_rest_neutral_ou3`.
  - The O/U rule is unchanged: at the 5 h lock with current game logs, the
    model's side when |P(over) − q| ≥ 0.05, main-line DraftKings totals (both
    prices −135..+115), games 1–9 and 10+ separate, game 0 never, per season, a
    push returns the stake, CLV = lock total against the close.
  - **Record.** The O/U test restarts with ou3 in new rows OU·L·ou3 and
    OU·E·L·ou3, from the 2026-10-29 slate. Locks taken before activation (tags
    ending `_ou1` or `_ou2`) are rescored once with ou3 at their recorded lock:
    the lock's book, line and prices, game logs and box scores strictly before
    the slate, the latest NBA report at or before the lock snapshot, last
    season's BPM; nothing after the lock is read. These values are written once
    to `data/nba_ou3_rescore.csv` (keyed by game id); native first_ou_* and
    lock_ou_* columns and `data/nba_ou_rescore.csv` stay unchanged. From
    activation, locks carry ou3 natively. The voided locks stay void. The ou2
    rows OU·L·ou2 and OU·E·L·ou2 close with the last slate before activation and
    stay graded in the internal workbench and `data/ledger_report.txt`, never
    pooled with the ou3 rows.
  - Public Today, Results and Performance show only the ou3 rows. How it works
    describes ou3, with the dated clause in its O/U rule that O/U picks before
    activation were scored with the current totals model from the information
    available at each lock (as for ou2), not on Results or Performance. The
    margin model, its tests and their records are unchanged. Activation waits
    on the post-change checks (rescore, rebuild, build, model review, DraftKings
    and History backtests), reviewed before activation.

- **Large probability gap (H4) retirement (owner's decision, registered 2026-11-10T05:55:00Z).**
  Committed on its own before any retirement code: the native ledger at
  `f98459c` has 154 rows, the last lock at 2026-11-09T22:02:30Z, all eight
  games of the 2026-11-09 slate locked, no lock on a later slate and no
  pending H4 bet.
  - **Retire.** H4's v11-family rows H4·L·v11 and H4·E·L·v11 close with the
    2026-11-09 slate: games from the 2026-11-10 slate on make no H4 bet. At
    registration they stand H4·E·L·v11 0–7 (−7.00u; null −4.0%, same-game
    favourite +41.5%, CLV +1.09 pp ± 0.77) and H4·L·v11 no bets. The closed
    rows keep their bets, grades and CLV in the internal workbench and
    `data/ledger_report.txt`, never pooled, like H4's earlier closed rows.
    Nothing is rescored or deleted; lock values in `data/nba_ledger.csv`
    are unchanged.
  - Reason, stated as it is: the owner retired the test after reading its
    forward record (0–7, about 1.8 SE below its null on a null-based error;
    the registered ± 1 SE of 0.0 is degenerate). This is a decision made on
    outcomes, not a pre-set stopping rule, and seven bets do not show the
    rule loses. H4 is not replaced, re-thresholded or inverted; a later
    test on model–market gaps needs its own registration before any bet.
  - **Public pages.** Today, Results and Performance show three tests
    (H5·3, H2·F·5, OU) by stage and season; H4's picks, record and rule
    leave every public page. Results, Performance and How it works carry one dated
    line that a fourth test, large probability gap, was retired after the
    November 9, 2026 games and that its picks are kept separately and not
    included. No H4 figure appears on a public page.
  - The margin model, v11's maths, snapshots, tags, the other tests' rules,
    thresholds and records, and the model-prediction record are unchanged.

## Sources of truth and engineering rules

- **Public matchup context (owner request, 2026-11-01).** Today cards show
  each team's current-season W/L beside its abbreviation. Game details show
  season-to-date offensive and opponent four factors from saved game logs,
  plus captured ESPN player injuries, statuses, details and timestamps.
  Records and factors use completed regular-season games strictly before
  the slate and require complete coverage against the forecast's GP count.
  Factors use raw season totals, clearly distinguished from the model's
  faded/decayed inputs. Missing data stays unavailable. Injury lists are
  display context, labeled ESPN and distinct from the NBA report used by
  the model. This supersedes their research-only presentation restriction
  below. No model, rule, threshold, snapshot or grade changes.

- **Public model-prediction record (owner request, 2026-10-31).** Add all
  native v11 winner predictions at their saved five-hour locks, including
  forecast-only games, to Today, Results and Performance. This explicitly
  extends the prior public-only-registered-tests presentation rule. Keep
  prediction W/L separate from betting returns. Read lock P/tag/time only;
  never rescore or substitute the latest forecast. Both routes and neutral
  tags count. No thresholds or prices are required for forecast membership.
  Keep season and stage separate; opening games and unknown stages get
  distinct rows. Pending games do not enter accuracy. Show Wilson 95%
  intervals and matched closing-favourite W/L, log loss and Brier scores.
  Missing closes stay in the full model record. Earlier versions, rebuilt
  rows and research records stay outside the public forecast record.

- **Moneyline return on model predictions (owner request, 2026-11-09).**
  Results' model-prediction table adds each pick's flat 1-unit return at
  its frozen DraftKings lock moneyline and a running ROI (return divided
  by stakes through that game, tip order) within its season × stage, with
  each season × stage's figure to date beside its ± 1 SE and no-vig
  baseline. This extends the 2026-10-31 rule that prediction W/L stays
  apart from betting returns, on Results only. It is every prediction with
  no threshold, not a registered test, and was first shown after 70 graded
  results were known: descriptive, never evidence for a rule or pooled
  with a tracked record. A lock without both DraftKings prices keeps its
  W/L and has no return. Lock values only; nothing rescored. Performance's
  prediction record and the tracked tests are unchanged.

- **Winner accuracy by stated win chance (owner request, 2026-11-12).**
  Each Performance view (stage × source × season) adds one table on its
  graded games with a usable DraftKings closing pair: the model's picks by
  the model's saved win probability beside the closing favourites by their
  no-vig implied probability, each with W/L, accuracy and mean stated
  probability (`analysis.prediction_confidence_bands`). Bands were fixed
  before the table was drawn (5 points to 80%, 80–90%, 90%+). Both columns
  hold the same games in total, not row by row. Lock P for live picks,
  replay P for replays; nothing rescored. Descriptive only: never a filter,
  rule or evidence for one, and the ML rule keeps no confidence filter.

- **Live win chance (owner request, 2026-11-07).** Today cards show each
  team's in-game win chance while a game is in progress, computed in the
  reader's browser from the live ESPN score, period and clock the page
  already fetches and the game's DraftKings spread frozen at tip
  (`pre_spread`): P(home) = Φ(a(τ)·lead/√τ + b(τ)·μ·√τ), μ = −pre_spread,
  τ = regulation left / 48 min (overtime: the period's time left / 48 min).
  `live_wp.py` holds the formula and its browser copy; `model/live_wp.json`
  the coefficients, fitted by `research/live_wp.py` on ESPN play-by-play
  and archived pregame lines of 2015-16 .. 2025-26 and chosen by held-out
  log loss ([live win chance](diagnostics/live_wp/RESULTS.txt)). Display
  only: never written to `data/`, never a model input, rule, snapshot,
  grade or record. It is labelled apart from the pregame model forecast
  (still the card's one primary probability) and from tracked picks, and
  is absent without a DraftKings spread, before tip and after the final.
  The open page re-fetches ESPN's scoreboard every 30 s while one of its
  games is in progress (owner request, 2026-11-07), at each tip before
  that, and not once every game is final; a hidden tab waits.
  Today lists live games first, then upcoming, then final (tip order within
  each), re-sorting as the overlay changes a game's state (owner request,
  2026-11-07).
  The public pages' slate rolls over at 6:00 ET, not midnight: before then
  they show the previous ET slate when it had games, so late West Coast games
  stay on Today; grading and snapshots keep the ET date (owner request,
  2026-11-07).

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
- `data/nba_box_<season>.csv` and preseason boxes are model inputs. ESPN
  injury lists (`data/nba_injuries.csv`) are public display context under the
  2026-11-01 request above, never presented as the NBA injury-report inputs.
  Sleeper projections (`data/nba_sleeper.csv`, first/latest snapshots) stay
  research-only apart from the approved DFS page. Their
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
- Early-season calibration and home lean; watch without fitting to thin data
  (repeatable check: `research/model_calibration_check.py`,
  [model calibration](diagnostics/model_calibration/RESULTS.txt)).
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
  [model review](diagnostics/model_review/RESULTS.txt). A walk-forward stretch of P
  (one scale on m / sigma, learned from earlier seasons) adds nothing:
  [calibration stretch](diagnostics/calibration_stretch/RESULTS.txt).
  The model-WP x market-WP cell carries no walk-forward signal at any
  window (10 .. all); larger windows only sit closer to the price:
  [matrix window](diagnostics/matrix_window/RESULTS.txt).
- Kalshi and live trading: [benchmark](diagnostics/kalshi_benchmark/RESULTS.txt),
  [strength vs reaction](diagnostics/strength_vs_reaction/RESULTS.txt).
  In-game win probability from the score and the pregame line (display only):
  [live win chance](diagnostics/live_wp/RESULTS.txt), and against Kalshi's
  live price ([live win chance vs Kalshi](diagnostics/live_wp_kalshi/RESULTS.txt)),
  with the model's margin ([model adds nothing](diagnostics/live_wp_model/RESULTS.txt)),
  and team quarter patterns ([quarter profiles](diagnostics/quarter_profiles/RESULTS.txt),
  [stabilization](diagnostics/quarter_stabilization/RESULTS.txt), [live volatility](diagnostics/live_wp_pace/RESULTS.txt)),
  and comeback ability ([comebacks](diagnostics/comebacks/RESULTS.txt)).
- Fantasy and Sleeper signals: [fantasy](diagnostics/fantasy_signal/RESULTS.txt),
  [Sleeper](diagnostics/sleeper_signal/RESULTS.txt).
- Pace, totals (O/U) and the picks' timing under v11:
  [v11 review](diagnostics/v11_review/RESULTS.txt),
  [totals fit](diagnostics/totals_model/RESULTS.txt),
  [totals improvements and market-context medians](diagnostics/totals_improve/RESULTS.txt).

Detailed dated entries, original registrations and decisions remain in
[the research history](docs/research_history.md). Scripts and results stay;
closed workflows can be recovered from git history.

## Workflows

Exactly nine remain: `build`, `watch`, `tests`, `fit-model`, `backfill`,
`rebuild`, `box-history`, `history-backtest`, `margin-model`. `watch` writes
no data: it long-polls each slate and dispatches `build` at every game's
five-hour lock window (`watch_slate.py`), since GitHub drops most scheduled
builds. Fifteen closed research workflows are retired. Fit model already
fits availability.

DFS uncertainty also refits inside the serialized Build workflow, once per
ET morning from 08:00 ET after the latest prior slate with stored Sleeper
lines is fully graded. It runs after native grading and pregame snapshots
are saved; historical `--date` builds skip it. New candidates use the
growing native history and the existing chronological holdout gate. A
rejected candidate updates diagnostics but retains the last accepted fit.
The same build renders accepted changes and commits only the DFS fit/report
alongside its normal data. A timeout or failed refit must not prevent
snapshot commits or publication. No extra workflow or tracked-model refit.
