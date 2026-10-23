# Claude — working agreement for NBA Matchups

The owner directs the product. Be rigorous, inspect the code and current data,
state uncertainty, and turn findings into concrete tests.

## Live model and forward tests

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
figures and ungraded native bets are frozen under Ledger → Retired and in
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
- Cards show one primary probability and margin. Shrunk edge/EV (w=0.40)
  and descriptive price windows are display only; graded rules use the raw
  frozen first P. The display margin never replaces H5's frozen 13.5 conversion.
- Keep site-build serialized with cancel-in-progress false and its data
  validate/commit on !cancelled(), even when rendering fails. Tests gate PRs,
  never block a pregame snapshot. Other data-writing workflows use their own
  queue and retry pushes.
- `data/nba_box_<season>.csv` and preseason boxes are model inputs. Injury
  lists (`data/nba_injuries.csv`) shown on cards and Sleeper projections
  (`data/nba_sleeper.csv`, first/latest snapshots) stay research-only. Their
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

Detailed dated entries, original registrations and decisions remain in
[the research history](docs/research_history.md). Scripts and results stay;
closed workflows can be recovered from git history.

## Workflows

Exactly eight remain: `build`, `tests`, `fit-model`, `backfill`, `rebuild`,
`box-history`, `history-backtest`, `margin-model`. Fifteen closed research
workflows are retired. Fit model already fits availability.
