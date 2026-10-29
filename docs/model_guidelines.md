# Model guidelines: v9, one model and three first-snapshot tests

Updated 2026-10-23. Current evidence: [model review](../diagnostics/model_review/RESULTS.txt),
[DraftKings backtest](../diagnostics/dk_hypotheses/RESULTS.txt),
[margin model](../diagnostics/margin_model/RESULTS.txt), and
[early segments](../diagnostics/first_snapshot_registration/early_hindsight.txt).
These are hindsight results, not a forward track record. Research intervals
are 95%; the pages/report and early-segment printout explicitly use 1 SE.
The full earlier synthesis remains in [guideline history](model_guidelines_history.md).

## What the evidence supports

- V9 is a calibrated outcome forecaster with less information than the close.
  On games 10+, model minus close log loss is +0.0170 ± 0.0107 on DraftKings
  2025-26 and +0.0130 ± 0.0093 on ESPN BET 2024-25. Brier agrees. Model
  calibration slopes are 1.15 ± 0.19 and 1.07 ± 0.16 respectively.
- Any useful disagreement is earlier. The outcome's weight on the model's
  disagreement with the opening price is +0.45 ± 0.42 and +0.54 ± 0.38,
  versus +0.01 ± 0.45 and +0.07 ± 0.41 at the close, on those same seasons.
  The first snapshot is later than the open and may have lost part of that
  information. It is the executable forward test, not an opening-price replica.
- Early games remain thin: 138 ESPN BET games in each priced season, no
  DraftKings early hindsight. V9 early uses injury reports, unlike the
  retired carryover rules; injury-timing uncertainty applies to it too.
- Margin fitting retains probability accuracy while improving the margin
  interpretation. The displayed margin is the direct v9 prediction. H5
  keeps its registered 13.5 × Φ⁻¹(P), regardless of the display margin.
- Blanket value and closing-price spread bets were retired. Claimed EV,
  raw disagreement, a winning price band or an isolated high-ROI cell is
  not evidence of an edge. Many closed box-score variants added nothing.
  See the dated [research history](research_history.md) and its result links.

## Live setup

One v9 model is used from game 0, with a base arm and a covered NBA
injury-report arm from game 1. The rating fades last season out by game 10.
The two deployed fits are `model/logit.json` and `model/logit_avail.json`;
they predict margin by least squares and P = Φ(margin / residual sigma).
No carryover/v4 fallback is loaded. Missing optional talent/luck/FT terms
contribute zero; missing injury inputs contribute no injury terms. These
cases get distinct neutral-input v9 tags. Missing core ratings abstain.
Complete-input predictions and tags are unchanged.

The pre-implementation inventory found zero fallback cases in 2,450 scored
reconstructed and 16 rebuilt rows. The empirical paired sample is therefore
n=0, not evidence of improvement. Controlled tests check the changed cases.
Historical carryover columns remain readable and unchanged; new rows leave
them blank. Frozen legacy fits and scoring stay research-only.

**v10** (owner's decision 2026-10-28, CLAUDE.md amendment): ORB% in the
composite, drafted rookies valued by pick, and the injury-report route
refitted before each slate (MODEL.md "v10"). It goes live when "Fit model"
runs with version v10; v9 snapshots keep their tags and P. The games 1–9
rows accept `_v9` and `_v10` first tags. Walk-forward on production routing,
v10 − v9 games 1+ log loss −0.0009 ± 0.0004; it still trails the close
(DraftKings 2025-26 games 10+ +0.0153 ± 0.0106).

**v11** (owner's decision 2026-10-28, CLAUDE.md amendment): v10 plus
Questionable weighting on the report route (MODEL.md "v11"). It failed the
G7 rule set before its test (DraftKings 2025-26 games 10+ +0.0007 at 5 h)
and was adopted by the owner regardless; its post-change checks are reviewed
before activation. The games 1–9 rows accept `_v11` tags too.

**Five-hour lock** (owner's decision 2026-10-28, CLAUDE.md amendment): the
tests bet the first priced snapshot within 5 h of tip, in new ·L rows; the
first-snapshot rows close at the lock's first slate. Never pooled.

**Edge floors and O/U test** (owner's decision 2026-10-29, CLAUDE.md
amendment): the public tests are H5·3 (spread edge ≥ 3 points), H2·F·5
(value favourite with P − q ≥ 0.05), H4 and a new O/U test (the totals
model's side when |P(over) − q| ≥ 0.05), all at the 5 h lock, v11 family,
by stage and season. The floors follow cost (the opening disagreement is
~0.3–0.5 right, so a gap needs ~0.05 to clear half the hold; spread line
moves appear only from ~3 points), not a return scan. The old-floor v11 rows
continue internally as supersets. Never pooled.

**Model-family reset** (owner's decision 2026-10-29, CLAUDE.md amendment):
v11 starts a new model family and the public records restart with it. From
the 2026-10-29 slate the tests run in v11-family rows (·L·v11) that need a
`_v11` lock tag on both stages; the ·L rows close with the 2026-10-28 slate.
Closed first-snapshot and ·L records stay graded internally, never pooled.
Rules, thresholds and prices are unchanged.

## Frozen forward tests (G4)

Registration: 2026-10-23T19:46:58Z, commit `2e301d3`, before CHI @ MIL was
final. Original games-10+ registrations and thresholds are unchanged.

| Test | Games 10+ row | Games 1–9 row | Rule at the frozen first snapshot |
|---|---|---|---|
| Spread | H5 | H5·E | 13.5 × Φ⁻¹(P); edge ≥ 1 point; both spread prices −135..+115 |
| Value favourite | H2·F | H2·F·E | Value side when no-vig q > 0.5 |
| Big value edge | H4 | H4·E | Value side when P − no-vig q ≥ 0.12 |

Games 1–9 and 10+ stay on separate rows, never pooled until at least a
season of each exists; later pooling needs an explicit amendment. Game 0
is excluded. Every native result is reported, win or lose. Thresholds,
prices and membership are not retuned on native outcomes. The early rows
read the recorded primary first P, never a carryover P. Old first tags/P
remain immutable, including a v8 snapshot captured before rollout.

H1/H6 and late-pregame H2/H3 are **Retired**, with registered hindsight and
native bets frozen in the retirement archive. None was graded at retirement.
They never receive new active grades. Early hindsight uses the 276 ESPN BET
v9 rows, explicitly information only, apart from native and DraftKings.

## Use and monitoring (G1–G6)

Public presentation (owner approval 2026-10-23): Today shows current forecasts
and explicitly frozen tracked picks. Results and Performance report only
registered native selections, separately by test, stage and season, and
since the model-family reset only the v11-family rows, with one dated line
that the record restarted on October 29, 2026. Keep
all qualifying wins, losses, pushes and pending picks visible. Do not make
a legacy all-game lean record the headline of the active tests. Retired
rules, reconstructed evidence, adjusted edge/EV, price windows and model
diagnostics remain in the internal workbench and research reports. Publish
an explicit four-page allowlist; research outputs are outside `public/`.

1. Read model P as a forecast, not a promise of profit. The no-vig close is
   the better probability where the comparison resolves. No-vig q differs
   from posted break-even: judge ROI/EV against the market-hold null.
2. Never size a bet by raw disagreement with the close. Cards display
   no adjusted edge/EV. Internal research retains w=0.40; this does not change model P or any rule.
3. Evaluate book × season × basis × stage separately, against the null
   and the same-game market favourite. Native is forward evidence;
   reconstructed and current-season rebuilt rows are hindsight. Never pool.
4. Track native closing-line value as an earlier signal than ROI. Expect
   lower effects than the opening-price hindsight, and wait for power.
   Established H5 needs roughly a season for the full hindsight effect;
   the other tests and early segments need more. Thin early cells do not
   justify replacing thresholds or adding filters.
5. Price bands, months, spread bands and card context windows are descriptive.
   Colour means a gap of at least 2 SE from its null, not a positive return.

## Change gate (G7–G10)

- Walk-forward using production routing on identical games. Log loss must
  not worsen on either large priced book-season, with Brier agreeing.
  An improvement over 0.008 per season is a leakage tripwire.
- Fit on outcomes only (G9), never the market residual. A maths change
  requires a new model tag. Preserve all native first snapshots.
- After prediction changes: rescore historical rows, rebuild this season,
  build pages, run Model review, the DraftKings backtest (including its
  retired blanket-bet check) and History backtest. Review those results
  before activation. No hand commits of bot-generated data or public pages.
- Keep closed research closed (G8) without a new reason. The useful open
  questions are the native forward record/CLV, genuinely pregame roster and
  newly-out information when snapshots accumulate, early calibration and
  shrinkage measured at the first snapshot. Rookies by draft slot are a
  small future candidate, not included in this simplification.

## Markets and operations (G11–G12)

DraftKings is the production market. New Kalshi prices and exhibitions
are no longer recorded or graded. Existing `kalshi_*` columns and
`data/nba_preseason.csv` remain validated history; research scripts/results
remain. Preseason **box scores** still supply regular-season game-0 rosters.
The Preseason page and its navigation link are removed.

Protect early prices and pregame snapshots. A render failure must not lose
a captured slate. Keep the serialized build and its validate/commit step.
Only eight workflows remain: build, tests, fit-model, backfill, rebuild,
box-history, history-backtest, margin-model. The fifteen closed research
workflows are recoverable from git history.

Every PR runs `python validate_data_files.py` and `python -m pytest tests/ -q`;
page changes also get browser screenshots.
