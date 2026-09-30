# Claude — working agreement for NBA Matchups

This is the sibling of `Dave356w/Dave356w` (XWOBA MLB Matchups), and the same
working agreement applies: be a rigorous, constructive collaborator; inspect
real code and data; recognize measured progress; describe uncertainty in
proportion to the evidence; and turn each observation into the next test. The
owner directs the product.

## Sources of truth

- `MODEL.md` + `nba_composite.py`: the model. `build_site.py`: the daily
  pipeline and pages. `market.py`: all price arithmetic, in one place.
  `ledger.py`: ledger invariants. `analysis.py`: calibration statistics.
- `data/nba_ledger.csv`: **native** rows, pregame-locked (snapshot < tip).
- `data/nba_reconstructed.csv`: **reconstructed** rows (leave-one-season-out,
  closing price). This is hindsight and is not forward evidence.

## Evidence rules

1. Native and reconstructed rows answer different questions. Report them
   with separate labels and counts; never pool them silently.
2. Compare the model with the market on the **same rows** using proper
   scores (Brier, log loss) and calibration. Accuracy or ROI alone does not
   show calibration. Each row's close comes from one book (`close_book`:
   DraftKings, or ESPN BET before late Nov 2025); report books separately.
   ESPN provider 59 (live odds) is in-game and never read.
3. No-vig q and posted break-even are different thresholds. An EV figure
   (win% − break-even) is centred on `market.ev_null` (q − break-even, about
   minus the hold), **not zero**. Print the null beside every EV figure.
4. Price bands are descriptive monitoring dimensions, not filters. An
   in-sample winning band is a hypothesis to test forward.
5. An interval crossing zero means the sample has not resolved the effect;
   it does not establish zero effect.

## Engineering rules

- **No lookahead.** Features use games strictly before the slate date. A
  pregame row is written or refreshed only before tip and is frozen after.
  A pending game never receives a closing line. `tests/test_ledger.py` and
  `tests/test_scoring.py` enforce these rules.
- Grading writes only result and open/close columns (`ledger.GRADE_COLUMNS`).
- A change to prediction math requires a new `MODEL_TAG`. A display change
  does not.
- Keep `site-build` serialized (`cancel-in-progress: false`). Fit model and
  Backfill history each have their own queue (a pending run in `site-build`
  is cancelled by the next hourly build) and retry their push. Tests gate
  PRs and are deliberately not wired into the daily build.
- Do not hand-commit bot-generated `data/` changes or `public/`.

Before a PR, run:

    python validate_data_files.py
    python -m pytest tests/ -q

## Open research questions (from the report §7–8)

- Market benchmark: Brier/log loss vs the DK close, and native CLV.
  Production v4, walk-forward (MODEL.md §6.1): trails the close by
  +0.0163 ± 0.0106 (2024-25 ESPN BET) and +0.0216 ± 0.0114 (2025-26 DK) log
  loss, games 10+; games 1–9 and the 107 early 2025-26 ESPN BET games are
  unresolved. No demonstrated edge; native rows are the test.
- Player availability (injury reports, minutes shares) is the main missing
  information. `research/availability.py` (workflow "Availability ceiling")
  measures its HINDSIGHT ceiling: who actually played, relative to each
  team's rating window, added to walk-forward logits and compared with the
  close on the same games (v1: on/off-weighted; v2: last-season BBR BPM,
  name-matched, plus players new to the team). `research/pregame_availability.py`
  (workflow "Pregame availability") is the pregame version: the NBA's
  archived injury report, last edition at least --lead-minutes before tip
  (Out/Doubtful = out; or status play rates fitted on training seasons),
  same v2 values, and how often each status actually played. **Model v3/v4**
  (`player_availability.py`, `model/logit_avail.json`) ships its Out/Doubtful
  arm for games 10+: the build reads the latest NBA report and this season's
  box scores (`data/nba_box_<season>.csv`, extended daily). Rows the report
  does not cover (`player_availability.covers`: matchup on it, no team NOT
  YET SUBMITTED; an empty or unparsed report covers nothing) fall back to
  the base logit (v4 tag without "avail"), in the build, the fit and the
  backfill alike. Historically 4,922 of 4,923 games were covered.
  The daily build snapshots each game's ESPN injury list to
  `data/nba_injuries.csv` under the pregame lock (replaced only before tip,
  frozen after; "NONE" rows mark teams with nobody listed). The model does
  not read it yet.
- Early-season cold start: v2 (`cold_start.py`) ships the probe's
  carryover arm (ρ = 0.25) for games 1–9, labelled "early · carryover" on
  the card; game 0 still abstains. `research/cold_start_probe.py` remains
  the backtest (ρ, preseason κ, both) against the close on the same games.
  Watch the native early-season rows against it.
- ROI is the product goal: the ledger page grades 1u flat bets on the lean
  and on the value side (model P > no-vig q), with model WP, market WP,
  actual, and the ROI null beside every ROI. Graded rows now also record the
  closing spread (`close_spread`, home line, plus both spread prices, same
  book as the moneyline close), and the ledger page grades ATS the same way
  (lean and value side, break-even and ROI null beside each ROI).
  It also grades both sides by the picked price band (EV beside its null,
  z vs the ROI null); native rows are banded at the pregame price and the
  v3 rows are shown alone. The hindsight rows leave one band hypothesis to
  test forward: DraftKings leans at −249 to −130.
- More game-log seasons to settle the half-life and to test travel/altitude.
- Walk-forward: `research/walk_forward.py` (workflow "Walk-forward
  backtest") re-scores reconstructed seasons with every fit on earlier
  seasons only, beside the leave-one-season-out rows and the close on the
  same games, plus a possession-based pace arm (games 10+). Its `prod` arm
  is the shipped v4 routing (`backfill_history.reconstruct_season(...,
  walk_forward=True)`: early carryover, base v4, availability v4 on covered
  games with `--avail`), split by route. Walk-forward costs +0.0010 ±
  0.0030 vs leave-one-season-out, so the reconstructed rows are not
  materially flattered. Pace adds nothing. Research only.
- One feature builder: `nba_composite.logit_inputs` (delta, b2b_net, phase
  from opening night, d_phase) serves build_games, the CLI scorer and
  `build_site.score_game`; research scripts reuse the `phase` column.
- Calibration shape: in-sample, the outcome's slope on the model logit is
  ~1 overall but ~1.4–1.9 in March–April (too flat) and ~0.8 before March.
  `research/calibration_shape.py` (workflow "Calibration shape") tests
  delta·|delta|, delta·phase and delta·[Mar+] arms walk-forward against the
  close on the same games. Walk-forward (logits 2016–2024/25): delta·phase
  beats base by −0.0036 ± 0.0036 (2024-25 ESPN BET) and −0.0075 ± 0.0033
  (2025-26 DK) log loss (reproduced exactly with the opening-night clock,
  2026-09-30); delta·|delta| adds nothing. The Pregame availability
  workflow also fits base+phase and od+phase to test it on top of v3.
  **Model v4** ships delta·phase in both logits (MODEL.md; tags
  `..._phase_v4` without a report, `..._phase_avail_v4` with one). The
  reconstructed rows are re-scored with v4 by `backfill_history.py
  --rescore` (workflow "Backfill history", prices and results kept).
