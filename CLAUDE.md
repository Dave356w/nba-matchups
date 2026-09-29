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
- Player availability (injury reports, minutes shares) is the main missing
  information.
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
- More game-log seasons to settle the half-life and to test travel/altitude.
