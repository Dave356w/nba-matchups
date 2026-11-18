# Cell ROI forward log — registration

Owner request, registered 2026-11-18T18:40:00Z. Committed on its own before
any code: the native ledger at `22118dc` has 222 rows, the last lock at
2026-11-17T23:02:30Z, and no lock on the 2026-11-18 slate (12 upcoming games,
none locked).

## Purpose

The owner's original intent for the model-WP × market-WP matrix was to track
the realized result and 1-unit ROI of the model's pick in each cell. The
matrix and the cards' pooled-cell line do this from saved data at render
time. This log adds a frozen, forward record: at each lock, the game's cell
and that cell's trailing record as the card showed it, written once, so the
cell's later results can be read against what was known before tip.

Research and monitoring only. It never selects, filters or grades a pick,
never enters Realized WP, the ML record, a rule or a model, and is never
pooled with replay.

## Capture, fixed now

- **Games.** Native rows with a valid v11-family five-hour lock: lock tag
  ending `_v11` (neutral variants included), DraftKings lock book, both lock
  moneylines, not a voided lock, tip after this registration. One row per
  game, write-once. Openers (game 0) are recorded under their own stage.
- **When.** The first build after the lock snapshot and before tip. A game
  not captured before tip has no row; nothing is backfilled.
- **Pick and cell.** The model's lock pick (home at exactly 50%), its lock P
  and no-vig lock q for that team, the lock moneyline for that team, and the
  cell from the existing bands (model 50/60/70/80/100%, market
  0/50/60/70/80/100%, lower inclusive).
- **Trailing cell record.** The card's pooled-cell population for that cell
  and stage at the lock: the latest 200 results on strictly earlier slates,
  previous-season eligible v11 replay at its DraftKings close plus this
  season's graded native locks (`wp_calibration.RollingCalibration`, as the
  cards use it), read with the lock P and q. Saved: games, wins, priced
  results, net units, ROI and its 1 SE at each result's own price, and the
  previous/current-season split.
- **Not saved.** Results. Grades are read at report time from the native
  ledger (`home_won`); a push or void leaves the game out.
- File: `data/cell_roi_forward_<season>.csv`, written by `cell_forward.py`
  inside the build's existing capture step; a failure is logged and never
  blocks grading, snapshots, the commit or publication. Validation checks
  uniqueness, timing (lock ≤ capture < tip), bands and internal arithmetic.

## Reading, fixed now

Primary, once, at the end of the 2026-27 regular season: on graded games
1+ (early and established pooled; openers separate), the OLS slope of the
game's 1-unit return on the model pick at the lock price against the
trailing cell ROI saved at its lock, with a game-level 95% interval. Trailing
cell ROI "carries over" only if that interval lies above zero; otherwise the
reading is that a cell's past ROI did not predict its next results.

Descriptive, at any time: forward 1-unit ROI ± 1 SE overall and by cell. No
cell figure is evidence for a rule. The matrix window backtest already found
no carry-over in replay (diagnostics/matrix_window/RESULTS.txt); this log is
its forward test, not a search for profitable cells.

## Clarification before implementation (2026-11-18, before any capture)

The saved trailing "1 SE" is the standard error of the mean 1-unit return,
left blank for fewer than two priced results and when every result is a win
or every one a loss (the results' spread would then understate the
uncertainty). The pages' matrix and cards follow the same rule.
