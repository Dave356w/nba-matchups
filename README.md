# NBA Matchups

NBA forecasts and four registered tracking tests, published by GitHub
Actions: <https://dave356w.github.io/nba-matchups/>.

| Public page | Purpose |
|---|---|
| Today | One probability, predicted margin and total per game, current DraftKings prices, each team's season W/L, and each tracked pick once its game locks (the first priced snapshot within 5 hours of tip with current game logs). While a game is in progress, each team's live win chance from the score, the time left and the DraftKings spread at tip (display only). Game details include season four-factor comparisons and timestamped ESPN player injury lists. |
| Results | Every frozen v11 winner prediction, including forecast-only games, with stage, season and outcome filters; a separate view contains every tracked win, loss, push and pending bet. |
| Performance | Every v11 prediction's W/L, accuracy, pending count and Wilson 95% interval, with same-game closing-favourite W/L, log loss and Brier scores. Separate betting records for spread (H5·3), value favourite (H2·F·5), large probability gap (H4) and total (OU). Each season and stage stands alone. No combined portfolio return. |
| How it works | Plain-language forecast, rule, timing and results definitions. |
| DFS lineups | A Yahoo classic-contest lineup optimizer for the day's slate: Yahoo salaries with Sleeper's projections in Yahoo scoring; the best lineups are found in the browser for the main, early or late slate by start time (lock, exclude, re-project, exposure caps). Cash mode maximizes projected points; tournament mode simulates the slate with each player's measured spread and correlations (`diagnostics/dfs_correlation/`), solves every simulated slate exactly and ranks lineups by how often they finish in the top 5%. A separate tool: no forecast, tracked pick or record. |

The public records restarted with the v11 model family on the 2026-10-29
slate and read only the v11-family rows at the 5 h lock (CLAUDE.md,
"Model-family reset" and "Edge floors and O/U test" amendments). Earlier
first-snapshot and 5 h-lock rows stay graded in the internal workbench and
`data/ledger_report.txt`, never pooled. Forecast-only games, rebuilt
history, retired rules, coefficient tables, price-band comparisons and
exploratory diagnostics remain research outputs.

The model-prediction record reads every native v11 five-hour lock from
2026-10-29, with no betting threshold and no market-price requirement.
Later forecast updates cannot change a recorded prediction. Missing locks
are never replaced with a later forecast. Opening games and unknown stages
get their own rows when present. Missing closes remain in the model W/L;
the market comparison uses only matched, graded DraftKings closing pairs.

`public/` is exclusively generated publication output. It contains only the
five approved HTML pages and `.nojekyll`; stale files and directories are
removed, and an allowlist is checked again before the Pages upload.
`diagnostics/site/` is the generated internal workbench, including the full
Ledger, retired archive display, calibration/model pages and plain-text
report. It is kept as a separate `research-workbench` Actions artifact for
seven days and is not uploaded to Pages. The quotable `data/ledger_report.txt`
continues to be maintained by the build bot. This separation does not make
the public GitHub repository private.

The model is v11 from game 0 (`MODEL.md`), with base and NBA injury-report
routes and the same faded rating; P(home) = Φ(predicted margin / σ).
Optional missing inputs contribute neutral values with distinct tags;
missing core ratings abstain. No legacy carryover or v4 model is loaded in
production. A separate totals model (`totals.py`) gives each game's total
and P(over). No stage, season, test or model family is pooled.

| File | Role |
|---|---|
| `nba_composite.py` | Model fitting, prior-game features and predictions. |
| `totals.py` | Totals (O/U) model: expected total and P(over). |
| `build_site.py` | Grade, score, preserve snapshots, generate internal reports and publish the approved pages. |
| `public_site.py` | Public native-only forecasts, tracking records and guide. |
| `matchup_context.py` | Display-only team records and four factors from saved inputs strictly before the slate; no network or model changes. |
| `analysis.py` | Registered selections/grades and shared statistics. |
| `ledger.py` | Schema, write-once snapshots, pregame locks and grading. |
| `market.py` | Scoreboard, sportsbook prices and probability arithmetic. |
| `player_availability.py` | NBA injury-report inputs. |
| `cold_start.py` | Shared decayed-feature helpers (legacy carryover fits are under `research/legacy_*`). |
| `sleeper.py` | Sleeper projection snapshots written by the build; never a model input (the DFS page is their one public use). |
| `dfs.py` | DFS lineups page: the Yahoo salary snapshot (`data/nba_yahoo_dfs.csv`), Sleeper projections in Yahoo scoring, and the in-browser optimizer. |
| `kalshi.py` | Historical Kalshi helpers for research; the build no longer fetches Kalshi. |
| `report.py` | Full plain-text ledger report, generated outside the public artifact. |
| `research/` and `diagnostics/` | Research scripts, preserved historical models, evidence and results. |
| `CLAUDE.md`, `MODEL.md`, `docs/model_guidelines.md` | Working agreement, model specification and change gates. |

## Operations

Enable Settings → Pages → Source: GitHub Actions. Fit model maintains the
v11 and totals fits; Backfill history rescores prior seasons; Rebuild season maintains
current-season hindsight separately. Build grades at 04:17 UTC and is
attempted every 30 minutes through each slate's afternoon and evening.
GitHub drops most scheduled runs, so Lock watch (`watch_slate.py`) also
long-polls each slate: one delivered run sleeps until each game's five-hour
lock window opens and dispatches a build there, retrying a game until it
locks. Build preserves captured ledger rows even if rendering fails, subject
to data validation.

```bash
pip install -r requirements.txt
python build_site.py                  # score, grade and render
python build_site.py --render-only    # render committed data only
python validate_data_files.py
python -m pytest tests/ -q
```

Tests CI also runs desktop and mobile Chromium checks, including labelled
fixtures for pending selections, overlapping tests, wins, losses and pushes.
Screenshots are uploaded as the `public-ui-review` artifact. Fixture rows
are never written to production data or public output. To run those checks:

```bash
pip install playwright==1.56.0
python -m playwright install chromium
python research/check_public_ui.py
```

Exactly nine workflows remain: build, watch, tests, fit-model, backfill,
rebuild, box-history, history-backtest and margin-model. Closed research scripts and
results remain available; their workflows can be recovered from git history.
