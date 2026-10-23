# NBA Matchups

NBA forecasts and three registered first-snapshot tracking tests, published
by GitHub Actions: <https://dave356w.github.io/nba-matchups/>.

| Public page | Purpose |
|---|---|
| Today | One probability and predicted margin per game, current DraftKings prices, and qualifying tracked selections at their frozen first-snapshot prices. |
| Results | Every tracked win, loss, push and pending pick; filters for test, stage and season. |
| Performance | Spread, value favourite and large probability gap, each with separate games 1–9 and games 10+ records, uncertainty and same-game benchmarks. No combined portfolio return. |
| How it works | Plain-language forecast, rule, timing and results definitions. |

Forecast-only games and legacy lean records do not enter the public tracked
performance. Rebuilt history, retired rules, coefficient tables, price-band
comparisons and exploratory diagnostics remain research outputs.

`public/` is exclusively generated publication output. It contains only the
four approved HTML pages and `.nojekyll`; stale files and directories are
removed, and an allowlist is checked again before the Pages upload.
`diagnostics/site/` is the generated internal workbench, including the full
Ledger, retired archive display, calibration/model pages and plain-text
report. It is not uploaded to Pages. The quotable `data/ledger_report.txt`
continues to be maintained by the build bot. This separation does not make
the public GitHub repository private.

The model is v9 from game 0, with base and NBA injury-report arms. Optional
missing inputs contribute neutral values with distinct tags; missing core
ratings abstain. No legacy carryover or v4 model is loaded in production.
The early tracking segments require a v9 first snapshot. Games 10+ keep
their original registrations. No stage, season or test is pooled.

| File | Role |
|---|---|
| `nba_composite.py` | Model fitting, prior-game features and predictions. |
| `build_site.py` | Grade, score, preserve snapshots, generate internal reports and publish the approved pages. |
| `public_site.py` | Public native-only forecasts, tracking records and guide. |
| `analysis.py` | Registered selections/grades and shared statistics. |
| `ledger.py` | Schema, write-once snapshots, pregame locks and grading. |
| `market.py` | Scoreboard, sportsbook prices and probability arithmetic. |
| `player_availability.py` | NBA injury-report inputs. |
| `report.py` | Full plain-text ledger report, generated outside the public artifact. |
| `research/` and `diagnostics/` | Research scripts, preserved historical models, evidence and results. |
| `CLAUDE.md`, `MODEL.md`, `docs/model_guidelines.md` | Working agreement, model specification and change gates. |

## Operations

Enable Settings → Pages → Source: GitHub Actions. Fit model maintains the
v9 fits; Backfill history rescores prior seasons; Rebuild season maintains
current-season hindsight separately. Build grades at 04:17 ET and refreshes
pregame forecasts hourly from 10:07–22:07 ET. It preserves captured ledger
rows even if rendering fails, subject to data validation.

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

Exactly eight workflows remain: build, tests, fit-model, backfill, rebuild,
box-history, history-backtest and margin-model. Closed research scripts and
results remain available; their workflows can be recovered from git history.
