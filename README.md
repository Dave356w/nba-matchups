# NBA Matchups

Moneyline forecasts and v11 performance, published by GitHub Actions:
<https://dave356w.github.io/nba-matchups/>.

| Public page | Purpose |
|---|---|
| Today | Defaults to games where both teams have completed 10 games. One primary win probability and ML forecast, with the saved five-hour ML pick and price shown separately. All games and early games remain accessible. Game details retain margin, four factors, injuries and forecast context. Live scores and win chances remain display-only. |
| Results | Every v11 ML prediction, source, execution price, outcome and running return. Defaults to games 10+, with stage, source, season and outcome filters. Tracked betting tests remain internal. |
| Performance | Leads with the combined v11 games-10+ ML ROI, net units, W/L, accuracy and uncertainty, plus market benchmarks. Source and season filters show saved live locks or historical replay separately. Tracked betting tests are excluded. |
| How it works | Frozen ML rule, mixed-record provenance, stage/price definitions, and model explanation. |
| DFS | The existing Yahoo lineup optimizer using Sleeper projections; a separate tool with no ML forecast or tracking record. |

**ML rule, frozen November 10, 2026:** one unit on the team with the higher
v11 win probability (home at exactly 50%), at its saved DraftKings moneyline.
No edge, favorite, confidence or price-band filter. Both teams must have
completed at least 10 games for the main record; other stages stay separate.
Live picks read only valid five-hour locks, including the existing stale-log
exclusions. Missing prices preserve winner W/L without fabricating a return.

**Owner-approved mixed record:** historical reconstructed/rebuilt v11 replays
and saved live v11 picks are pooled across seasons in the headline. Replays
use their historical DraftKings closing prices; live picks use their saved
five-hour prices. This combines different evidence and price timings, is
labeled as mixed, and is not described as a pure forward test. Replays require
a graded outcome, v11 tag and usable DraftKings pair. Native game IDs always
take precedence, including missing/void locks; rebuilt supersedes duplicate
reconstructed games. Source and season views remain available. Running ROI
on Results combines sources within each season and stage in tip order.

Spread H5·3, value-favorite H2·F·5 and totals OU continue unchanged on paper,
with every saved result and future qualifying selection retained internally.
Their picks, rules and records do not appear on public pages. H4 and all earlier retired rules remain
retired in the internal archive. Model maths, fitting, tags, ledger snapshots
and grading are unchanged. Earlier model versions remain separate.

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
and P(over) for paper research. Model families and stages are never pooled;
the owner-approved mixed ML headline pools sources and seasons explicitly.

| File | Role |
|---|---|
| `nba_composite.py` | Model fitting, prior-game features and predictions. |
| `totals.py` | Totals (O/U) model: expected total and P(over). |
| `build_site.py` | Grade, score, preserve snapshots, generate internal reports and publish the approved pages. |
| `public_site.py` | ML forecasts, mixed-record presentation and guide. |
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
