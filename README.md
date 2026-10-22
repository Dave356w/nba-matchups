# NBA Matchups

Daily NBA leans from a four-factors composite model, compared with the
DraftKings market and graded, published as a static site by GitHub Actions.
It is a sibling of the XWOBA MLB Matchups project and follows the same
approach: a pregame-locked ledger, closing-line grading, and a calibration
page comparing implied probability with actual results.

Site (once Pages is enabled): **<https://dave356w.github.io/nba-matchups/>**

| Page | Shows |
|---|---|
| `index.html` — Picks | The NFL site's layout (shared tokens, header, tiles, cards). The season's lean record (record, win rate vs break-even, units, ROI ± 1 SE beside its null and the market favourite; native rows only, one closing book × season), then one card per game: both teams' model P(win) and moneyline over spread (one sportsbook, DraftKings, named on the card), the lean and its price (its edge and EV beside its null, both counting only 40% of the model's disagreement with the price: `analysis.SHRINK_W`), the pre-registered spread pick when H5/H6 bets, *early · carryover* / *injury report* / back-to-back badges; All / Upcoming / Live / Final filter and live scores on reload. Tap a card for model vs the sportsbook's no-vig price, EV, composite Δ, the H5/H6 pick at its first-snapshot price (or why the rule passes) and each team's ESPN injury list as recorded at the last pregame snapshot (status and reason; display only, the model reads the NBA's official report) |
| `results.html` — Results | The same record tiles, the H6 / H5 spread-pick tiles (cover vs break-even, ROI beside its null and the spread favourite), then every graded pick by slate (lean, model, market, final, W/L, units, spread pick) |
| `performance.html` — Performance | The record tiles, running units for the lean, the lean beside the market favourite and the home side on the same games and prices, a by-week table (ROI ± 1 SE, null), the market on its own (DraftKings' closing prices on every graded game this season: favourites straight up, at the moneyline and against the spread, home sides against the spread, the closing line against the final margin, and the same by the favourite's price, by spread size and for home and road favourites), the model's and the market's forecasts on the same games (log loss and Brier, model − market ± 1 SE; picked-the-winner rates; calibration slope for each; the lean won vs stated and the favourite won vs implied; closing-line value; a model-vs-market reliability chart; forecasts by week), and the spread-pick tiles with a running-units chart for each rule (H6, H5; never pooled) |
| **Research tab** | The pages below, unchanged in content, under one tab with their own sub-nav |
| `grades.html` — Research · Ledger | The reading key and glossary (collapsed), the pre-registered hypotheses (H1–H4 and H2·F on the moneyline, H5 and H6 against the spread: native ROI at the pregame or first-snapshot price beside the hindsight rule, each beside its null), with H5/H6 also split by basis × book × season, then the forward record (native, open) and the rebuilt history (reconstructed, one collapsed fold per book × season whose summary line shows the lean's W–L, ROI, null and favourite ROI); each section has a verdict strip (log loss vs the close, lean ROI and value ROI each vs the market-correct null ± 1 SE, native CLV), **ROI of 1u flat bets** on the lean and value side (model %, market %, break-even, win %, ROI, null) by early/later games and model edge; collapsed: ROI by price band (EV beside its null, z), against the spread, game by game |
| `market-calibration.html` — Research · Calibration | Reliability charts (stated vs actual, ±2 SE): the market's no-vig close per book, then the model and the market on the same games per section, with Brier/log loss model vs market; the numbers in collapsed tables |
| `model.html` — Research · Model | Routing, fitted coefficients of each logit, and flat 1u ROI by basis × book × season (lean and value, at the close) with no-vig q, excess, market null, the same-row market favourite and log loss vs the close |
| `ledger_report.txt` — linked from every footer | The plain-text readout of every block above (`report.py`), committed as `data/ledger_report.txt`: quote live numbers from it |
| `preseason.html` — Research · Preseason | Exhibition games only (`data/nba_preseason.csv`): each scored by the early carryover logit on last season's games alone, priced from Kalshi (YES asks with the taker fee; close = the last 1-minute candle before tip) and graded like native rows; model vs the close (Brier/log loss ± 1 SE, reliability), flat-bet ROI at the pregame price beside its null, game by game. Never pooled with the other pages |

Native (pregame-locked) and reconstructed (leave-one-season-out, hindsight)
rows are stored in separate files and shown in separate sections. They are
never pooled into one record.

| File | Role |
|---|---|
| `nba_composite.py` | The core model: BBR download, ridge composite weights, decayed game features, `logit_inputs` (shared by every scoring path), logit, backtest. |
| `cold_start.py` / `player_availability.py` | Games 1–9 carryover; NBA injury-report availability terms and report coverage (games 10+). |
| `build_site.py` | Daily pipeline: grade → score today's slate → write ledger → render pages. |
| `market.py` | ESPN scoreboard + sportsbook odds (DraftKings, else ESPN BET; each price labelled with its book), devig, break-even, EV null, SEs, price ladder. The single home for price arithmetic. |
| `kalshi.py` | Kalshi exchange prices for preseason rows only: live asks per build, the close from the last pregame 1-minute candle. |
| `ledger.py` | Ledger schema, pregame-lock ingest, and grading rules. |
| `analysis.py` | Calibration and same-row model-vs-market statistics, including the same-row market-favourite baseline (`fav_roi`). |
| `report.py` | `data/ledger_report.txt`: the plain-text ledger readout (the siblings' format), built only from `analysis`, stamped from the data so an unchanged ledger makes no commit. |
| `backfill_history.py` | Reconstruction of completed seasons with the production routing (leave-one-season-out, or walk-forward), with historical closes. |
| `research/` | Walk-forward, calibration-shape, availability and cold-start backtests (workflows; write nothing to the repo). |
| `MODEL.md` / `docs/nba_composite_model_report.pdf` | Model specification and validation. |
| `CLAUDE.md` | Working agreement for research and code changes. |

## First-time setup

1. **Settings → Pages → Source: GitHub Actions.**
2. **Actions → Fit model → Run** fits the weights (2015–19, 2021–26), the
   v4 base logit (2016–19, 2021–26), the early-season logit and the
   availability logit (2023–26), then commits `model/*.json`. Takes about
   25 minutes because of Basketball-Reference's rate limit.
3. **Actions → Backfill history → Run** (optional, about 1–2 hours) writes
   the reconstructed 2024-25 and 2025-26 seasons against the close (ESPN BET
   before late November 2025, DraftKings after; `close_book`), so
   the calibration page has data before the season starts.
4. The **Build** workflow then runs on its schedule: 04:17 ET grading and
   hourly pregame refreshes from 10:07 to 22:07 ET.

```bash
pip install -r requirements.txt
python build_site.py                  # today's ET slate
python build_site.py --render-only    # pages from committed data only
python nba_composite.py backtest --years 2023-2026 --train-years 2015-2019 2021-2022

pip install pytest
python validate_data_files.py && python -m pytest tests/ -q
```

The 2026-27 regular season starts in late October. The model abstains only
until both teams have played once (games 1–9 use the carryover model), so
native rows start in the first week. See `MODEL.md` for the routing, the
committed coefficients and the evidence.
