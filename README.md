# NBA Matchups

Daily NBA leans from a four-factors composite model, compared with the
DraftKings market and graded, published as a static site by GitHub Actions.
It is a sibling of the XWOBA MLB Matchups project and follows the same
approach: a pregame-locked ledger, closing-line grading, and a calibration
page comparing implied probability with actual results.

Site (once Pages is enabled): **<https://dave356w.github.io/nba-matchups/>**

| Page | Shows |
|---|---|
| `index.html` — Today | Composite Δ, the model's P(win) (games 1–9 tagged *early · carryover*), the moneyline with the no-vig probability, the lean, **model − market** (pp) and the model's EV estimate at the posted price |
| `grades.html` — Ledger | The pre-registered hypotheses (H1–H4: native ROI at the pregame or first-snapshot price beside the hindsight rule, each beside its null), then one section per basis × book × season: a verdict strip (log loss vs the close, lean ROI and value ROI each vs the market-correct null ± 1 SE, native CLV), **ROI of 1u flat bets** on the lean and value side (model %, market %, break-even, win %, ROI, null) by early/later games and model edge; collapsed: ROI by price band (EV beside its null, z), against the spread, game by game |
| `market-calibration.html` — Calibration | Reliability charts (stated vs actual, ±2 SE): the market's no-vig close per book, then the model and the market on the same games per section, with Brier/log loss model vs market; the numbers in collapsed tables |
| `preseason.html` — Preseason | Exhibition games only (`data/nba_preseason.csv`): each scored by the early carryover logit on last season's games alone, priced from Kalshi (YES asks with the taker fee; close = the last 1-minute candle before tip) and graded like native rows; model vs the close (Brier/log loss ± 1 SE, reliability), flat-bet ROI at the pregame price beside its null, game by game. Never pooled with the other pages |

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
| `analysis.py` | Calibration and same-row model-vs-market statistics. |
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
