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
| `grades.html` — Ledger | Lean record, excess over no-vig close ± SE, closing-line value (native rows), and **ROI of 1u flat bets** on the lean and on the value side (model P > no-vig P): model WP, market WP, break-even, actual, units, ROI ± SE **beside its market-correct null**, by early/later games and model edge; per-pick P/L |
| `market-calibration.html` — Calibration | Market implied vs actual by price rung; model P vs actual with the market on the same games; Brier/log loss model vs market; leans by closing-price band with excess, EV **and its market-correct null** |

Native (pregame-locked) and reconstructed (leave-one-season-out, hindsight)
rows are stored in separate files and shown in separate sections. They are
never pooled into one record.

| File | Role |
|---|---|
| `nba_composite.py` | The core model: BBR download, ridge composite weights, decayed game features, `logit_inputs` (shared by every scoring path), logit, backtest. |
| `cold_start.py` / `player_availability.py` | Games 1–9 carryover; NBA injury-report availability terms and report coverage (games 10+). |
| `build_site.py` | Daily pipeline: grade → score today's slate → write ledger → render pages. |
| `market.py` | ESPN scoreboard + sportsbook odds (DraftKings, else ESPN BET; each price labelled with its book), devig, break-even, EV null, SEs, price ladder. The single home for price arithmetic. |
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
