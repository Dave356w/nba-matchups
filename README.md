# NBA Matchups

Daily NBA leans from a four-factors composite model, compared with the
DraftKings market and graded, published as a static site by GitHub Actions.
It is a sibling of the XWOBA MLB Matchups project and follows the same
approach: a pregame-locked ledger, closing-line grading, and a calibration
page comparing implied probability with actual results.

Site (once Pages is enabled): **<https://dave356w.github.io/nba-matchups/>**

| Page | Shows |
|---|---|
| `index.html` — Today | Composite Δ, the model's P(win), DK moneyline with the no-vig probability, the lean, **model − market** (pp) and the model's EV estimate at the posted price |
| `grades.html` — Ledger | Lean record, excess over no-vig close ± SE, flat units at close, closing-line value (native rows) |
| `market-calibration.html` — Calibration | Market implied vs actual by price rung; model P vs actual with the market on the same games; Brier/log loss model vs market; leans by closing-price band with excess, EV **and its market-correct null** |

Native (pregame-locked) and reconstructed (leave-one-season-out, hindsight)
rows are stored in separate files and shown in separate sections. They are
never pooled into one record.

| File | Role |
|---|---|
| `nba_composite.py` | The model: BBR download, ridge composite weights, decayed game features, logit, backtest. Unchanged from the reference implementation. |
| `build_site.py` | Daily pipeline: grade → score today's slate → write ledger → render pages. |
| `market.py` | ESPN scoreboard + sportsbook odds (DraftKings, else ESPN BET; each price labelled with its book), devig, break-even, EV null, SEs, price ladder. The single home for price arithmetic. |
| `ledger.py` | Ledger schema, pregame-lock ingest, and grading rules. |
| `analysis.py` | Calibration and same-row model-vs-market statistics. |
| `backfill_history.py` | Leave-one-season-out reconstruction of completed seasons, with historical closes. |
| `MODEL.md` / `docs/nba_composite_model_report.pdf` | Model specification and validation. |
| `CLAUDE.md` | Working agreement for research and code changes. |

## First-time setup

1. **Settings → Pages → Source: GitHub Actions.**
2. **Actions → Fit model → Run** fits the weights (2015–19, 2021–26) and
   the logit (2023–26), then commits `model/*.json`. Takes about 25 minutes
   because of Basketball-Reference's rate limit.
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

The 2026-27 regular season starts in late October. The model abstains until
both teams have 10 games, so the first native rows arrive in mid-November.
