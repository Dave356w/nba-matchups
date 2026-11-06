# ou3 activation checks (CLAUDE.md O/U model ou3 amendment, registered 2026-11-06T21:10:26Z)

Prepared 2026-11-06 on branch `claude/compassionate-meitner-aunjeq`, for the owner's review
before activation. Nothing here is live: `model/totals.json` stays ou2 until Fit model runs.

## Activation
Merge, then run **Fit model** (its totals step now fits rest_tot). That writes an ou3
`model/totals.json`; the next build forecasts and locks with ou3 tags, writes
`data/nba_ou3_rescore.csv` once per earlier lock, and the public pages switch to the
OU·L·ou3 / OU·E·L·ou3 rows. Until then everything scores, tags and displays as ou2.

## Local checks (session container, main at a249715)
- **Fit (scratch, not committed):** 10,767 games, 2017-19 and 2021-26. T*: rel +0.926,
  form +0.326, b2b +0.618, phase +1.067, **rest_tot +0.748 per day**; σ 19.05 / 18.48
  (ou2 19.04 / 18.49). Report stage refitted on ou3's residual: av_o +0.909, av_d −1.232,
  av_m +0.357 (ou2 +0.920 / −1.232 / +0.342).
- **Rescore dry run** (scratch file, passes `check_ou3_rescore`): 54 locks (62 O/U locks
  minus the 8 voided), all with report terms. ou3 − recorded total: ou2 locks −0.38 ± 0.27
  points (n 23), ou1 locks −0.60 ± 1.15 (n 31).
- **Pages:** rendered from main's ledger in both states (screenshots here). ou2 state:
  wording and rows unchanged. ou3 state: How it works adds "both teams' days of rest"; the
  O/U card reads the ou3 rows. Preview early O/U record: ou3 9–12 (21 graded, ROI −18.2%)
  against ou2 10–13 (23 graded, −16.9%); games 10+ none yet.
- `python validate_data_files.py`: OK. `python -m pytest tests/ -q`: 528 passed.

## Research evidence
`diagnostics/totals_dow/RESULTS.txt` (REST arm against ou2: −0.25 ± 0.71 points², 4 of 7
seasons) and `diagnostics/totals_od/RESULTS.txt`. The margin model, its tests and records,
and the DraftKings / History backtests (margin model) are unchanged by ou3.
