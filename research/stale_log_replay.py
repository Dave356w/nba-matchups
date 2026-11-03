#!/usr/bin/env python3
"""Replay one slate's 5 h locks with stale or complete game logs (HINDSIGHT).

  git archive <lock commit> | tar -x -C <tree>
  python research/stale_log_replay.py <tree> <stale|complete> <out.csv> \
      [slate stale_cutoff [lock times...]]

Runs that commit's own build_site.score_slate at each lock time (UTC, as in
lock_snapshot_utc), with the lock's frozen prices (lock_*, lock_ou_*) in place
of a live odds fetch, and 2026-27 game logs truncated to games before:
  stale    = stale_cutoff (what the builds had: the previous night missing)
  complete = the slate (every game strictly before it)
Game logs, BPM and injury reports are fetched into <tree>/bbr_cache as the
build fetches them; nothing outside <tree> is written. The stale run must
reproduce the frozen lock P, total and P(over) before the complete run means
anything. Information only: never a record, a lock or a page figure
(stale-log locks amendment, CLAUDE.md 2026-11-03).
Defaults: slate 2026-11-02, stale cutoff 2026-11-01, every lock time.
"""
import os, sys
from datetime import datetime, timezone

root, variant, out = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
SLATE, STALE = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else ("2026-11-02", "2026-11-01")
ONLY = set(sys.argv[6:])
os.chdir(root)
sys.path.insert(0, root)
import pandas as pd  # noqa: E402
import build_site as bs, ledger, market, nba_composite as nc  # noqa: E402

CUTOFF = {"stale": STALE, "complete": SLATE}[variant]
messages = []
bs.log = lambda m: (messages.append(m), print(m, flush=True))

_read = nc.read_gamelog
def read_gamelog(tm, y, refresh=False):
    df = _read(tm, y, refresh=False)         # cached page, fetched once
    if df is not None and int(y) == 2027:
        df = df[pd.to_datetime(df["date"]) < pd.Timestamp(CUTOFF)].reset_index(drop=True)
    return df
nc.read_gamelog = read_gamelog
bs.fresh_logs = lambda year, today: nc.load_logs(year)

L = ledger.load(ledger.NATIVE_PATH)
day = L[L["slate_date"].astype(str) == SLATE].set_index("game_id")
market.scoreboard = lambda date: [dict(game_id=g, tip_utc=r.tip_utc, season_type=market.REGULAR_SEASON,
                                       away=r.away, home=r.home, state="pre", completed=False,
                                       away_pts=None, home_pts=None) for g, r in day.iterrows()]

def lock_prices(r):
    x = day.loc[str(r["game_id"])]
    r.update(pre_book=x.lock_book, pre_home_ml=x.lock_home_ml, pre_away_ml=x.lock_away_ml,
             pre_q_home=x.lock_q_home, pre_spread=x.lock_spread,
             pre_home_spread_odds=x.lock_home_spread_odds, pre_away_spread_odds=x.lock_away_spread_odds,
             pre_ou_line=x.lock_ou_line, pre_ou_over=x.lock_ou_over, pre_ou_under=x.lock_ou_under)
bs.add_pregame_price = lock_prices

weights, model = bs.load_model()
rows = []
for t in sorted(day["lock_snapshot_utc"].dropna().unique()):
    if ONLY and t not in ONLY:
        continue
    now = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    print(f"== {variant}: lock time {t}", flush=True)
    got = bs.score_slate(SLATE, weights, model, now=now)
    for r in got:
        if day.loc[str(r["game_id"]), "lock_snapshot_utc"] == t:
            rows.append({**r, "replay_now": t, "refit": next((m for m in reversed(messages) if "refitted" in m or "refit failed" in m), "")})
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {len(rows)} rows to {out}")
