#!/usr/bin/env python3
"""Replay every 2026-27 five-hour lock with a tree's model files (HINDSIGHT).

  git worktree add <tree> <commit>      # model/ files of the arm to replay
  python research/suffix_lock_replay.py <tree> <out.csv> [slate ...]

Report-name suffix fix gate (CLAUDE.md, registered 2026-11-13): "the 2026-27
lock reconstruction". Runs the tree's own build_site.score_slate at each lock
time (lock_snapshot_utc), with the lock's frozen prices in place of a live odds
fetch and 2026-27 game logs truncated to games strictly before the slate (as
research/stale_log_replay.py's "complete" variant). The tree's model files
decide the matcher (pav.splits_suffix): the deployed files must reproduce the
frozen lock P (the eight voided stale-log locks excepted) before the fixed
files' P means anything. Game logs, BPM and injury reports are fetched into
<tree>; nothing outside <tree> and <out.csv> is written. Information only:
locks are never rescored (CLAUDE.md), and no value here enters a record.
"""
import os
import sys
from datetime import datetime, timezone

root, out = sys.argv[1], os.path.abspath(sys.argv[2])
ONLY = set(sys.argv[3:])
os.chdir(root)
sys.path.insert(0, root)
import pandas as pd  # noqa: E402
import build_site as bs, ledger, market, nba_composite as nc  # noqa: E402

messages = []
bs.log = lambda m: (messages.append(m), print(m, flush=True))
CUT = {"slate": None}

_read = nc.read_gamelog


def read_gamelog(tm, y, refresh=False):
    df = _read(tm, y, refresh=False)         # cached page, fetched once
    if df is not None and int(y) == 2027 and CUT["slate"]:
        df = df[pd.to_datetime(df["date"]) < pd.Timestamp(CUT["slate"])].reset_index(drop=True)
    return df


nc.read_gamelog = read_gamelog
bs.fresh_logs = lambda year, today: nc.load_logs(year)

L = ledger.load(ledger.NATIVE_PATH)
L = L[L["season"].astype(str).eq("2027") & L["lock_snapshot_utc"].notna()]
DAY = {}


def scoreboard(date):
    return [dict(game_id=g, tip_utc=r.tip_utc, season_type=market.REGULAR_SEASON,
                 away=r.away, home=r.home, state="pre", completed=False,
                 away_pts=None, home_pts=None) for g, r in DAY[date].iterrows()]


def lock_prices(r):
    x = DAY[r["slate_date"]].loc[str(r["game_id"])]
    r.update(pre_book=x.lock_book, pre_home_ml=x.lock_home_ml, pre_away_ml=x.lock_away_ml,
             pre_q_home=x.lock_q_home, pre_spread=x.lock_spread,
             pre_home_spread_odds=x.lock_home_spread_odds,
             pre_away_spread_odds=x.lock_away_spread_odds,
             pre_ou_line=x.lock_ou_line, pre_ou_over=x.lock_ou_over, pre_ou_under=x.lock_ou_under)


market.scoreboard = scoreboard
bs.add_pregame_price = lock_prices

weights, model = bs.load_model()
rows = []
for slate in sorted(L["slate_date"].astype(str).unique()):
    if ONLY and slate not in ONLY:
        continue
    day = L[L["slate_date"].astype(str) == slate].set_index("game_id")
    DAY[slate] = day
    CUT["slate"] = slate
    for t in sorted(day["lock_snapshot_utc"].unique()):
        now = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        print(f"== {slate}: lock time {t}", flush=True)
        try:
            got = bs.score_slate(slate, weights, model, now=now)
        except Exception as e:  # noqa: BLE001 - one slate's failure is reported, not fatal
            print(f"!! {slate} {t}: {e!r}", flush=True)
            continue
        for r in got:
            if day.loc[str(r["game_id"]), "lock_snapshot_utc"] == t:
                rows.append({**r, "replay_now": t,
                             "refit": next((m for m in reversed(messages)
                                            if "refitted" in m or "refit failed" in m), "")})
    pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {len(rows)} rows to {out}")
