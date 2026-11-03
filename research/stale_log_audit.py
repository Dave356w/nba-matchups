#!/usr/bin/env python3
"""Which first snapshots and 5 h locks were taken with stale game logs.

  python research/stale_log_audit.py [--since 2026-10-19] [--csv out.csv]

The ledger's gp_* and *_b2b columns are refreshed with every pregame
snapshot, so they are read from the ledger commit that first wrote each
game's first snapshot and lock (git history of data/nba_ledger.csv). They are
compared with the saved ESPN box scores (data/nba_box_<season>.csv): the
team's games strictly before the slate, and whether it played the day before.
A side is stale when the snapshot's games played fall short of the box count.

Outcome-blind: no result, price or probability is read. The stale-log locks
amendment (CLAUDE.md, 2026-11-03) voids exactly the locks this lists.
Slates are audited only while the box file covers the day before them.
"""
import argparse
import io
import subprocess

import pandas as pd

COLS = ["game_id", "slate_date", "season", "home", "away", "gp_home", "gp_away",
        "home_b2b", "away_b2b", "first_snapshot_utc", "lock_snapshot_utc"]


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def snapshots(since):
    """{(which, game_id): (commit, row)} at the commit that first wrote
    first_snapshot_utc / lock_snapshot_utc for the game."""
    commits = git("log", "--reverse", "--format=%h", f"--since={since}", "--",
                  "data/nba_ledger.csv").split()
    out = {}
    for c in commits:
        L = pd.read_csv(io.StringIO(git("show", f"{c}:data/nba_ledger.csv")),
                        dtype={"game_id": str}, low_memory=False)
        L = L[L["slate_date"].astype(str) >= since]
        for _, r in L[[x for x in COLS if x in L.columns]].iterrows():
            for which, col in (("first", "first_snapshot_utc"), ("lock", "lock_snapshot_utc")):
                if (which, r.game_id) not in out and pd.notna(r.get(col)):
                    out[(which, r.game_id)] = (c, r)
    return out


def audit(since="2026-10-19"):
    snaps = snapshots(since)
    boxes = {}
    rows = []
    for (which, gid), (c, r) in snaps.items():
        y = int(r.season)
        if y not in boxes:
            b = pd.read_csv(f"data/nba_box_{y}.csv", dtype={"game_id": str})
            b = b[["game_id", "date", "team"]].drop_duplicates()
            b["date"] = pd.to_datetime(b["date"])
            boxes[y] = b
        b = boxes[y]
        slate = pd.Timestamp(r.slate_date)
        if slate > b["date"].max() + pd.Timedelta(days=1):
            continue                    # box truth not saved yet
        for side in ("home", "away"):
            g = b[(b["team"] == r[side]) & (b["date"] < slate)]
            rows.append(dict(which=which, commit=c, game_id=gid, slate=r.slate_date,
                             side=side, team=r[side], gp=r[f"gp_{side}"], gp_box=len(g),
                             b2b=r[f"{side}_b2b"],
                             b2b_box=bool((g["date"] == slate - pd.Timedelta(days=1)).any())))
    a = pd.DataFrame(rows)
    a["short"] = a["gp_box"] - pd.to_numeric(a["gp"], errors="coerce")
    a["b2b_wrong"] = (pd.to_numeric(a["b2b"], errors="coerce").ne(a["b2b_box"].astype(float))
                      & a["b2b"].notna())
    return a


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2026-10-19")
    ap.add_argument("--csv")
    o = ap.parse_args(argv)
    a = audit(o.since)
    if o.csv:
        a.to_csv(o.csv, index=False)
    for which in ("first", "lock"):
        x = a[a["which"] == which]
        print(f"{which}: {x.game_id.nunique()} games audited; sides short of the box "
              f"{int((x.short > 0).sum())}, ahead {int((x.short < 0).sum())}, "
              f"b2b wrong {int(x.b2b_wrong.sum())}")
        t = x[x.b2b_box].groupby("slate").agg(b2b_sides=("team", "size"),
                                              stale=("short", lambda s: int((s > 0).sum())))
        print(t.T.to_string())
    bad = a[(a["which"] == "lock") & (a["short"] > 0)]
    print("\nstale-log locks (a side short of its box games at the lock):")
    for gid, x in bad.groupby("game_id"):
        r = x.iloc[0]
        print(f"  {r.slate}  {gid}  stale: {' '.join(x.team)}  (lock commit {r.commit})")


if __name__ == "__main__":
    main()
