#!/usr/bin/env python3
"""Fail fast on a broken committed CSV (conflict markers, bad schema, dup ids)."""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

import ledger
import market
import player_availability as pav


def check(path):
    if not os.path.exists(path):
        return []
    errs = []
    text = open(path, encoding="utf-8").read()
    if any(m in text for m in ("<<<<<<<", ">>>>>>>", "\n=======\n")):
        errs.append(f"{path}: merge-conflict marker")
        return errs
    df = pd.read_csv(path, dtype={"game_id": str})
    if list(df.columns) == ledger.LEGACY_COLUMNS:
        print(f"note {path}: pre-book schema (all prices DraftKings); "
              "rewritten with book columns on the next bot write")
    elif list(df.columns) == ledger.PRE_SPREAD_COLUMNS:
        print(f"note {path}: pre-spread schema; spread columns added on the "
              "next bot write")
    elif list(df.columns) == ledger.PRE_SEEN_COLUMNS:
        print(f"note {path}: pre-decision-time schema; first_report_utc, "
              "first_route and seen_* added (blank) on the next bot write")
    elif list(df.columns) == ledger.PRE_KALSHI_COLUMNS:
        print(f"note {path}: pre-Kalshi schema; kalshi_* columns added "
              "(blank) on the next bot write")
    elif list(df.columns) == ledger.PRE_FIRST_COLUMNS:
        print(f"note {path}: pre-first-snapshot schema; first_* columns added "
              "(blank) on the next bot write")
    elif list(df.columns) == ledger.PRE_PREGAME_SPREAD_COLUMNS:
        print(f"note {path}: pre-pregame-spread schema; pre_spread / "
              "first_spread columns added (blank) on the next write")
    elif list(df.columns) == ledger.PRE_CARRY_COLUMNS:
        print(f"note {path}: pre-v7 schema; carry_p_home / first_carry_p_home "
              "added (blank) on the next write")
    elif list(df.columns) == ledger.PRE_MARGIN_COLUMNS:
        print(f"note {path}: pre-v9 schema; margin_home added (blank) on the "
              "next write")
    elif list(df.columns) != ledger.COLUMNS:
        errs.append(f"{path}: columns differ from ledger.COLUMNS")
    if list(df.columns) in (ledger.COLUMNS, ledger.PRE_MARGIN_COLUMNS,
                            ledger.PRE_CARRY_COLUMNS,
                            ledger.PRE_PREGAME_SPREAD_COLUMNS,
                            ledger.PRE_KALSHI_COLUMNS, ledger.PRE_SEEN_COLUMNS,
                            ledger.PRE_FIRST_COLUMNS, ledger.PRE_SPREAD_COLUMNS):
        books = {b for _, b in market.BOOKS} | set(market.EXCHANGES)
        for col, q in (("pre_book", "pre_q_home"), ("close_book", "close_q_home")):
            priced = pd.to_numeric(df[q], errors="coerce").notna()
            if not df.loc[priced, col].isin(books).all():
                errs.append(f"{path}: {col} missing or unknown on a priced row")
    if "kalshi_close_q_home" in df.columns:
        kq = pd.to_numeric(df["kalshi_close_q_home"], errors="coerce")
        if (kq.notna() & ~kq.between(0, 1, inclusive="neither")).any():
            errs.append(f"{path}: kalshi_close_q_home outside (0, 1)")
        if (kq.notna() & pd.to_numeric(df["home_won"], errors="coerce").isna()).any():
            errs.append(f"{path}: Kalshi close on a pending row")
    if "close_spread" in df.columns:
        spread = pd.to_numeric(df["close_spread"], errors="coerce")
        if (spread.notna() & pd.to_numeric(df["close_q_home"],
                                           errors="coerce").isna()).any():
            errs.append(f"{path}: close_spread on a row without a moneyline close")
    # a pregame spread comes only with its book's pregame moneyline
    for line, q in (("pre_spread", "pre_q_home"), ("first_spread", "first_q_home")):
        if line in df.columns:
            if (pd.to_numeric(df[line], errors="coerce").notna()
                    & pd.to_numeric(df[q], errors="coerce").isna()).any():
                errs.append(f"{path}: {line} on a row without {q}")
    if "first_snapshot_utc" in df.columns:
        first = pd.to_datetime(df["first_snapshot_utc"], utc=True, errors="coerce")
        tip = pd.to_datetime(df["tip_utc"], utc=True, errors="coerce")
        snap = pd.to_datetime(df["snapshot_utc"], utc=True, errors="coerce")
        has = df["first_snapshot_utc"].notna()
        if (has & (first.isna() | ~(first < tip))).any():
            errs.append(f"{path}: first snapshot missing a time or not before tip")
        if (has & snap.notna() & (first > snap)).any():
            errs.append(f"{path}: first snapshot later than the latest snapshot")
        books = {b for _, b in market.BOOKS} | set(market.EXCHANGES)
        if not df.loc[has, "first_book"].isin(books).all():
            errs.append(f"{path}: first_book missing or unknown on a first snapshot")
        if "seen_utc" in df.columns:
            seen = pd.to_datetime(df["seen_utc"], utc=True, errors="coerce")
            hs = df["seen_utc"].notna()
            if (hs & (seen.isna() | ~(seen < tip))).any():
                errs.append(f"{path}: seen_utc missing a time or not before tip")
            if (hs & has & (seen > first)).any():
                errs.append(f"{path}: first snapshot earlier than the row's first sight")
    if df["game_id"].duplicated().any():
        errs.append(f"{path}: duplicate game_id")
    won = pd.to_numeric(df["home_won"], errors="coerce").dropna()
    if not won.isin([0, 1]).all():
        errs.append(f"{path}: home_won outside {{0,1}}")
    p = pd.to_numeric(df["p_home"], errors="coerce").dropna()
    if not p.between(0, 1).all():
        errs.append(f"{path}: p_home outside [0,1]")
    for c in ledger.CARRY_COLUMNS + ledger.FIRST_CARRY_COLUMNS:
        if c in df.columns:
            cp = pd.to_numeric(df[c], errors="coerce")
            if not cp.dropna().between(0, 1).all():
                errs.append(f"{path}: {c} outside [0,1]")
            gp = np.fmin(pd.to_numeric(df["gp_home"], errors="coerce"),
                         pd.to_numeric(df["gp_away"], errors="coerce"))
            if (cp.notna() & ~(gp < 10)).any():
                errs.append(f"{path}: {c} on a game outside games 1-9")
    if "margin_home" in df.columns:
        # a model margin only beside its P, on the same side of 50%
        m = pd.to_numeric(df["margin_home"], errors="coerce")
        pp = pd.to_numeric(df["p_home"], errors="coerce")
        if (m.notna() & pp.isna()).any():
            errs.append(f"{path}: margin_home on a row without p_home")
        if (m.notna() & ~m.between(-60, 60)).any():
            errs.append(f"{path}: margin_home outside +-60")
        if (m.notna() & (m != 0) & ((m > 0) != (pp > 0.5))).any():
            errs.append(f"{path}: margin_home and p_home disagree on the favourite")
    return errs


def check_injuries(path=ledger.INJURY_PATH):
    if not os.path.exists(path):
        return []
    text = open(path, encoding="utf-8").read()
    if any(m in text for m in ("<<<<<<<", ">>>>>>>", "\n=======\n")):
        return [f"{path}: merge-conflict marker"]
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ledger.INJURY_COLUMNS:
        return [f"{path}: columns differ from ledger.INJURY_COLUMNS"]
    errs = []
    snap = pd.to_datetime(df["snapshot_utc"], utc=True, errors="coerce")
    tip = pd.to_datetime(df["tip_utc"], utc=True, errors="coerce")
    if snap.isna().any() or tip.isna().any():
        errs.append(f"{path}: unparseable snapshot_utc or tip_utc")
    elif (snap >= tip).any():
        errs.append(f"{path}: injury snapshot at or after tip")
    if (df.groupby("game_id")["snapshot_utc"].nunique() > 1).any():
        errs.append(f"{path}: a game mixes injury snapshots")
    return errs


def check_sleeper(path=ledger.SLEEPER_PATH):
    if not os.path.exists(path):
        return []
    text = open(path, encoding="utf-8").read()
    if any(m in text for m in ("<<<<<<<", ">>>>>>>", "\n=======\n")):
        return [f"{path}: merge-conflict marker"]
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ledger.SLEEPER_COLUMNS:
        return [f"{path}: columns differ from ledger.SLEEPER_COLUMNS"]
    errs = []
    snap = pd.to_datetime(df["snapshot_utc"], utc=True, errors="coerce")
    tip = pd.to_datetime(df["tip_utc"], utc=True, errors="coerce")
    if snap.isna().any() or tip.isna().any():
        errs.append(f"{path}: unparseable snapshot_utc or tip_utc")
    elif (snap >= tip).any():
        errs.append(f"{path}: Sleeper snapshot at or after tip")
    if not df["kind"].isin(ledger.SLEEPER_KINDS).all():
        errs.append(f"{path}: kind other than {ledger.SLEEPER_KINDS}")
    if (df.groupby(["game_id", "kind"])["snapshot_utc"].nunique() > 1).any():
        errs.append(f"{path}: a game's snapshot mixes snapshot times")
    if df.duplicated(["game_id", "kind", "team", "sleeper_id"]).any():
        errs.append(f"{path}: a player twice in one snapshot")
    if not snap.isna().any():
        t = df.assign(_s=snap).groupby(["game_id", "kind"])["_s"].first().unstack()
        if {"first", "pre"} <= set(t.columns) and (t["first"] > t["pre"]).any():
            errs.append(f"{path}: a first snapshot later than the latest one")
    return errs


def check_box(path):
    """data/nba_box_<y>.csv: past seasons (Box-score history) and this
    season's (the daily build)."""
    text = open(path, encoding="utf-8").read()
    if any(m in text for m in ("<<<<<<<", ">>>>>>>", "\n=======\n")):
        return [f"{path}: merge-conflict marker"]
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != pav.BOX_COLUMNS:
        return [f"{path}: columns differ from player_availability.BOX_COLUMNS"]
    errs = []
    if df["player_id"].str.strip().isin(["", "None", "nan", "NaN"]).any():
        errs.append(f"{path}: a row without a player id")
    if df.duplicated(["game_id", "player_id"]).any():
        errs.append(f"{path}: a player twice in one game")
    y = int(re.search(r"nba_box_(\d{4})\.csv$", path).group(1))
    d = pd.to_datetime(df["date"], errors="coerce")
    if d.isna().any() or (d < pd.Timestamp(f"{y - 1}-10-01")).any() \
            or (d > pd.Timestamp(f"{y}-08-31")).any():
        errs.append(f"{path}: a date outside the {y - 1}-{y % 100:02d} season")
    return errs


def check_pre_box(path):
    """data/nba_box_pre_<y>.csv: preseason box scores (model v8's game-0
    rosters), written once per season."""
    text = open(path, encoding="utf-8").read()
    if any(m in text for m in ("<<<<<<<", ">>>>>>>", "\n=======\n")):
        return [f"{path}: merge-conflict marker"]
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != pav.PRE_COLUMNS:
        return [f"{path}: columns differ from player_availability.PRE_COLUMNS"]
    errs = []
    if df["player_id"].str.strip().isin(["", "None", "nan", "NaN"]).any():
        errs.append(f"{path}: a row without a player id")
    if df.duplicated(["game_id", "player_id"]).any():
        errs.append(f"{path}: a player twice in one game")
    y = int(re.search(r"nba_box_pre_(\d{4})\.csv$", path).group(1))
    d = pd.to_datetime(df["date"], errors="coerce")
    if d.isna().any() or (d < pd.Timestamp(f"{y - 1}-09-01")).any() \
            or (d > pd.Timestamp(f"{y - 1}-12-31")).any():
        errs.append(f"{path}: a date outside the {y - 1}-{y % 100:02d} preseason")
    return errs


def main():
    errs = check(ledger.NATIVE_PATH) + check(ledger.RECON_PATH) + check_injuries()
    errs += check_sleeper()
    for path in sorted(glob.glob(os.path.join("data", "nba_box_[0-9][0-9][0-9][0-9].csv"))):
        errs += check_box(path)
    for path in sorted(glob.glob(os.path.join("data", "nba_box_pre_[0-9][0-9][0-9][0-9].csv"))):
        errs += check_pre_box(path)
    errs += check(ledger.PRESEASON_PATH)
    errs += check(ledger.REBUILT_PATH)
    if os.path.exists(ledger.REBUILT_PATH):
        rb = pd.read_csv(ledger.REBUILT_PATH, dtype={"game_id": str})
        if not rb["basis"].eq("rebuilt").all():
            errs.append(f"{ledger.REBUILT_PATH}: basis other than rebuilt")
        if rb["first_snapshot_utc"].notna().any() or rb["pre_q_home"].notna().any():
            errs.append(f"{ledger.REBUILT_PATH}: a pregame price on a rebuilt row")
        if pd.to_numeric(rb["home_won"], errors="coerce").isna().any():
            errs.append(f"{ledger.REBUILT_PATH}: an ungraded rebuilt row")
    for path in (ledger.NATIVE_PATH, ledger.RECON_PATH, ledger.REBUILT_PATH):
        # exchanges: preseason only
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={"game_id": str})
            cols = [c for c in ("pre_book", "close_book", "first_book") if c in df]
            if df[cols].isin(market.EXCHANGES).any().any():
                errs.append(f"{path}: an exchange price outside the preseason ledger")
    if os.path.exists(ledger.PRESEASON_PATH):
        b = pd.read_csv(ledger.PRESEASON_PATH, dtype={"game_id": str})["basis"]
        if not b.eq("preseason").all():
            errs.append(f"{ledger.PRESEASON_PATH}: basis other than preseason")
    if os.path.exists(ledger.PRESEASON_PATH) and os.path.exists(ledger.NATIVE_PATH):
        ids = set(pd.read_csv(ledger.PRESEASON_PATH, dtype={"game_id": str})["game_id"])
        if ids & set(pd.read_csv(ledger.NATIVE_PATH, dtype={"game_id": str})["game_id"]):
            errs.append("a preseason game_id is also in the native ledger")
    for e in errs:
        print("ERROR", e)
    print("data files OK" if not errs else f"{len(errs)} problem(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
