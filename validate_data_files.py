#!/usr/bin/env python3
"""Fail fast on a broken committed CSV (conflict markers, bad schema, dup ids)."""
import os
import sys

import pandas as pd

import ledger
import market


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
    elif list(df.columns) != ledger.COLUMNS:
        errs.append(f"{path}: columns differ from ledger.COLUMNS")
    if list(df.columns) in (ledger.COLUMNS, ledger.PRE_KALSHI_COLUMNS,
                            ledger.PRE_SEEN_COLUMNS,
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


def main():
    errs = check(ledger.NATIVE_PATH) + check(ledger.RECON_PATH) + check_injuries()
    errs += check(ledger.PRESEASON_PATH)
    for path in (ledger.NATIVE_PATH, ledger.RECON_PATH):     # exchanges: preseason only
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
