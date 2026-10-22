"""Game ledger: schema, pregame-locked ingest, and grading.

Two files, never blended:

  data/nba_ledger.csv        NATIVE rows. Written by the daily build from a
                             pregame snapshot (snapshot_utc < tip_utc). These
                             are the only forward observations.
  data/nba_reconstructed.csv RECONSTRUCTED rows. Written by backfill_history.py
                             for completed seasons, scored leave-one-season-out
                             and priced at the historical close. Useful for
                             calibration and debugging; not forward evidence.

Invariants (tests/test_ledger.py pins each one):
  * A row is accepted only while now < tip. After tip it can never be created.
  * Before tip a newer pregame snapshot replaces an older one (injury news and
    line moves arrive during the day); after tip the pregame fields freeze.
  * The FIRST snapshot with both a model probability and a priced pair is
    also kept (`first_*`, FIRST_COLUMNS): written once, before tip, and never
    replaced by a later snapshot or touched by grading. It is the price and
    probability an early bettor had (the open-price hypothesis, H4); the
    refreshed columns end near the close. The first snapshot also records
    the injury-report edition the build read then (`first_report_utc`) and
    the route that scored it (`first_route`): what news its P could contain.
  * The pregame spread (`pre_spread` and both prices, PREGAME_SPREAD_COLUMNS)
    comes from the pregame moneyline's book and follows the same lock:
    refreshed until tip, frozen after; `first_spread` and its prices are
    written once with the rest of `first_*`. Grading never touches them.
  * The row's very first snapshot (`seen_utc`, SEEN_COLUMNS) is written once
    when the row is created, with `seen_note` saying why `first_*` was not
    filled then (no price yet, odds fetch failed, model abstained; blank when
    it was). Never refreshed or graded, so a first snapshot that came later
    than the row is visible, with its reason.
  * Grading fills ONLY result and open/close market columns (moneyline and
    closing spread). It never touches
    a model or pregame-market column.
  * A pending (unfinished) game never receives a closing line.
  * Injury snapshots (data/nba_injuries.csv) follow the same pregame lock:
    replaced only before tip, frozen after.
  * Sleeper/RotoWire projection snapshots (data/nba_sleeper.csv) too: the
    "first" snapshot is written once, the "pre" snapshot is replaced only
    before tip, and both are frozen after. The model does not read them.
  * Every price names its book (`pre_book`, `close_book`: see market.BOOKS).
    Open and close come from one book; rows priced by different books are
    reported separately, never pooled.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import market

NATIVE_PATH = os.path.join("data", "nba_ledger.csv")
RECON_PATH = os.path.join("data", "nba_reconstructed.csv")
# PRESEASON rows (exhibitions, basis "preseason"): same schema and pregame
# lock as the native ledger, written by build_site.score_preseason. Never
# read by the ledger or calibration pages, the hypotheses or any fit.
PRESEASON_PATH = os.path.join("data", "nba_preseason.csv")

COLUMNS = [
    # identity
    "game_id", "slate_date", "season", "tip_utc", "snapshot_utc", "model_tag",
    "basis", "home", "away",
    # model (pregame)
    "gp_home", "gp_away", "home_b2b", "away_b2b", "delta", "p_home", "lean",
    "p_lean",
    # pregame market snapshot
    "pre_book", "pre_home_ml", "pre_away_ml", "pre_q_home",
    # filled by grading
    "open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml",
    "close_q_home", "close_book", "home_pts", "away_pts", "home_won",
    # closing spread from the same book as the closing moneyline: the home
    # line (-5.5 = home gives 5.5) and each side's price at that line
    "close_spread", "close_home_spread_odds", "close_away_spread_odds",
    # first pregame snapshot with a model P and a price: written once, frozen
    "first_snapshot_utc", "first_model_tag", "first_p_home", "first_book",
    "first_home_ml", "first_away_ml", "first_q_home",
    # the injury-report edition (UTC) the first snapshot's build read, and the
    # route that scored it (early / avail / base / v4)
    "first_report_utc", "first_route",
    # the row's earliest pregame snapshot, written once at creation: when,
    # and why first_* was not filled then (blank when it was)
    "seen_utc", "seen_note",
]
# Kalshi (exchange) prices beside the sportsbook's, from 2025-26 on (Kalshi
# lists NBA games from the 2025 play-in). Same locks as their book twins:
# kalshi_pre_* is refreshed with the pregame snapshot until tip; kalshi_first_*
# is written once with first_* (same snapshot and model P); kalshi_open/close
# are written by grading (open = first quoted hour; close = the last 1-minute
# candle ending at or before tip; q = bid/ask midpoints, kalshi.mid_q). The
# pages grade against these where a row has a Kalshi close
# (analysis.market_view); the sportsbook columns are kept.
KALSHI_PRE_COLUMNS = ["kalshi_pre_home_ml", "kalshi_pre_away_ml", "kalshi_pre_q_home"]
KALSHI_FIRST_COLUMNS = ["kalshi_first_home_ml", "kalshi_first_away_ml",
                        "kalshi_first_q_home"]
KALSHI_GRADE_COLUMNS = ["kalshi_open_home_ml", "kalshi_open_away_ml",
                        "kalshi_close_home_ml", "kalshi_close_away_ml",
                        "kalshi_close_q_home"]
KALSHI_COLUMNS = KALSHI_PRE_COLUMNS + KALSHI_FIRST_COLUMNS + KALSHI_GRADE_COLUMNS
COLUMNS = COLUMNS + KALSHI_COLUMNS
# Schema before the pregame spreads: `load` reads it with them blank and the
# next save writes the current schema (2026-10-20, before any native row had
# a model P, so no first snapshot lacks its spread).
PRE_PREGAME_SPREAD_COLUMNS = list(COLUMNS)
# Pregame spread from the pregame moneyline's book and odds response: the home
# line (-5.5 = home gives 5.5) and each side's price. pre_* is refreshed with
# the pregame snapshot until tip, frozen after; first_* is written once with
# the other first_* columns (same snapshot, model P and moneyline).
PREGAME_SPREAD_COLUMNS = ["pre_spread", "pre_home_spread_odds", "pre_away_spread_odds"]
FIRST_SPREAD_COLUMNS = ["first_spread", "first_home_spread_odds",
                        "first_away_spread_odds"]
COLUMNS = COLUMNS + PREGAME_SPREAD_COLUMNS + FIRST_SPREAD_COLUMNS
# Schema before the carryover P: `load` reads it with them blank and the next
# save writes the current schema (2026-10-22, before any native row had a
# model P).
PRE_CARRY_COLUMNS = list(COLUMNS)
# Model v7 (owner's decision 2026-10-22): games 1-9 are scored with roster
# talent; the v2-v6 carryover logit's P is recorded beside it for those games
# (blank otherwise), because hypotheses H1 and H6 were registered on it and
# stay on it (analysis.carry_view). carry_p_home is refreshed with the pregame
# snapshot until tip; first_carry_p_home is written once with first_*.
CARRY_COLUMNS = ["carry_p_home"]
FIRST_CARRY_COLUMNS = ["first_carry_p_home"]
COLUMNS = COLUMNS + CARRY_COLUMNS + FIRST_CARRY_COLUMNS
SEEN_COLUMNS = ["seen_utc", "seen_note"]
FIRST_COLUMNS = (COLUMNS[COLUMNS.index("first_snapshot_utc"):
                         COLUMNS.index("seen_utc")] + KALSHI_FIRST_COLUMNS
                 + FIRST_SPREAD_COLUMNS + FIRST_CARRY_COLUMNS)
# Where each first_* column comes from: a pregame column, or (report, route)
# a key of the scored row that is not itself a ledger column.
FIRST_SOURCE = {"first_snapshot_utc": "snapshot_utc", "first_model_tag": "model_tag",
                "first_p_home": "p_home", "first_book": "pre_book",
                "first_home_ml": "pre_home_ml", "first_away_ml": "pre_away_ml",
                "first_q_home": "pre_q_home", "first_report_utc": "report_utc",
                "first_route": "route",
                **{f"kalshi_first_{k}": f"kalshi_pre_{k}"
                   for k in ("home_ml", "away_ml", "q_home")},
                **dict(zip(FIRST_SPREAD_COLUMNS, PREGAME_SPREAD_COLUMNS)),
                **dict(zip(FIRST_CARRY_COLUMNS, CARRY_COLUMNS))}
# Columns written once and never refreshed or graded.
WRITE_ONCE_COLUMNS = FIRST_COLUMNS + SEEN_COLUMNS
# String-valued columns (kept as object dtype).
TEXT_COLUMNS = ("pre_book", "close_book", "first_book", "first_model_tag",
                "first_snapshot_utc", "first_report_utc", "first_route",
                "seen_utc", "seen_note")
SPREAD_COLUMNS = ["close_spread", "close_home_spread_odds",
                  "close_away_spread_odds"]
# Schema before the Kalshi columns: `load` reads it with them blank and the
# next save writes the current schema (2026-10-06; the native ledger was empty).
PRE_KALSHI_COLUMNS = [c for c in PRE_PREGAME_SPREAD_COLUMNS if c not in KALSHI_COLUMNS]
# Schema before the report / route / seen columns: `load` reads it with them
# blank and the next save writes the current schema. (The native ledger was
# empty when they were added, 2026-10-01.)
PRE_SEEN_COLUMNS = [c for c in PRE_KALSHI_COLUMNS if c not in
                    ("first_report_utc", "first_route", *SEEN_COLUMNS)]
# Schema before the first-snapshot columns: `load` reads it with them blank
# and the next save writes the current schema. (The native ledger was empty
# when they were added, 2026-10-01, so no row's first snapshot is unknown.)
PRE_FIRST_COLUMNS = [c for c in PRE_KALSHI_COLUMNS if c not in WRITE_ONCE_COLUMNS]
# Schema before the spread columns: `load` reads it with the spreads blank
# and the next save writes the current schema.
PRE_SPREAD_COLUMNS = [c for c in PRE_FIRST_COLUMNS if c not in SPREAD_COLUMNS]
# Schema before the book columns. Every price in such a file is DraftKings
# (the parser read nothing else), so `load` labels it "dk".
LEGACY_COLUMNS = [c for c in PRE_SPREAD_COLUMNS
                  if c not in ("pre_book", "close_book")]
PREGAME_COLUMNS = ([c for c in COLUMNS[:COLUMNS.index("pre_q_home") + 1]]
                   + KALSHI_PRE_COLUMNS + PREGAME_SPREAD_COLUMNS + CARRY_COLUMNS)
GRADE_COLUMNS = [c for c in COLUMNS
                 if c not in PREGAME_COLUMNS and c not in WRITE_ONCE_COLUMNS]


# Pregame injury snapshots: one row per listed player (or one "NONE" row per
# team with nobody listed), from the latest snapshot taken before tip. Same
# lock as the pregame columns: written or replaced only while now < tip,
# frozen after. A failed fetch (None) leaves the previous snapshot in place.
INJURY_PATH = os.path.join("data", "nba_injuries.csv")
INJURY_COLUMNS = ["game_id", "tip_utc", "snapshot_utc", "team", "player_id",
                  "name", "status", "detail"]


def utc_now():
    return datetime.now(timezone.utc)


def parse_utc(s):
    """ESPN '2026-03-10T23:00Z' (or with seconds / offset) -> aware datetime."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return None
    ts = pd.Timestamp(str(s))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()


def fmt_utc(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty():
    return pd.DataFrame(columns=COLUMNS)


def load(path):
    if not os.path.exists(path):
        return empty()
    df = pd.read_csv(path, dtype={"game_id": str})
    legacy = "close_book" not in df.columns
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    for c in TEXT_COLUMNS:
        df[c] = df[c].astype(object)
    if legacy:
        df.loc[df["pre_q_home"].notna(), "pre_book"] = "dk"
        df.loc[df["close_q_home"].notna(), "close_book"] = "dk"
    return df[COLUMNS]


def save(df, path):
    """Atomic write: a killed process never leaves a half-written CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".", suffix=".tmp",
                               dir=os.path.dirname(path) or ".")
    os.close(fd)
    df = df[COLUMNS].sort_values(["slate_date", "tip_utc", "game_id"],
                                 kind="stable")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def upsert_pregame(led, rows, now=None, basis="native"):
    """Add or refresh pregame rows. Returns (ledger, accepted, rejected).

    A row is written only if `now` is before its tip. An existing row is
    replaced only while its game is still before tip AND ungraded, so once a
    game starts its pregame record is frozen for good. The first snapshot
    with a model P and a price fills `first_*` once; later ones never do.
    A new row also gets `seen_*` once. Optional keys of a row: `report_utc`
    and `route` (copied to first_*), `price_note` (why it has no price).
    `basis` labels the rows ("native"; "preseason" for PRESEASON_PATH).
    """
    now = now or utc_now()
    led = led.copy()
    accepted, rejected = [], []
    for r in rows:
        tip = parse_utc(r.get("tip_utc"))
        if tip is None or now >= tip:
            rejected.append((r.get("game_id"), "at or after tip"))
            continue
        gid = str(r["game_id"])
        rec = {c: r.get(c, np.nan) for c in COLUMNS}
        rec["game_id"] = gid
        rec["snapshot_utc"] = fmt_utc(now)
        rec["basis"] = basis
        for c in WRITE_ONCE_COLUMNS:
            rec[c] = np.nan
        src = {**rec, "report_utc": r.get("report_utc", np.nan),
               "route": r.get("route", np.nan)}
        has_p = pd.notna(pd.to_numeric(rec["p_home"], errors="coerce"))
        has_q = pd.notna(pd.to_numeric(rec["pre_q_home"], errors="coerce"))
        usable = has_p and has_q
        hit = led.index[led["game_id"].astype(str) == gid]
        if len(hit):
            old = led.loc[hit[0]]
            if pd.notna(old["home_won"]):
                rejected.append((gid, "already graded"))
                continue
            for c in PREGAME_COLUMNS:
                if isinstance(rec[c], str) and led[c].dtype != object:
                    led[c] = led[c].astype(object)
                led.at[hit[0], c] = rec[c]
            if usable and pd.isna(old["first_snapshot_utc"]):
                for c in FIRST_COLUMNS:
                    if led[c].dtype != object:
                        led[c] = led[c].astype(object)
                    led.at[hit[0], c] = src[FIRST_SOURCE[c]]
        else:
            if usable:
                for c in FIRST_COLUMNS:
                    rec[c] = src[FIRST_SOURCE[c]]
            rec["seen_utc"] = rec["snapshot_utc"]
            rec["seen_note"] = "" if usable else "; ".join(
                n for n in ((None if has_p else "no model P"),
                            (None if has_q else
                             "no price" + (f" ({r['price_note']})"
                                           if r.get("price_note") else "")))
                if n)
            led = pd.concat([led, pd.DataFrame([rec])[COLUMNS]],
                            ignore_index=True) if len(led) else \
                pd.DataFrame([rec])[COLUMNS]
        accepted.append(gid)
    return led, accepted, rejected


# Sleeper / RotoWire pregame projections (owner's decision 2026-10-22; sleeper.py):
# one row per player Sleeper lists for the game (minutes 0 = listed, not
# expected to play) with his projected line and injury status at the snapshot.
# kind "first": the first snapshot that captured the game, written once;
# kind "pre": the latest snapshot before tip, replaced each build. Both frozen
# at tip. Research data; the model and the pages do not read it.
SLEEPER_PATH = os.path.join("data", "nba_sleeper.csv")
SLEEPER_COLUMNS = ["game_id", "tip_utc", "snapshot_utc", "kind", "team", "sleeper_id",
                   "name", "minutes", "pts", "reb", "ast", "stl", "blk", "to", "tpm",
                   "dd", "td", "injury_status", "injury_body_part"]
SLEEPER_KINDS = ("first", "pre")


def load_injuries(path=INJURY_PATH):
    if not os.path.exists(path):
        return pd.DataFrame(columns=INJURY_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df[INJURY_COLUMNS]


def save_injuries(df, path=INJURY_PATH):
    """Atomic write, sorted like the ledger."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".", suffix=".tmp",
                               dir=os.path.dirname(path) or ".")
    os.close(fd)
    df = df[INJURY_COLUMNS].sort_values(["tip_utc", "game_id", "team", "name"],
                                        kind="stable")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def upsert_injuries(inj, snaps, now=None):
    """Replace each game's injury rows with a new pregame snapshot.

    snaps: {game_id: (tip_utc, rows or None)}. Accepted only while now < tip;
    None (fetch failed / not provided) keeps the existing rows. Returns
    (frame, accepted, rejected).
    """
    now = now or utc_now()
    inj = inj.copy()
    accepted, rejected = [], []
    for gid, (tip_s, rows) in snaps.items():
        gid = str(gid)
        tip = parse_utc(tip_s)
        if tip is None or now >= tip:
            rejected.append((gid, "at or after tip"))
            continue
        if rows is None:
            rejected.append((gid, "no injury data"))
            continue
        new = pd.DataFrame([{**{c: "" for c in INJURY_COLUMNS}, **r,
                             "game_id": gid, "tip_utc": str(tip_s),
                             "snapshot_utc": fmt_utc(now)} for r in rows],
                           columns=INJURY_COLUMNS)
        keep = inj[inj["game_id"].astype(str) != gid]
        inj = pd.concat([d for d in (keep, new) if len(d)], ignore_index=True) \
            if len(keep) or len(new) else pd.DataFrame(columns=INJURY_COLUMNS)
        accepted.append(gid)
    return inj[INJURY_COLUMNS] if len(inj) else pd.DataFrame(columns=INJURY_COLUMNS), \
        accepted, rejected


def load_sleeper(path=SLEEPER_PATH):
    if not os.path.exists(path):
        return pd.DataFrame(columns=SLEEPER_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df[SLEEPER_COLUMNS]


def save_sleeper(df, path=SLEEPER_PATH):
    """Atomic write, sorted by tip, game, kind, team and projected minutes."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".", suffix=".tmp",
                               dir=os.path.dirname(path) or ".")
    os.close(fd)
    df = df[SLEEPER_COLUMNS].assign(_m=pd.to_numeric(df["minutes"], errors="coerce"))
    df = df.sort_values(["tip_utc", "game_id", "kind", "team", "_m", "name"],
                        ascending=[True, True, True, True, False, True], kind="stable")
    df.drop(columns="_m").to_csv(tmp, index=False)
    os.replace(tmp, path)


def upsert_sleeper(df, snaps, now=None):
    """Record a pregame Sleeper snapshot per game.

    snaps: {game_id: (tip_utc, rows or None)}. Accepted only while now < tip;
    None or [] (fetch failed, nothing listed) keeps the existing rows. The
    rows become the game's "pre" snapshot (replacing the previous one) and,
    if the game has none yet, its "first". Returns (frame, accepted,
    rejected).
    """
    now = now or utc_now()
    df = df.copy()
    accepted, rejected = [], []
    for gid, (tip_s, rows) in snaps.items():
        gid = str(gid)
        tip = parse_utc(tip_s)
        if tip is None or now >= tip:
            rejected.append((gid, "at or after tip"))
            continue
        if not rows:
            rejected.append((gid, "no projections"))
            continue
        base = [{**{c: "" for c in SLEEPER_COLUMNS}, **r, "game_id": gid,
                 "tip_utc": str(tip_s), "snapshot_utc": fmt_utc(now)} for r in rows]
        game = df["game_id"].astype(str) == gid
        has_first = (game & (df["kind"] == "first")).any()
        new = [{**r, "kind": "pre"} for r in base]
        if not has_first:
            new += [{**r, "kind": "first"} for r in base]
        keep = df[~(game & (df["kind"] == "pre"))]
        add = pd.DataFrame(new, columns=SLEEPER_COLUMNS)
        df = pd.concat([d for d in (keep, add) if len(d)], ignore_index=True)
        accepted.append(gid)
    return (df[SLEEPER_COLUMNS] if len(df) else pd.DataFrame(columns=SLEEPER_COLUMNS),
            accepted, rejected)


def apply_result(led, game_id, game, odds, kalshi_odds=None):
    """Grade one row from a scoreboard game dict and ONE book's odds (or None).

    `odds` is what market.pick_close returns: that book's open/close plus its
    `book` label. `kalshi_odds` (kalshi.grade_odds) holds the KALSHI_GRADE_COLUMNS
    keys, written beside the book's. Writes only GRADE_COLUMNS, and only for a
    completed game with scores.
    """
    hit = led.index[led["game_id"].astype(str) == str(game_id)]
    if not len(hit) or not game or not game.get("completed"):
        return False
    hp, ap = game.get("home_pts"), game.get("away_pts")
    if hp is None or ap is None or hp == ap:
        return False
    i = hit[0]
    led.at[i, "home_pts"] = hp
    led.at[i, "away_pts"] = ap
    led.at[i, "home_won"] = int(hp > ap)
    if odds:
        for c in ("open_home_ml", "open_away_ml", "close_home_ml",
                  "close_away_ml"):
            if odds.get(c) is not None:
                led.at[i, c] = odds[c]
        # an exchange supplies its own q (kalshi.mid_q); a book is devigged
        q = odds.get("close_q_home")
        if q is None or not np.isfinite(q):
            q = market.devig(odds.get("close_home_ml"), odds.get("close_away_ml"))
        if np.isfinite(q):
            led.at[i, "close_q_home"] = round(q, 5)
            if led["close_book"].dtype != object:
                led["close_book"] = led["close_book"].astype(object)
            led.at[i, "close_book"] = odds.get("book", "dk")
            # the spread is kept only with its book's moneyline close, so
            # both closes on a row always name the same book
            for c in SPREAD_COLUMNS:
                if odds.get(c) is not None:
                    led.at[i, c] = odds[c]
    for c in KALSHI_GRADE_COLUMNS:
        v = (kalshi_odds or {}).get(c)
        if v is not None and np.isfinite(v):
            led.at[i, c] = v
    return True


def pending(led, today):
    """Ungraded rows from slates before `today` (ET date string)."""
    if not len(led):
        return led
    return led[led["home_won"].isna() & (led["slate_date"].astype(str) < today)]


CLOSE_RETRY_DAYS = 7   # graded rows with no close are re-asked this long
KALSHI_FROM_SEASON = 2026   # Kalshi's KXNBAGAME lists games from the 2025 play-in


def missing_close(led, today, days=CLOSE_RETRY_DAYS, kalshi=False):
    """Graded rows with no closing pair from the last `days` slates before
    `today`: grading found the result but no close (a failed or empty odds
    fetch), so the next builds ask again. `kalshi` asks about the Kalshi
    close instead (seasons from KALSHI_FROM_SEASON). Finished games only, so
    a close is still never written to a pending row."""
    if not len(led):
        return led
    d = pd.to_datetime(led["slate_date"].astype(str), errors="coerce")
    t = pd.Timestamp(today)
    won = pd.to_numeric(led["home_won"], errors="coerce").isin([0, 1])
    pre = "kalshi_close" if kalshi else "close"
    no_close = (pd.to_numeric(led[f"{pre}_home_ml"], errors="coerce").isna()
                | pd.to_numeric(led[f"{pre}_away_ml"], errors="coerce").isna())
    if kalshi:
        no_close &= pd.to_numeric(led["season"], errors="coerce") >= KALSHI_FROM_SEASON
    return led[won & no_close & (d < t) & (d >= t - pd.Timedelta(days=days))]


def graded(df):
    """Rows with a final result, as floats where numeric."""
    if not len(df):
        return df
    g = df[pd.to_numeric(df["home_won"], errors="coerce").isin([0, 1])].copy()
    for c in ("p_home", "delta", "home_won", "close_home_ml", "close_away_ml",
              "close_q_home", "pre_home_ml", "pre_away_ml", "pre_q_home",
              "p_lean", "home_pts", "away_pts") + tuple(SPREAD_COLUMNS):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g
