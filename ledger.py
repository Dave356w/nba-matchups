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
  * Grading fills ONLY result and open/close market columns. It never touches
    a model or pregame-market column.
  * A pending (unfinished) game never receives a closing line.
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

COLUMNS = [
    # identity
    "game_id", "slate_date", "season", "tip_utc", "snapshot_utc", "model_tag",
    "basis", "home", "away",
    # model (pregame)
    "gp_home", "gp_away", "home_b2b", "away_b2b", "delta", "p_home", "lean",
    "p_lean",
    # pregame market snapshot
    "pre_home_ml", "pre_away_ml", "pre_q_home",
    # filled by grading
    "open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml",
    "close_q_home", "home_pts", "away_pts", "home_won",
]
PREGAME_COLUMNS = [c for c in COLUMNS[:COLUMNS.index("pre_q_home") + 1]]
GRADE_COLUMNS = [c for c in COLUMNS if c not in PREGAME_COLUMNS]


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
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
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


def upsert_pregame(led, rows, now=None):
    """Add or refresh pregame rows. Returns (ledger, accepted, rejected).

    A row is written only if `now` is before its tip. An existing row is
    replaced only while its game is still before tip AND ungraded, so once a
    game starts its pregame record is frozen for good.
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
        rec["basis"] = "native"
        hit = led.index[led["game_id"].astype(str) == gid]
        if len(hit):
            old = led.loc[hit[0]]
            if pd.notna(old["home_won"]):
                rejected.append((gid, "already graded"))
                continue
            for c in PREGAME_COLUMNS:
                led.at[hit[0], c] = rec[c]
        else:
            led = pd.concat([led, pd.DataFrame([rec])[COLUMNS]],
                            ignore_index=True) if len(led) else \
                pd.DataFrame([rec])[COLUMNS]
        accepted.append(gid)
    return led, accepted, rejected


def apply_result(led, game_id, game, odds):
    """Grade one row from a scoreboard game dict and DK odds dict (or None).

    Writes only GRADE_COLUMNS, and only for a completed game with scores.
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
        q = market.devig(odds.get("close_home_ml"), odds.get("close_away_ml"))
        if np.isfinite(q):
            led.at[i, "close_q_home"] = round(q, 5)
    return True


def pending(led, today):
    """Ungraded rows from slates before `today` (ET date string)."""
    if not len(led):
        return led
    return led[led["home_won"].isna() & (led["slate_date"].astype(str) < today)]


def graded(df):
    """Rows with a final result, as floats where numeric."""
    if not len(df):
        return df
    g = df[pd.to_numeric(df["home_won"], errors="coerce").isin([0, 1])].copy()
    for c in ("p_home", "delta", "home_won", "close_home_ml", "close_away_ml",
              "close_q_home", "pre_home_ml", "pre_away_ml", "pre_q_home",
              "p_lean"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g
