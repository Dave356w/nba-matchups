#!/usr/bin/env python3
"""Reconstruct completed seasons against the historical DraftKings close.

  python backfill_history.py --seasons 2025 2026

For each test season Y this reproduces the report's leave-one-season-out
protocol, then attaches ESPN's DraftKings open/close for every game:

  * composite weights: ridge fit on WEIGHT_YEARS with Y removed;
  * logit:             fit on game logs from GAME_YEARS with Y removed;
  * features:          decayed pregame totals, games strictly before each date.

Rows are written to data/nba_reconstructed.csv with basis="reconstructed".
They are hindsight analysis -- the design choices (half-life, B2B term) were
made on overlapping seasons and the price is the close, not a price that was
available when a decision would have been made. They never enter the native
ledger and every page shows them separately.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

import build_site
import ledger
import market
import nba_composite as nc

WEIGHT_YEARS = [2015, 2016, 2017, 2018, 2019] + list(range(2021, 2027))
GAME_YEARS = [2023, 2024, 2025, 2026]


def reconstruct_season(y, game_years=GAME_YEARS, weight_years=WEIGHT_YEARS):
    """Leave-season-y-out predictions for every qualifying game of season y."""
    weights = nc.fit_weights([w for w in weight_years if w != y])
    train = pd.concat([nc.build_games(t, weights) for t in game_years if t != y])
    model = nc.fit_logit(train[nc.LOGIT_FEATURES].values, train["win"],
                         nc.LOGIT_FEATURES)
    test = nc.build_games(y, weights)
    test["p_home"] = nc.predict(model, test[nc.LOGIT_FEATURES].values)
    return test, weights, model


def attach_espn(test, sleep=0.25):
    """Map each (date, away, home) to an ESPN event; fetch DK open/close."""
    rows = []
    for date, day in test.groupby(test["date"].dt.strftime("%Y-%m-%d")):
        try:
            sb = {(g["away"], g["home"]): g for g in market.scoreboard(date)}
        except Exception as e:  # noqa: BLE001
            print(f"{date}: scoreboard failed {e!r}", flush=True)
            continue
        for _, r in day.iterrows():
            g = sb.get((r["away"], r["home"]))
            if g is None:
                continue
            try:
                odds = market.dk_odds(g["game_id"]) or {}
            except Exception:  # noqa: BLE001
                odds = {}
            time.sleep(sleep)
            p = float(r["p_home"])
            lean_home = p >= 0.5
            rec = dict(
                game_id=g["game_id"], slate_date=date, season=int(r["year"]),
                tip_utc=g["tip_utc"], snapshot_utc=np.nan,
                model_tag=build_site.MODEL_TAG, basis="reconstructed",
                home=r["home"], away=r["away"], gp_home=np.nan, gp_away=np.nan,
                home_b2b=r["h_b2b"], away_b2b=r["a_b2b"],
                delta=round(float(r["delta"]), 3), p_home=round(p, 5),
                lean=r["home"] if lean_home else r["away"],
                p_lean=round(p if lean_home else 1 - p, 5),
            )
            rows.append(rec)
            led = pd.DataFrame([rec], columns=ledger.COLUMNS)
            # Scores come from ESPN so the graded result and the price share
            # one source; BBR's win flag must agree (checked below).
            ledger.apply_result(led, g["game_id"], g, odds)
            rows[-1] = led.iloc[0].to_dict()
            if pd.notna(rows[-1]["home_won"]) and int(rows[-1]["home_won"]) != int(r["win"]):
                print(f"WARNING {date} {r['away']}@{r['home']}: ESPN/BBR "
                      "winner disagree", flush=True)
    return pd.DataFrame(rows, columns=ledger.COLUMNS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    a = ap.parse_args(argv)
    out = [ledger.load(ledger.RECON_PATH)]
    for y in a.seasons:
        test, _, model = reconstruct_season(y)
        print(f"season {y}: {len(test)} games scored; logit {model}", flush=True)
        rec = attach_espn(test)
        have = pd.to_numeric(rec["close_q_home"], errors="coerce").notna().sum()
        print(f"season {y}: {len(rec)} matched to ESPN, {have} with DK close",
              flush=True)
        out.append(rec)
    df = pd.concat([d for d in out if len(d)], ignore_index=True)
    df = df.drop_duplicates("game_id", keep="last")
    ledger.save(df, ledger.RECON_PATH)
    print(f"wrote {ledger.RECON_PATH}: {len(df)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
