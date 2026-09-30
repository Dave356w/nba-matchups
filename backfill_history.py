#!/usr/bin/env python3
"""Reconstruct completed seasons against the historical sportsbook close.

  python backfill_history.py --seasons 2025 2026            # fetch prices too
  python backfill_history.py --seasons 2025 2026 --rescore  # keep prices/results

For each test season Y this reproduces the report's leave-one-season-out
protocol, then attaches one book's open/close for every game (DraftKings,
else ESPN BET; `close_book` says which -- see market.pick_close):

  * composite weights: ridge fit on WEIGHT_YEARS with Y removed;
  * logit:             fit on game logs from GAME_YEARS with Y removed;
  * features:          decayed pregame totals, games strictly before each date.

Model v5 (default; --v4 for v4): v4 below plus luck_def and talent_diff in
both games-10+ logits (nba_composite.V5_FEATURES; talent from the cached box
scores); games whose v5 terms are missing keep the v4 base prediction.

Model v4 (build_site.MODEL_TAG_V4 / _V4_AVAIL), every fit without Y:
  * games 10+:  sigma(a + b*delta + c*b2b_net + e*delta*phase), logit on
                PHASE_YEARS; where the NBA injury report and box history
                exist (REPORT_YEARS), the availability logit with the same
                phase term (player_availability.FEATURES_V4) fit on the other
                report seasons;
  * games 1-9:  the v2 carryover model (logit on GAME_YEARS without Y).

--rescore replaces only the model columns (delta, p_home, lean, p_lean,
model_tag, games played) of rows already in data/nba_reconstructed.csv and
keeps their prices and results, so no odds are refetched.

Rows are written to data/nba_reconstructed.csv with basis="reconstructed".
They are hindsight analysis -- the design choices (half-life, B2B term) were
made on overlapping seasons and the price is the close, not a price that was
available when a decision would have been made. They never enter the native
ledger and every page shows them separately.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import build_site
import cold_start
import ledger
import market
import nba_composite as nc
import player_availability as pav

WEIGHT_YEARS = [2015, 2016, 2017, 2018, 2019] + list(range(2021, 2027))
GAME_YEARS = [2023, 2024, 2025, 2026]
PHASE_YEARS = [2016, 2017, 2018, 2019] + list(range(2021, 2027))
REPORT_YEARS = [2023, 2024, 2025, 2026]
MODEL_COLUMNS = ["gp_home", "gp_away", "delta", "p_home", "lean", "p_lean",
                 "model_tag"]


def training_years(years, y, walk_forward=False):
    """Seasons a fit for test season y may use: every other season (leave one
    season out), or only earlier ones (walk-forward)."""
    return [t for t in years if (t < y if walk_forward else t != y)]


def reconstruct_season(y, game_years=GAME_YEARS, weight_years=WEIGHT_YEARS,
                       phase_years=PHASE_YEARS, report_years=REPORT_YEARS,
                       terms=None, walk_forward=False, v5=False, talent=None):
    """Predictions, routed as in production, for every qualifying game of
    season y: `route` is "early" (games 1-9, carryover), "avail" (games 10+
    the injury report covers) or "base" (other games 10+).

    Every fit leaves season y out; with walk_forward=True it uses earlier
    seasons only (research/walk_forward.py). `terms` ({season:
    player_availability.season_terms frame, covered games only}) enables the
    availability arm for games 10+ of report seasons; None skips it.

    v5=True adds luck_def and talent_diff to both games-10+ logits (talent:
    {season: talent function}; default player_availability.season_talent,
    from the cached box scores). As in the daily build, a game whose v5 terms
    are missing keeps the v4 base prediction and tag. p_v4 always holds the
    v4 routing's prediction for the same game (the v5 gate). With v5, p_fixed
    is the v5 routing with the availability logit's luck_def / talent_diff
    coefficients fixed from the v5 base fit (player_availability.fit_fixed);
    research only."""
    def use(years):
        out = training_years(years, y, walk_forward)
        if not out:
            raise SystemExit(f"season {y}: no training seasons in {list(years)}")
        return out
    memo = dict(talent or {})

    def tal(t):
        if t not in memo:
            try:
                memo[t] = pav.season_talent(t)
            except Exception as e:  # noqa: BLE001 - rows fall back to v4
                print(f"season {t}: no talent ({e!r})", flush=True)
                memo[t] = None
        return memo[t]

    def games_for(t):
        return nc.build_games(t, weights, talent=tal(t)) if v5 else nc.build_games(t, weights)

    weights = nc.fit_weights(use(weight_years))
    phase_train = use(phase_years)
    games = {t: games_for(t) for t in set(phase_train) | {y}}
    train = pd.concat([games[t] for t in phase_train])
    feats = nc.PHASE_FEATURES
    model = nc.fit_logit(train[feats].values, train["win"], feats)
    model["features"] = feats
    test = games[y].copy()
    test["p_home"] = nc.predict(model, test[feats].values)
    test["p_v4"] = test["p_home"]
    test["model_tag"] = build_site.MODEL_TAG_V4
    test["route"] = "base"
    if v5:
        f5 = nc.V5_FEATURES
        ok = np.isfinite(train[f5].to_numpy(float)).all(axis=1)
        model = nc.fit_logit(train.loc[ok, f5].values, train.loc[ok, "win"], f5)
        fin = np.isfinite(test[f5].to_numpy(float)).all(axis=1)
        test.loc[fin, "p_home"] = nc.predict(model, test.loc[fin, f5].values)
        test.loc[fin, "model_tag"] = build_site.MODEL_TAG_V5
        print(f"season {y}: v5 base on {int(fin.sum())}/{len(test)} games 10+ "
              f"(fit n={int(ok.sum())})", flush=True)
    test["p_fixed"] = test["p_home"]
    base5 = model
    if terms and y in terms:
        others = [t for t in training_years(report_years, y, walk_forward)
                  if t in terms]
        if others:
            tr = pd.concat([pav.with_terms(games[t] if t in games else games_for(t),
                                           terms[t]) for t in others])
            te = pav.with_terms(test, terms[y])
            tkey = (pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d")
                    + test["home"] + test["away"])
            key = te["slate_date"] + te["home"] + te["away"]
            for af, tag, col in ((pav.FEATURES_V4, build_site.MODEL_TAG_V4_AVAIL, "p_v4"),
                                 (pav.FEATURES_V5, build_site.MODEL_TAG_V5_AVAIL, "p_home")):
                if af is pav.FEATURES_V5 and not v5:
                    continue
                ok = np.isfinite(tr[af].to_numpy(float)).all(axis=1)
                am = nc.fit_logit(tr.loc[ok, af].values, tr.loc[ok, "win"], af)
                fin = np.isfinite(te[af].to_numpy(float)).all(axis=1)
                pmap = dict(zip(key[fin], nc.predict(am, te.loc[fin, af].values)))
                hit = tkey.isin(pmap)
                test.loc[hit, col] = tkey[hit].map(pmap)
                if col == "p_home" or not v5:
                    test.loc[hit, "p_home"] = tkey[hit].map(pmap)
                    test.loc[hit, "model_tag"] = tag
                    test.loc[hit, "route"] = "avail"
                print(f"season {y}: availability arm ({'v5' if af is pav.FEATURES_V5 else 'v4'}) "
                      f"on {int(hit.sum())}/{len(test)} games 10+ (fit on {others}, "
                      f"n={int(ok.sum())})", flush=True)
                if af is pav.FEATURES_V5:
                    test.loc[hit, "p_fixed"] = test.loc[hit, "p_home"]
                    fm = pav.fit_fixed(tr.loc[ok], base5)
                    fx = dict(zip(key[fin], nc.predict(fm, te.loc[fin, af].values)))
                    test.loc[hit, "p_fixed"] = tkey[hit].map(fx)
                    print(f"season {y}: fixed-talent arm  " + "  ".join(
                        f"{f}={c:+.4f}" for f, c in zip(fm["features"], fm["coef"]))
                        + f"  (free fit: " + "  ".join(
                        f"{f}={c:+.4f}" for f, c in zip(am["features"], am["coef"])
                        if f in pav.FIXED_V5) + ")", flush=True)
    # games 1-9: the v2 carryover model, its logit also fit without y.
    early = cold_start.fit_early(use(game_years), weights)
    e = cold_start.early_games(y, weights, min_gp=1, window=nc.MIN_GAMES)
    if len(e):
        e["p_home"] = nc.predict(early, e[cold_start.FEATURES].values)
        e["p_v4"] = e["p_home"]
        e["p_fixed"] = e["p_home"]
        e["model_tag"] = build_site.MODEL_TAG_V5 if v5 else build_site.MODEL_TAG_V4
        e["route"] = "early"
        test = pd.concat([test, e], ignore_index=True)
    return test, weights, model


def rescore(recon, test):
    """Replace the model columns of existing reconstructed rows with `test`'s
    predictions (matched on date and teams); prices and results are kept."""
    out = recon.copy()
    t = test.copy()
    t["slate_date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
    t = t.drop_duplicates(["slate_date", "home", "away"]).set_index(
        ["slate_date", "home", "away"])
    idx = pd.MultiIndex.from_frame(out[["slate_date", "home", "away"]].astype(str))
    hit = idx.isin(t.index)
    src = t.reindex(idx[hit])
    p = src["p_home"].astype(float).to_numpy()
    lean_home = p >= 0.5
    new = dict(
        gp_home=src["gp_home"].to_numpy() if "gp_home" in src else np.nan,
        gp_away=src["gp_away"].to_numpy() if "gp_away" in src else np.nan,
        delta=np.round(src["delta"].astype(float).to_numpy(), 3),
        p_home=np.round(p, 5),
        lean=np.where(lean_home, src.index.get_level_values("home"),
                      src.index.get_level_values("away")),
        p_lean=np.round(np.where(lean_home, p, 1 - p), 5),
        model_tag=src["model_tag"].to_numpy())
    for c in MODEL_COLUMNS:
        out[c] = out[c].astype(object)
        out.loc[hit, c] = new[c]
    return out, int(hit.sum())


def attach_espn(test, sleep=0.25):
    """Map each (date, away, home) to an ESPN event; fetch one book's open/close."""
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
                odds = market.pick_close(market.book_odds(g["game_id"]))
            except Exception:  # noqa: BLE001
                odds = {}
            time.sleep(sleep)
            p = float(r["p_home"])
            lean_home = p >= 0.5
            rec = dict(
                game_id=g["game_id"], slate_date=date, season=int(r["year"]),
                tip_utc=g["tip_utc"], snapshot_utc=np.nan,
                model_tag=r.get("model_tag", build_site.MODEL_TAG),
                basis="reconstructed",
                home=r["home"], away=r["away"],
                gp_home=r.get("gp_home", np.nan), gp_away=r.get("gp_away", np.nan),
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
    ap.add_argument("--rescore", action="store_true",
                    help="replace model columns only; keep prices and results")
    ap.add_argument("--no-avail", action="store_true",
                    help="skip the injury-report arm (v4 base everywhere)")
    ap.add_argument("--cache", default=pav.DEFAULT_CACHE)
    ap.add_argument("--v4", action="store_true",
                    help="score with the v4 routing (no luck_def / talent_diff)")
    a = ap.parse_args(argv)
    terms = None
    if not a.no_avail:
        archive = pav.ReportArchive(os.path.join(a.cache, "injury_reports"))
        terms = {}
        try:
            for t in REPORT_YEARS:
                terms[t], s = pav.season_terms(t, archive, a.cache)
                print(f"report season {t}: {s['with_report']}/{s['games']} "
                      f"games with a report, {len(s['covered'])} covered",
                      flush=True)
        finally:
            archive.save()
    if a.rescore:
        df = ledger.load(ledger.RECON_PATH)
        for y in a.seasons:
            test, _, model = reconstruct_season(y, terms=terms, v5=not a.v4)
            print(f"season {y}: {len(test)} games scored; logit {model}", flush=True)
            df, n = rescore(df, test)
            print(f"season {y}: rescored {n} reconstructed rows; tags "
                  f"{df['model_tag'].value_counts().to_dict()}", flush=True)
        ledger.save(df, ledger.RECON_PATH)
        print(f"wrote {ledger.RECON_PATH}: {len(df)} rows", flush=True)
        return 0
    out = [ledger.load(ledger.RECON_PATH)]
    for y in a.seasons:
        test, _, model = reconstruct_season(y, terms=terms, v5=not a.v4)
        print(f"season {y}: {len(test)} games scored; logit {model}", flush=True)
        rec = attach_espn(test)
        by_book = rec["close_book"].value_counts().to_dict()
        print(f"season {y}: {len(rec)} matched to ESPN; closes by book {by_book}",
              flush=True)
        out.append(rec)
    df = pd.concat([d for d in out if len(d)], ignore_index=True)
    df = df.drop_duplicates("game_id", keep="last")
    ledger.save(df, ledger.RECON_PATH)
    print(f"wrote {ledger.RECON_PATH}: {len(df)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
