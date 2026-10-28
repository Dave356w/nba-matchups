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

Model v9 (default; --v8 for v8): v8's routing and seasons (below), each
route fitted by least squares on the final home margin; P = Phi(m / sigma)
and margin_home = m (nba_composite.fit_margin). --seasons may include the
season in progress (2027 = 2026-27): its played games are rebuilt with every
fit on earlier seasons only (leave-one-season-out leaves nothing of it in),
priced at the close, as reconstructed (hindsight) rows.

Model v8 (--v8): one faded rating and one pair of logits for every game
(reconstruct_season_v8).

Model v6 (--v6; --v5 for v5, --v4 for v4): v5 plus ft_diff, the own FT%
gap (nba_composite.V6_FEATURES), in both games-10+ logits.

Model v5: v4 below plus luck_def and talent_diff in
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
import ledger
import market
import nba_composite as nc
import player_availability as pav

WEIGHT_YEARS = [2015, 2016, 2017, 2018, 2019] + list(range(2021, 2027))
GAME_YEARS = [2023, 2024, 2025, 2026]
PHASE_YEARS = [2016, 2017, 2018, 2019] + list(range(2021, 2027))
REPORT_YEARS = [2023, 2024, 2025, 2026]
MODEL_COLUMNS = ["gp_home", "gp_away", "delta", "p_home", "lean", "p_lean",
                 "model_tag", "margin_home"]
# v7: the early (games 1-9) talent logit's seasons, the box-score seasons
# (research/early_roster.py; the first needs the BPM table of the season
# before it). The carryover logit keeps GAME_YEARS.
EARLY_TALENT_YEARS = [2016, 2017, 2018, 2019] + list(range(2021, 2027))


def training_years(years, y, walk_forward=False):
    """Seasons a fit for test season y may use: every other season (leave one
    season out), or only earlier ones (walk-forward)."""
    return [t for t in years if (t < y if walk_forward else t != y)]


from research.legacy_backfill import reconstruct_season  # closed-research compatibility


def reconstruct_season_v8(y, terms=None, walk_forward=False, talent=None,
                          talent0=None, weight_years=WEIGHT_YEARS,
                          phase_years=PHASE_YEARS, report_years=REPORT_YEARS,
                          game_years=GAME_YEARS, margin=False, v10=False):
    """Reconstruct with the same v9 base/report arms as live scoring.
    Optional missing terms contribute zero and receive neutral-input tags;
    core missing inputs abstain. Fits exclude the test season (or use only
    earlier seasons with walk_forward). No carryover/v4 model is fitted.
    margin=False retains the archived v8 link for research only.
    v10=True (with margin): ORB% weights, drafted rookies valued by pick
    (`talent` / `terms` must be built with rookies), and the report route
    refitted before each slate on its training seasons plus season y's
    covered games before that slate, as the build does (pav.refit).
    """
    margin = margin or v10
    fit = ((lambda X, t, f: nc.fit_margin(X, t["margin"], f)) if margin else
           (lambda X, t, f: nc.fit_logit(X, t["win"], f)))
    tag, tag_av = ((build_site.MODEL_TAG_V10, build_site.MODEL_TAG_V10_AVAIL) if v10 else
                   (build_site.MODEL_TAG_V9, build_site.MODEL_TAG_V9_AVAIL) if margin
                   else (build_site.MODEL_TAG_V8, build_site.MODEL_TAG_V8_AVAIL))
    tag_n, tag_nav = ((build_site.MODEL_TAG_V10_NEUTRAL, build_site.MODEL_TAG_V10_NEUTRAL_AVAIL)
                      if v10 else (build_site.MODEL_TAG_V9_NEUTRAL,
                                   build_site.MODEL_TAG_V9_NEUTRAL_AVAIL))
    def use(years):
        out = training_years(years, y, walk_forward)
        if not out:
            raise SystemExit(f"season {y}: no training seasons in {list(years)}")
        return out
    memo, memo0 = dict(talent or {}), dict(talent0 or {})

    def tal(t):
        if t not in memo:
            try:
                memo[t] = pav.season_talent(t, rookies=True) if v10 else pav.season_talent(t)
            except Exception as e:  # noqa: BLE001 - rows fall back
                print(f"season {t}: no talent ({e!r})", flush=True)
                memo[t] = None
        return memo[t]

    def tal0(t):
        if t not in memo0:
            try:
                memo0[t] = pav.preseason_talent(t)
            except Exception as e:  # noqa: BLE001 - game 0 falls back
                print(f"season {t}: no preseason rosters ({e!r})", flush=True)
                memo0[t] = {}
        return memo0[t]

    weights = (nc.fit_weights(use(weight_years), orb_pct=True) if v10 else
               nc.fit_weights(use(weight_years)))
    phase_train = use(phase_years)
    games = {t: nc.fade_v8_games(t, weights, talent=tal(t), talent0=tal0(t))
             for t in set(phase_train) | {y}}
    feats, afeats = nc.V6_FEATURES, pav.FEATURES_V6
    train = pd.concat([games[t] for t in phase_train], ignore_index=True)
    ok = np.isfinite(train[feats].to_numpy(float)).all(axis=1)
    base = fit(train.loc[ok, feats].values, train.loc[ok], feats)
    test = games[y].copy()
    test["gmin"] = np.minimum(test["gp_home"], test["gp_away"])
    test["p_home"], test["model_tag"], test["route"] = np.nan, None, None
    test["margin_home"] = np.nan
    fin = np.isfinite(test[feats].to_numpy(float)).all(axis=1)
    test.loc[fin, "p_home"] = nc.predict(base, test.loc[fin, feats].values)
    test.loc[fin, "margin_home"] = nc.predict_margin(base, test.loc[fin, feats].values)
    test.loc[fin, "model_tag"], test.loc[fin, "route"] = tag, "base"
    ver = "v10" if v10 else "v9" if margin else "v8"
    print(f"season {y}: {ver} base on {int(fin.sum())}/{len(test)} games (fit n="
          f"{int(ok.sum())})", flush=True)
    tkey = pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d") + test["home"] + test["away"]
    if terms and y in terms:
        others = [t for t in training_years(report_years, y, walk_forward) if t in terms]
        if others:
            tr = pd.concat([pav.with_terms(games[t] if t in games else nc.fade_v8_games(
                t, weights, talent=tal(t), talent0=tal0(t)), terms[t]) for t in others])
            tr = tr[np.minimum(tr["gp_home"], tr["gp_away"]) >= 1]
            ok = np.isfinite(tr[afeats].to_numpy(float)).all(axis=1)
            am = fit(tr.loc[ok, afeats].values, tr.loc[ok], afeats)
            te = pav.with_terms(test, terms[y])
            te = te[te["gmin"] >= 1].copy()
            injury_missing = ~np.isfinite(te[["av_min", "av_bpm"]].to_numpy(float)).all(axis=1)
            te.loc[injury_missing, ["av_min", "av_bpm"]] = 0.0
            values, changed = nc.neutral_features(te, afeats)
            changed |= injury_missing
            usable = np.isfinite(values.to_numpy(float)).all(axis=1)
            te, values, changed = te[usable], values[usable], changed[usable]
            tk = te["slate_date"] + te["home"] + te["away"]
            if v10:
                pv, mv = slate_refits(am, tr.loc[ok], te, values)
            else:
                pv, mv = nc.predict(am, values.values), nc.predict_margin(am, values.values)
            pmap, mmap = dict(zip(tk, pv)), dict(zip(tk, mv))
            neutral_keys = set(tk[changed])
            hit = tkey.isin(pmap)
            test.loc[hit, "p_home"] = tkey[hit].map(pmap).to_numpy(float)
            test.loc[hit, "margin_home"] = tkey[hit].map(mmap).to_numpy(float)
            test.loc[hit, "model_tag"] = tag_av
            test.loc[hit, "route"] = "avail"
            neutral_hit = hit & tkey.isin(neutral_keys)
            if margin:
                test.loc[neutral_hit, "model_tag"] = tag_nav
            test.loc[neutral_hit, "route"] = "neutral"
            print(f"season {y}: {ver} availability on {int(hit.sum())}/{len(test)} games "
                  f"(fit on {others}, n={int(ok.sum())})", flush=True)
    # Fill only missing optional terms with neutral contributions, using the
    # fitted v9 model. No carryover/v4 fallback or extra model fit.
    values, changed = nc.neutral_features(test, feats)
    usable = np.isfinite(values.to_numpy(float)).all(axis=1)
    miss = test["p_home"].isna() & usable
    if miss.any():
        test.loc[miss, "p_home"] = nc.predict(base, values.loc[miss].values)
        test.loc[miss, "margin_home"] = nc.predict_margin(base, values.loc[miss].values)
        test.loc[miss, "model_tag"] = tag_n if margin else tag
        test.loc[miss, "route"] = "neutral"
    test = test[test["p_home"].notna()].reset_index(drop=True)
    print(f"season {y}: {ver} routes {test['route'].value_counts().to_dict()}", flush=True)
    return test, weights, base


def slate_refits(model, train, te, values):
    """v10's per-slate report route on season rows `te` (with their model
    inputs `values`): each slate scored by `model` refitted on `train` plus
    te's complete covered games before that slate (pav.refit), as the
    build does live. Returns (P, margin) arrays aligned with te."""
    feats = model["features"]
    train = train[feats + ["margin"]]
    dates = pd.to_datetime(te["date"]).to_numpy()
    done = np.isfinite(te[feats + ["margin"]].to_numpy(float)).all(axis=1)
    p, m = np.full(len(te), np.nan), np.full(len(te), np.nan)
    for d in np.unique(dates):
        idx = dates == d
        mod = pav.refit(model, te.loc[done & (dates < d), feats + ["margin"]], train)
        p[idx] = nc.predict(mod, values.values[idx])
        m[idx] = nc.predict_margin(mod, values.values[idx])
    return p, m


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
        model_tag=src["model_tag"].to_numpy(),
        margin_home=(np.round(src["margin_home"].astype(float).to_numpy(), 2)
                     if "margin_home" in src else np.nan))
    for c in MODEL_COLUMNS:
        out[c] = out[c].astype(object)
        out.loc[hit, c] = new[c]
    return out, int(hit.sum())


def attach_espn(test, sleep=0.25, basis="reconstructed"):
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
                basis=basis,
                home=r["home"], away=r["away"],
                gp_home=r.get("gp_home", np.nan), gp_away=r.get("gp_away", np.nan),
                home_b2b=r["h_b2b"], away_b2b=r["a_b2b"],
                delta=round(float(r["delta"]), 3), p_home=round(p, 5),
                lean=r["home"] if lean_home else r["away"],
                p_lean=round(p if lean_home else 1 - p, 5),
                margin_home=(round(float(r["margin_home"]), 2)
                             if pd.notna(r.get("margin_home", np.nan)) else np.nan),
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
    ap.add_argument("--v5", action="store_true",
                    help="score with the v5 routing (no ft_diff)")
    ap.add_argument("--v6", action="store_true",
                    help="score with the v6 routing (no talent on games 1-9)")
    ap.add_argument("--v7", action="store_true",
                    help="score with the v7 routing (games 1-9 on the early logit)")
    ap.add_argument("--v8", action="store_true",
                    help="score with v8 (the v9 routing fitted on the win, not the margin)")
    ap.add_argument("--v9", action="store_true",
                    help="score with v9 even when model/ holds v10 files")
    ap.add_argument("--rebuild", action="store_true",
                    help="the season in progress: score its played games after the fact "
                         "and write data/nba_rebuilt.csv (basis rebuilt), replacing it")
    a = ap.parse_args(argv)
    if a.rebuild and (a.rescore or min(a.seasons) <= max(PHASE_YEARS)):
        raise SystemExit("--rebuild takes only seasons in progress (after "
                         f"{max(PHASE_YEARS)}) and no --rescore")
    # v10 when the committed model files are v10 (the live model), unless an
    # earlier version is asked for
    older = a.v4 or a.v5 or a.v6 or a.v7 or a.v8 or a.v9
    try:
        v10 = not older and build_site.is_v10(nc.load_json("logit.json"))
    except (FileNotFoundError, ValueError):
        v10 = False
    terms = None
    if not a.no_avail:
        archive = pav.ReportArchive(os.path.join(a.cache, "injury_reports"))
        terms = {}
        try:
            # the season in progress (rebuilt from its played games) reads its
            # own reports too; it is never a training season
            for t in sorted(set(REPORT_YEARS) | {y for y in a.seasons if y > max(REPORT_YEARS)}):
                try:
                    terms[t], s = (pav.season_terms(t, archive, a.cache, rookies=True) if v10
                                   else pav.season_terms(t, archive, a.cache))
                except Exception as e:  # noqa: BLE001 - that season scores without the report
                    if t in REPORT_YEARS:
                        raise
                    print(f"report season {t}: no terms ({e!r})", flush=True)
                    continue
                print(f"report season {t}: {s['with_report']}/{s['games']} "
                      f"games with a report, {len(s['covered'])} covered",
                      flush=True)
        finally:
            archive.save()
    v8 = not (a.v4 or a.v5 or a.v6 or a.v7)        # v8 or v9: one faded rating
    margin = v8 and not a.v8

    def season(y):
        if v8:
            return reconstruct_season_v8(y, terms=terms, margin=margin, v10=v10)
        return reconstruct_season(y, terms=terms, v5=not a.v4, v6=not (a.v4 or a.v5),
                                  v7=not (a.v4 or a.v5 or a.v6))
    if a.rescore:
        df = ledger.load(ledger.RECON_PATH)
        for y in a.seasons:
            test, _, model = season(y)
            print(f"season {y}: {len(test)} games scored; logit {model}", flush=True)
            df, n = rescore(df, test)
            print(f"season {y}: rescored {n} reconstructed rows; tags "
                  f"{df['model_tag'].value_counts().to_dict()}", flush=True)
            if v8:
                # v8 scores game 0, which earlier versions skipped: add those
                # games (and any other game without a row) with their prices
                have = set(zip(df["slate_date"].astype(str), df["home"], df["away"]))
                key = zip(pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d"),
                          test["home"], test["away"])
                new = test[[k not in have for k in key]]
                if len(new):
                    rec = attach_espn(new)
                    df = pd.concat([df, rec], ignore_index=True)
                    df = df.drop_duplicates("game_id", keep="first")
                    print(f"season {y}: added {len(rec)} rows v7 had not scored "
                          f"(game 0 {int((rec[['gp_home', 'gp_away']].min(axis=1) == 0).sum())})",
                          flush=True)
        ledger.save(df, ledger.RECON_PATH)
        print(f"wrote {ledger.RECON_PATH}: {len(df)} rows", flush=True)
        return 0
    path = ledger.REBUILT_PATH if a.rebuild else ledger.RECON_PATH
    # a rebuild replaces the file: every played game is rescored from scratch
    out = [] if a.rebuild else [ledger.load(ledger.RECON_PATH)]
    for y in a.seasons:
        test, _, model = season(y)
        print(f"season {y}: {len(test)} games scored; logit {model}", flush=True)
        rec = attach_espn(test, basis="rebuilt" if a.rebuild else "reconstructed")
        if a.rebuild:                        # graded games only
            rec = rec[pd.to_numeric(rec["home_won"], errors="coerce").isin([0, 1])]
        by_book = rec["close_book"].value_counts().to_dict()
        print(f"season {y}: {len(rec)} matched to ESPN; closes by book {by_book}",
              flush=True)
        out.append(rec)
    df = pd.concat([d for d in out if len(d)], ignore_index=True) if any(
        len(d) for d in out) else ledger.empty()
    df = df.drop_duplicates("game_id", keep="last")
    ledger.save(df, path)
    print(f"wrote {path}: {len(df)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
