#!/usr/bin/env python3
"""Games 1-9 with roster information (owner's request, 2026-10-22).

  python research/early_roster.py [--seasons 2017-2019 2021-2026]

The early route (cold_start.py, games 1-9) sees no players: its rating is
last season's log (weight 0.25) then this season's games, so trades,
signings, rookies and long injuries enter only through results, while the
market prices them from opening night. From game 1 on, every team has a box
score, so the games-10+ roster term can be computed for these games too.

Arms, one early logit per test season, fitted on earlier seasons' games with
1 <= min(games played) < cold_start.WINDOW (20), both arms on the same rows:
  early      sigma(a + b delta + c b2b_net): production's features (carryover
             delta, rho 0.25)
  early_tal  + talent_diff: player_availability.talent_fn unchanged (the
             previous box score's players x minutes role, this season's mean
             before the date else last season's MP/G, x last-season BPM
             above replacement), home - away
Composite weights walk-forward (backfill_history.WEIGHT_YEARS before the test
season). Scored on games 1-9 (1 <= min(gp) <= 9); game 0 still abstains.

  1. Against outcomes, every test season and pooled: log loss early_tal -
     early (paired), the talent coefficient, calibration slope.
  2. Against the close on the priced seasons' reconstructed rows (same games,
     each book x season alone).
  3. Roster turnover: each team's share of last season's minutes played by
     players on its first box score this season. Does the close's
     correction of the early model (logit q - logit P) follow talent_diff and
     the turnover gap? The early model's gap to the close, and early_tal's
     gain on outcomes, by the lower of the two teams' shares.
  4. H1's hindsight bets (the reconstructed P, value side with P - q >= 0.08
     at the close): split by whether talent_diff favours the bet side; ROI at
     the close beside its null. Descriptive; the rule stays frozen.
Research only: nothing here changes the model, the ledger or MODEL_TAG.
Writes diagnostics/early_roster/RESULTS.txt. ± is 95%.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import backfill_history as bf  # noqa: E402
import cold_start  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import model_review as mr  # noqa: E402
import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "early_roster")
BOX_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
ARMS = {"early": ["delta", "b2b_net"], "early_tal": ["delta", "b2b_net", "talent_diff"]}
H1_EDGE = 0.08
_LOGS = {}


def logs(y):
    if y not in _LOGS:
        _LOGS[y] = nc.load_logs(y)
    return _LOGS[y]


def season_rows(y, weights, talent, window):
    """Early rows of season y (1 <= min gp < window) with talent_diff."""
    e = cold_start.early_games(y, weights, logs=logs(y), prior_logs=logs(y - 1),
                               min_gp=1, window=window)
    if not len(e):
        return e
    e["talent_diff"] = [talent(h, d) - talent(a, d) if talent else np.nan
                        for h, a, d in zip(e["home"], e["away"], e["date"])]
    return e


def returning_share(prev, cur):
    """{team: share of its previous-season minutes (box `prev`) played by
    players on its first box score of this season (`cur`; listed, played or
    not)}."""
    out = {}
    for tm, g in cur.groupby("team"):
        first = g[g["date"] == g["date"].min()]
        p = prev[(prev["team"] == tm) & (prev["minutes"] > 0)]
        total = p["minutes"].sum()
        if total > 0:
            out[tm] = float(p[p["player_id"].isin(set(first["player_id"]))]["minutes"].sum()
                            / total)
    return out


def fit_predict(train, test, feats):
    m = nc.fit_logit(train[feats].to_numpy(float), train["win"], feats)
    return nc.predict(m, test[feats].to_numpy(float)), m


def slope(p, y):
    s, _ = mr.slope(np.asarray(p, float), np.asarray(y, float))
    return s


def build(seasons, cache_dir=pav.DEFAULT_CACHE):
    """Walk-forward predictions for every test season's games 1-9."""
    talents, shares, out = {}, {}, []
    for t in BOX_YEARS:
        if pav.box_file(t, cache_dir):
            talents[t] = pav.season_talent(t, cache_dir)
            if t - 1 in BOX_YEARS and pav.box_file(t - 1, cache_dir):
                shares[t] = returning_share(pav.fetch_box(t - 1, cache_dir=cache_dir),
                                            pav.fetch_box(t, cache_dir=cache_dir))
    for y in seasons:
        train_years = [t for t in BOX_YEARS if t < y and t in talents]
        if not train_years or y not in talents:
            print(f"season {y}: no training box seasons; skipped", flush=True)
            continue
        weights = nc.fit_weights(bf.training_years(bf.WEIGHT_YEARS, y, walk_forward=True))
        tr = pd.concat([season_rows(t, weights, talents[t], cold_start.WINDOW)
                        for t in train_years], ignore_index=True)
        te = season_rows(y, weights, talents[y], nc.MIN_GAMES)
        tr, te = tr.dropna(subset=["talent_diff"]), te.dropna(subset=["talent_diff"])
        for arm, feats in ARMS.items():
            te[f"p_{arm}"], m = fit_predict(tr, te, feats)
            te[f"coef_{arm}"] = m["coef"][-1]
        sh = shares.get(y, {})
        te["share_home"] = te["home"].map(sh)
        te["share_away"] = te["away"].map(sh)
        te["n_train"] = len(tr)
        print(f"season {y}: train {train_years} (n={len(tr)}), test games 1-9 n={len(te)}",
              flush=True)
        out.append(te)
    return pd.concat(out, ignore_index=True)


def outcome_lines(d):
    L = ["== 1. Against outcomes, games 1-9, walk-forward (log loss; early_tal - early "
         "paired, ± 95%)"]
    for y, x in d.groupby("year"):
        yv = x["win"].to_numpy(float)
        m, ci = mr.mean_ci(mr.logloss(x["p_early_tal"], yv) - mr.logloss(x["p_early"], yv))
        L.append(f"    {mr.season_name(y)}  n {len(x):>4}  early {mr.logloss(x['p_early'], yv).mean():.4f}"
                 f"  early_tal {mr.logloss(x['p_early_tal'], yv).mean():.4f}  diff {m:+.4f} ± {ci:.4f}"
                 f"  talent coef {x['coef_early_tal'].iloc[0]:+.4f} (train n {x['n_train'].iloc[0]})")
    yv = d["win"].to_numpy(float)
    m, ci = mr.mean_ci(mr.logloss(d["p_early_tal"], yv) - mr.logloss(d["p_early"], yv))
    b, bci = mr.mean_ci((d["p_early_tal"] - yv) ** 2 - (d["p_early"] - yv) ** 2)
    better = sum(mr.logloss(x["p_early_tal"], x["win"].to_numpy(float)).mean()
                 < mr.logloss(x["p_early"], x["win"].to_numpy(float)).mean()
                 for _, x in d.groupby("year"))
    L.append(f"    pooled   n {len(d):>4}  diff {m:+.4f} ± {ci:.4f}  (Brier {b:+.4f} ± {bci:.4f});"
             f" better in {better} of {d['year'].nunique()} seasons; slope early "
             f"{slope(d['p_early'], yv):.2f}, early_tal {slope(d['p_early_tal'], yv):.2f}")
    return L


def priced(d):
    """Test rows joined to the graded reconstructed rows (close, prices, P)."""
    g = mr.frame(ledger.load(os.path.join(ROOT, ledger.RECON_PATH)))
    g = g[g["early"]].copy()
    d = d.assign(slate_date=pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"))
    return d.merge(g[["slate_date", "home", "away", "season", "close_book", "q_close",
                      "p", "close_home_ml", "close_away_ml"]],
                   on=["slate_date", "home", "away"], how="inner")


def close_lines(m):
    L = ["", "== 2. Against the close, games 1-9 (same games, each book x season alone)"]
    for (season, book), x in m.groupby(["season", "close_book"]):
        yv = x["win"].to_numpy(float)
        q = x["q_close"].to_numpy(float)
        parts = []
        for arm in ARMS:
            a, ci = mr.mean_ci(mr.logloss(x[f"p_{arm}"], yv) - mr.logloss(q, yv))
            parts.append(f"{arm} - close {a:+.4f} ± {ci:.4f}")
        t, tci = mr.mean_ci(mr.logloss(x["p_early_tal"], yv) - mr.logloss(x["p_early"], yv))
        L.append(f"  {mr.BOOK_NAMES.get(book, book)} {mr.season_name(season)} (n {len(x)}): "
                 + "; ".join(parts) + f"; early_tal - early {t:+.4f} ± {tci:.4f}")
    return L


def turnover_lines(d, m):
    L = ["", "== 3. Roster turnover (share of last season's team minutes on the first "
         "box score)"]
    sh = pd.concat([d[["year", "home", "share_home"]].rename(columns={"home": "team", "share_home": "s"}),
                    d[["year", "away", "share_away"]].rename(columns={"away": "team", "share_away": "s"})]
                   ).drop_duplicates(["year", "team"]).dropna()
    L.append(f"  team-seasons {len(sh)}: returning share median {sh['s'].median():.2f}, "
             f"10th pct {sh['s'].quantile(0.1):.2f}, 90th {sh['s'].quantile(0.9):.2f}")
    x = m.dropna(subset=["share_home", "share_away"]).copy()
    if len(x) > 30:
        x["corr_q"] = mr.logit(x["q_close"]) - mr.logit(x["p_early"])
        x["turn_gap"] = x["share_home"] - x["share_away"]
        for c in ("talent_diff", "turn_gap"):
            r = np.corrcoef(x[c], x["corr_q"])[0, 1]
            L.append(f"  priced games 1-9 (n {len(x)}): corr(market correction of the early "
                     f"model, {c}) {r:+.3f} ± {1.96 / np.sqrt(len(x)):.3f}")
        X1 = np.column_stack([np.ones(len(x)), x[["talent_diff", "turn_gap"]].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X1, x["corr_q"].to_numpy(float), rcond=None)
        L.append(f"  correction ~ talent_diff + turn_gap (OLS, logit): talent {beta[1]:+.4f}, "
                 f"turnover gap {beta[2]:+.3f}")
        x["low"] = x[["share_home", "share_away"]].min(axis=1)
        x["tercile"] = pd.qcut(x["low"], 3, labels=["most turnover", "middle", "least turnover"])
        for lab, z in x.groupby("tercile", observed=True):
            yv = z["win"].to_numpy(float)
            a, ci = mr.mean_ci(mr.logloss(z["p_early"], yv) - mr.logloss(z["q_close"], yv))
            t, tci = mr.mean_ci(mr.logloss(z["p_early_tal"], yv) - mr.logloss(z["p_early"], yv))
            L.append(f"    {lab:<15} (lower share {z['low'].min():.2f}-{z['low'].max():.2f}, n {len(z)}): "
                     f"early - close {a:+.4f} ± {ci:.4f}; early_tal - early {t:+.4f} ± {tci:.4f}")
    y = d.dropna(subset=["share_home", "share_away"]).copy()
    y["low"] = y[["share_home", "share_away"]].min(axis=1)
    y["tercile"] = pd.qcut(y["low"], 3, labels=["most turnover", "middle", "least turnover"])
    L.append(f"  every test season, against outcomes (n {len(y)}): early_tal - early by tercile")
    for lab, z in y.groupby("tercile", observed=True):
        yv = z["win"].to_numpy(float)
        t, tci = mr.mean_ci(mr.logloss(z["p_early_tal"], yv) - mr.logloss(z["p_early"], yv))
        L.append(f"    {lab:<15} (lower share {z['low'].min():.2f}-{z['low'].max():.2f}, n {len(z)}): "
                 f"{t:+.4f} ± {tci:.4f}")
    return L


def h1_lines(m):
    L = ["", f"== 4. H1's hindsight bets (reconstructed P, value side with P - q >= "
         f"{H1_EDGE:.2f} at the close), split by talent_diff"]
    x = m.copy()
    home = (x["p"] - x["q_close"]) >= H1_EDGE
    away = ((1 - x["p"]) - (1 - x["q_close"])) >= H1_EDGE
    x = x[home | away].assign(bet_home=home[home | away])
    if not len(x):
        return L + ["  no bets"]
    x["agree"] = np.where(x["bet_home"], x["talent_diff"] > 0, x["talent_diff"] < 0)
    for lab, z in (("all", x), ("talent backs the bet", x[x["agree"]]),
                   ("talent against the bet", x[~x["agree"]])):
        if not len(z):
            continue
        ml = np.where(z["bet_home"], z["close_home_ml"], z["close_away_ml"]).astype(float)
        q = np.where(z["bet_home"], z["q_close"], 1 - z["q_close"])
        won = np.where(z["bet_home"], z["win"], 1 - z["win"]).astype(bool)
        pl = np.array([market.unit_profit(a, w) for a, w in zip(ml, won)])
        dec = np.array([market.decimal_payout(a) for a in ml])
        ok = np.isfinite(pl) & np.isfinite(dec)
        roi, ci = mr.mean_ci(pl[ok])
        L.append(f"  {lab:<24} n {ok.sum():>3}  won {100 * won[ok].mean():5.1f}%  q "
                 f"{100 * q[ok].mean():5.1f}%  ROI {100 * roi:+6.1f}% ± {100 * ci:4.1f}  "
                 f"null {100 * float((q[ok] * dec[ok] - 1).mean()):+5.1f}%")
    return L


def report(d):
    L = ["Games 1-9 with roster information: the early logit + talent_diff (walk-forward)",
         "research/early_roster.py; ± is 95%; game 0 abstains in every arm.", ""]
    L += outcome_lines(d)
    m = priced(d)
    if len(m):
        L += close_lines(m)
    L += turnover_lines(d, m)
    if len(m):
        L += h1_lines(m)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=["2017-2019", "2021-2026"])
    a = ap.parse_args(argv)
    d = build(nc.parse_years(a.seasons))
    text = report(d)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
