#!/usr/bin/env python3
"""Dynamic chronological fit: the v9 base margin model refitted before every
slate on every game played so far, beside the season-start walk-forward fit.

  python research/dynamic_fit.py [--seasons 2017 ... 2027]

Owner's question (2026-10-27): the walk-forward checks (research/
margin_model.py, research/walk_forward.py) fit once per test season on the
seasons before it. Would a fit that keeps learning through the season --
including the test season's own earlier games -- do better, and does it
change the gap to the market or the home lean?

Arms, all OLS on the final home margin with v9's base terms
(nba_composite.V6_FEATURES on the faded rating, fade_v8_games), P =
Phi(m / sigma), sigma the fit's residual SD. Fixed before the run:

  S   static walk-forward: one fit on the game seasons before Y (the
      protocol of research/margin_model.py; for 2027 it is the shipped base
      fit's seasons).
  D   dynamic, expanding: refitted before each slate date d of season Y on
      the seasons before Y plus season Y's games strictly before d, equal
      weights.
  Dw  dynamic, recency-weighted: as D, weighted least squares with weight
      0.5 ** (age in days / 365) at d (a game a season ago counts half).

Composite weights are fitted once per fold on WEIGHT_YEARS before Y (team
season tables exist only for completed seasons). Every input uses games
strictly before each date (fade_v8_games). Fits use complete rows, as the
shipped fit does; predictions set a missing optional term to zero
(nc.neutral_features) and skip rows missing a core rating.

Injury-report arm (--avail, added 2026-10-27 on the owner's request): the
same three arms on v9's availability route (player_availability.FEATURES_V6,
av_min / av_bpm from the last NBA injury report at least LEAD_MINUTES
before tip, season_terms), fitted on the covered games 1+ of the report
seasons before Y (S) plus season Y's covered games before each slate (D,
Dw), as the shipped fit uses covered games 1+ of its report seasons. Folds
2024-2027 (2023 has no earlier report season); scored on covered games.

Readings: D - S and Dw - S log loss, Brier, margin MSE and bias against
outcomes, by season and stage (game 0, games 1-9, 10+), pooled over seasons
as a description only. Against the market on the reconstructed rows (each
book x season alone) and on the 2026-27 native rows (DraftKings, apart):
model - close log loss and the model's home lean beyond the close. The
intercept path of D and Dw through each season. Research only: nothing here
touches the ledgers, the pages or model/*.json; any production use needs a
new model tag. ± is 95%, game-level, paired.
"""
from __future__ import annotations

import argparse
import os
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import backfill_history as bf                     # noqa: E402
import nba_composite as nc                        # noqa: E402
import player_availability as pav                 # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "dynamic_fit")
FEATS = list(nc.V6_FEATURES)
CORE = ["delta", "b2b_net", "d_phase"]
GAME_YEARS = list(bf.PHASE_YEARS) + [2027]       # 2016-19, 2021-26, this season
SEASONS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
HALF_LIFE_DAYS = 365.0
Z95 = 1.96
ARMS = ("S", "D", "Dw")
AVAIL_FEATS = list(pav.FEATURES_V6)
REPORT_YEARS = [2023, 2024, 2025, 2026, 2027]
AVAIL_SEASONS = [2024, 2025, 2026, 2027]
_PHI = np.vectorize(NormalDist().cdf, otypes=[float])


def log(msg):
    print(msg, flush=True)


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan"), len(d)
    return float(d.mean()), float(Z95 * d.std(ddof=1) / np.sqrt(len(d))), len(d)


def fmt(m, h, n):
    return f"{m:+.4f} ± {h:.4f} (n {n})" if np.isfinite(m) else "--"


# ------------------------------------------------------------------ data --
_talent = {}


def talents(y):
    if y not in _talent:
        _talent[y] = (pav.season_talent(y), pav.preseason_talent(y))
    return _talent[y]


def games(y, weights):
    """Season y's games with v9's base terms under one fold's weights."""
    tal, tal0 = talents(y)
    g = nc.fade_v8_games(y, weights, talent=tal, talent0=tal0)
    g["date"] = pd.to_datetime(g["date"])
    g["slate_date"] = g["date"].dt.strftime("%Y-%m-%d")
    return g


def wls(df, w=None, feats=FEATS):
    """(intercept, coef, sigma) of margin on feats, complete rows."""
    X = np.column_stack([np.ones(len(df)), df[feats].to_numpy(float)])
    y = df["margin"].to_numpy(float)
    w = np.ones(len(df)) if w is None else np.asarray(w, float)
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r = y - X @ b
    # weighted residual SD with Kish's effective n for the degrees of freedom
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    sigma = float(np.sqrt((w * r ** 2).sum() / w.sum() * n_eff / max(1.0, n_eff - X.shape[1])))
    return b, sigma


def predict(b, sigma, test, feats=FEATS):
    X, _ = nc.neutral_features(test, feats)
    m = b[0] + X.to_numpy(float) @ b[1:]
    return m, _PHI(m / sigma)


def arms(past, cur, feats, Y, label):
    """S / D / Dw predictions on cur (season Y) from past (earlier seasons)."""
    past = past[np.isfinite(past[feats + ["margin"]].to_numpy(float)).all(axis=1)]
    cur = cur[np.isfinite(cur[CORE].to_numpy(float)).all(axis=1)].copy()
    cur_ok = np.isfinite(cur[feats + ["margin"]].to_numpy(float)).all(axis=1)
    bS, sS = wls(past, feats=feats)
    cur["m_S"], cur["p_S"] = predict(bS, sS, cur, feats)
    path = []
    for d, idx in cur.groupby("date").groups.items():
        seen = cur[cur_ok & (cur["date"] < d)]
        tr = pd.concat([past, seen], ignore_index=True)
        bD, sD = wls(tr, feats=feats) if len(seen) else (bS, sS)
        age = (d - tr["date"]).dt.days.to_numpy(float)
        bW, sW = wls(tr, 0.5 ** (age / HALF_LIFE_DAYS), feats)
        t = cur.loc[idx]
        cur.loc[idx, "m_D"], cur.loc[idx, "p_D"] = predict(bD, sD, t, feats)
        cur.loc[idx, "m_Dw"], cur.loc[idx, "p_Dw"] = predict(bW, sW, t, feats)
        path.append(dict(date=d, n_seen=len(seen), int_D=bD[0], int_Dw=bW[0],
                         sigma_D=sD, sigma_Dw=sW, **{f"Dw_{f}": c for f, c in zip(feats, bW[1:])}))
    mn = np.fmin(cur["gp_home"], cur["gp_away"])
    cur["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
    cur["season"] = Y
    log(f"{label} fold {Y}: {len(cur)} games, train {len(past)} past rows, S intercept "
        f"{bS[0]:+.2f} sigma {sS:.2f}; D end intercept {path[-1]['int_D']:+.2f}, "
        f"Dw end {path[-1]['int_Dw']:+.2f}")
    return cur, pd.DataFrame(path).assign(season=Y), dict(zip(["intercept"] + feats, bS), sigma=sS)


def fold(Y):
    """Per-game S / D / Dw predictions for season Y and the intercept paths."""
    wts = fold_weights(Y)
    past = pd.concat([games(t, wts) for t in GAME_YEARS if t < Y], ignore_index=True)
    return arms(past, games(Y, wts), FEATS, Y, "base")


_weights = {}


def fold_weights(Y):
    if Y not in _weights:
        _weights[Y] = nc.fit_weights(bf.training_years(bf.WEIGHT_YEARS, Y, True))
    return _weights[Y]


def report_terms(t):
    """Season t's covered-game availability terms (season_terms), cached."""
    path = os.path.join(pav.DEFAULT_CACHE, f"dyn_terms_{t}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    terms, _ = pav.season_terms(t, pav.ReportArchive(pav.REPORT_DIR))
    terms.to_csv(path, index=False)
    return terms


def covered_games(t, wts):
    """Season t's covered games 1+ with the availability terms."""
    g = games(t, wts).merge(report_terms(t), on=["slate_date", "home", "away"], how="inner")
    return g[np.minimum(g["gp_home"], g["gp_away"]) >= 1]


def avail_fold(Y):
    wts = fold_weights(Y)
    past = pd.concat([covered_games(t, wts) for t in REPORT_YEARS if t < Y], ignore_index=True)
    return arms(past, covered_games(Y, wts), AVAIL_FEATS, Y, "avail")


# ---------------------------------------------------------------- market --
def market_rows():
    """Close (and open) moneyline q and close spread by book x season, with
    the basis kept: reconstructed hindsight rows and 2026-27 native rows."""
    out = []
    for path, basis in (("data/nba_reconstructed.csv", "reconstructed"),
                        ("data/nba_ledger.csv", "native")):
        d = pd.read_csv(os.path.join(ROOT, path))
        if basis == "native":
            d = d[d["season"] == 2027]
        d = d[d["close_q_home"].notna()]
        out.append(d[["season", "slate_date", "home", "away", "close_book",
                      "close_q_home", "close_spread"]].assign(basis=basis))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- report --
def scores(d, arm):
    y = (d["margin"] > 0).to_numpy(float)
    p, m = d[f"p_{arm}"].to_numpy(float), d[f"m_{arm}"].to_numpy(float)
    return ll(p, y), (p - y) ** 2, (d["margin"].to_numpy(float) - m) ** 2


def outcome_block(P):
    L = ["== 1. Against outcomes (arm - S, negative = arm better; ± 95%)"]
    for label, part in [("all seasons (description; seasons differ)", P)] + \
            [(f"season {y}", g) for y, g in P.groupby("season")]:
        L.append(f"  {label}")
        for st, g in [("games 1+", part[part["stage"] != "game 0"])] + list(part.groupby("stage")):
            if not len(g):
                continue
            base = scores(g, "S")
            cells = []
            for arm in ("D", "Dw"):
                a = scores(g, arm)
                cells.append(f"{arm}-S logloss {fmt(*ci(a[0] - base[0]))}  "
                             f"MSE {ci(a[2] - base[2])[0]:+.2f} ± {ci(a[2] - base[2])[1]:.2f}")
            L.append(f"    {st:10s} S logloss {base[0].mean():.4f}  " + "  |  ".join(cells))
    L.append("")
    L.append("  Home lean, predicted minus actual home margin, by season (games 1+; ± 95%):")
    for y, g in P[P["stage"] != "game 0"].groupby("season"):
        L.append(f"    {y}  " + "  ".join(
            f"{arm} {ci(g[f'm_{arm}'] - g['margin'])[0]:+.2f} ± {ci(g[f'm_{arm}'] - g['margin'])[1]:.2f}"
            for arm in ARMS) + f"  (n {len(g)})")
    return L


def market_block(P, M):
    L = ["", "== 2. Against the close, each basis x book x season alone (± 95%)",
         "   logloss arm - close (negative = arm better); lean = arm margin + close home line"]
    J = P.merge(M, on=["season", "slate_date", "home", "away"], how="inner")
    for (basis, book, y), g in J.groupby(["basis", "close_book", "season"]):
        yy = (g["margin"] > 0).to_numpy(float)
        lc = ll(g["close_q_home"].to_numpy(float), yy)
        for st, h in [("games 1+", g[g["stage"] != "game 0"])] + list(g.groupby("stage")):
            if len(h) < 15:
                continue
            hy = (h["margin"] > 0).to_numpy(float)
            hc = ll(h["close_q_home"].to_numpy(float), hy)
            sp = h["close_spread"].to_numpy(float)
            cells = []
            for arm in ARMS:
                la = ll(h[f"p_{arm}"].to_numpy(float), hy)
                lean = ci(h[f"m_{arm}"].to_numpy(float) + sp)
                cells.append(f"{arm} {ci(la - hc)[0]:+.4f} ± {ci(la - hc)[1]:.4f} "
                             f"lean {lean[0]:+.2f}")
            L.append(f"  {basis:13s} {book:7s} {y} {st:10s} n {len(h):4d}  " + "  |  ".join(cells))
        del lc
    return L


def path_block(paths):
    L = ["", "== 3. Intercept (home edge, pts) of D and Dw at points in each season"]
    for y, g in paths.groupby("season"):
        g = g.sort_values("date").reset_index(drop=True)
        pick = sorted(set([0, len(g) // 4, len(g) // 2, 3 * len(g) // 4, len(g) - 1]))
        L.append(f"  {y}  " + "  ".join(
            f"{g.loc[i, 'date']:%m-%d} D {g.loc[i, 'int_D']:+.2f} Dw {g.loc[i, 'int_Dw']:+.2f}"
            for i in pick))
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=None)
    ap.add_argument("--avail", action="store_true",
                    help="the injury-report route (writes the *_avail outputs)")
    a = ap.parse_args()
    run, seasons, tag = ((avail_fold, AVAIL_SEASONS, "_avail") if a.avail
                         else (fold, SEASONS, ""))
    preds, paths, statics = [], [], {}
    for Y in a.seasons or seasons:
        p, pa, s = run(Y)
        preds.append(p)
        paths.append(pa)
        statics[Y] = s
    P, paths = pd.concat(preds, ignore_index=True), pd.concat(paths, ignore_index=True)
    route = ("injury-report route: covered games 1+ of the report seasons"
             if a.avail else "base route: every game")
    head = [__doc__.strip().splitlines()[0], f"Route: {route}.", "",
            "Static fits (S) by fold: " + "; ".join(
                f"{y} int {s['intercept']:+.2f} sigma {s['sigma']:.2f}" for y, s in statics.items()),
            ""]
    L = head + outcome_block(P) + market_block(P, market_rows()) + path_block(paths)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"RESULTS{tag}.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    P.drop(columns=["date"]).to_csv(os.path.join(OUT, f"predictions{tag}.csv"), index=False,
                                    float_format="%.5g")
    paths.to_csv(os.path.join(OUT, f"paths{tag}.csv"), index=False, float_format="%.5g")
    print("\n".join(L))


if __name__ == "__main__":
    main()
