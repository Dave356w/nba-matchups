#!/usr/bin/env python3
"""ORB% instead of ORB per 100: v9's base margin model, walk-forward, two arms.

  python research/orb_pct.py --run [--seasons 2017 ... 2027]
  python research/orb_pct.py --report

Owner's question (2026-10-28): the composite's rebounding factors are
offensive rebounds per 100 possessions (own, and opponents' allowed),
not the textbook ORB% = ORB / (ORB + opponent DRB). Per-100 ORB partly
rewards missing shots (more misses, more chances), which overlaps eFG%.

Arms, fixed before the run:
  per100  production: off ORB/100, -opp ORB/100.
  pct     off ORB% = ORB / (ORB + opp DRB), -opp ORB% = opp ORB / (opp ORB
          + DRB), both x 100. Nothing else changes.
The arm replaces the rebounding columns everywhere the composite is built:
the team-season weight fit (rates_from_per100), the decayed game features
(features_from_totals) and therefore opponents' 3P luck, which reads the
composite. Each fold Y refits the weights on WEIGHT_YEARS before Y and v9's
base OLS on the game seasons before Y (research/half_life_sweep.py's fold),
then scores season Y. Base route only; talent and FT% are unchanged.

Readings: log loss, Brier and margin MSE, pct - per100 on the same games
(negative = pct better), by stage and season; and against the close, each
basis x book x season alone. Decision rule, set before the run: per100
stays unless pct is better on games 1+ beyond the 95% interval AND in at
least 7 of the 10 folds; even then it is a candidate for a new model tag,
the owner's decision. Research only: nothing here touches the ledgers, the
pages or model/*.json. ± is 95%, game-level, paired.
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
import nba_composite as nc                        # noqa: E402
import dynamic_fit as dyn                         # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "orb_pct")
ARMS = ("per100", "pct")
BASE = "per100"
NAMES = ["off eFG", "off TOV", "off ORB", "off FTr",
         "def eFG", "def TOV", "def ORB", "def FTr"]

_features, _rates = nc.features_from_totals, nc.rates_from_per100


def features_pct(t, orb_pct=True):
    """features_from_totals with ORB% in place of ORB per 100."""
    f = _features(t)
    T = dict(zip(nc.STATS, t[:8]))
    O = dict(zip(nc.STATS, t[8:]))
    f[2] = T["ORB"] / (T["ORB"] + O["DRB"]) * 100
    f[6] = -O["ORB"] / (O["ORB"] + T["DRB"]) * 100
    return f


def rates_pct(o, p, orb_pct=True):
    """rates_from_per100 with ORB% (per-100 counts share a denominator)."""
    X = _rates(o, p)
    X[:, 2] = (o["ORB"] / (o["ORB"] + p["DRB"]) * 100).to_numpy()
    X[:, 6] = -(p["ORB"] / (p["ORB"] + o["DRB"]) * 100).to_numpy()
    return X


def use(arm):
    nc.features_from_totals = features_pct if arm == "pct" else _features
    nc.rates_from_per100 = rates_pct if arm == "pct" else _rates
    dyn._weights.clear()                           # weights depend on the arm


def fold(Y, arm):
    wts = dyn.fold_weights(Y)

    def games(t):
        g = dyn.games(t, wts)
        return g

    past = pd.concat([games(t) for t in dyn.GAME_YEARS if t < Y], ignore_index=True)
    past = past[np.isfinite(past[dyn.FEATS + ["margin"]].to_numpy(float)).all(axis=1)]
    cur = games(Y)
    cur = cur[np.isfinite(cur[dyn.CORE].to_numpy(float)).all(axis=1)].copy()
    b, s = dyn.wls(past)
    cur["m"], cur["p"] = dyn.predict(b, s, cur)
    mn = np.fmin(cur["gp_home"], cur["gp_away"])
    cur["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
    cur["season"], cur["arm"] = Y, arm
    share = np.abs(np.array(wts["w"])) / np.abs(np.array(wts["w"])).sum()
    dyn.log(f"{arm} fold {Y}: {len(cur)} games, sigma {s:.2f}, ORB share "
            f"{share[2] + share[6]:.3f}, R2 {wts['r2_in_sample']:.4f}")
    return cur[["season", "slate_date", "home", "away", "gp_home", "gp_away", "stage",
                "margin", "m", "p", "arm"]], wts


def run(seasons):
    os.makedirs(OUT, exist_ok=True)
    wrows = []
    for arm in ARMS:
        use(arm)
        parts = []
        for Y in seasons:
            d, w = fold(Y, arm)
            parts.append(d)
            wrows.append(dict(arm=arm, fold=Y, r2=w["r2_in_sample"],
                              **{n: x for n, x in zip(NAMES, w["w"])},
                              **{f"sd {n}": x for n, x in zip(NAMES, w["sd"])}))
        pd.concat(parts).to_csv(os.path.join(OUT, f"predictions_{arm}.csv"), index=False)
    use("per100")
    pd.DataFrame(wrows).to_csv(os.path.join(OUT, "weights.csv"), index=False)


def report():
    P = pd.concat([pd.read_csv(os.path.join(OUT, f"predictions_{a}.csv")) for a in ARMS],
                  ignore_index=True)
    key = ["season", "slate_date", "home", "away"]
    W = P.pivot_table(index=key + ["stage", "margin"], columns="arm", values=["m", "p"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index().dropna()
    L = [__doc__.strip().splitlines()[0], "",
         f"Games scored by both arms: {len(W)}. Each row: pct - per100 "
         "(negative = pct better), ± 95%.", ""]

    def cell(g):
        y = (g["margin"] > 0).to_numpy(float)
        d = dyn.ll(g["p_pct"], y) - dyn.ll(g["p_per100"], y)
        b = (g["p_pct"] - y) ** 2 - (g["p_per100"] - y) ** 2
        mse = (g["margin"] - g["m_pct"]) ** 2 - (g["margin"] - g["m_per100"]) ** 2
        return dyn.ci(d), dyn.ci(b), dyn.ci(mse)

    L.append("== 1. Against outcomes (log loss, Brier, margin MSE)")
    for st, g in [("games 1+", W[W["stage"] != "game 0"])] + list(W.groupby("stage")):
        (m, e, _), (bm, be, _), (mm, me, _) = cell(g)
        wins = sum(cell(s)[0][0] < 0 for _, s in g.groupby("season"))
        y = (g["margin"] > 0).to_numpy(float)
        L.append(f"  {st:10s} n {len(g):5d}  per100 log loss {dyn.ll(g['p_per100'], y).mean():.4f}"
                 f"  pct - per100 {m:+.4f} ± {e:.4f}  Brier {bm:+.4f} ± {be:.4f}"
                 f"  MSE {mm:+.2f} ± {me:.2f}  better in {wins} of {g['season'].nunique()}")
    L += ["", "  By season, games 1+:"]
    for s, g in W[W["stage"] != "game 0"].groupby("season"):
        (m, e, _), _, (mm, me, _) = cell(g)
        L.append(f"    {s}  n {len(g):5d}  log loss {m:+.4f} ± {e:.4f}  MSE {mm:+.2f} ± {me:.2f}")

    M = dyn.market_rows()
    J = W.merge(M, on=key, how="inner")
    L += ["", "== 2. Against the close, each basis x book x season alone (log loss arm - close, ± 95%)"]
    for (basis, book, s), g in J.groupby(["basis", "close_book", "season"]):
        for st, h in [("games 1+", g[g["stage"] != "game 0"])] + list(g.groupby("stage")):
            if len(h) < 100:
                continue
            y = (h["margin"] > 0).to_numpy(float)
            lc = dyn.ll(h["close_q_home"].to_numpy(float), y)
            a = [dyn.ci(dyn.ll(h[f"p_{arm}"], y) - lc) for arm in ARMS]
            d = dyn.ci(dyn.ll(h["p_pct"], y) - dyn.ll(h["p_per100"], y))
            L.append(f"  {basis:13s} {book:7s} {s} {st:10s} n {len(h):4d}  "
                     f"per100 {a[0][0]:+.4f} ± {a[0][1]:.4f}  pct {a[1][0]:+.4f} ± {a[1][1]:.4f}"
                     f"  pct - per100 {d[0]:+.4f} ± {d[1]:.4f}")

    w = pd.read_csv(os.path.join(OUT, "weights.csv"))
    L += ["", "== 3. Composite weights, last fold (share of |w| per SD; R2 on team-season win%)"]
    for arm in ARMS:
        r = w[w["arm"] == arm].iloc[-1]
        v = np.abs(r[NAMES].to_numpy(float))
        s = v / v.sum()
        L.append(f"  {arm:6s} fold {int(r['fold'])}  R2 {r['r2']:.4f}  " +
                 "  ".join(f"{n} {x:.3f}" for n, x in zip(NAMES, s)) +
                 f"  | ORB {s[2] + s[6]:.3f}")
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seasons", nargs="+", type=int, default=dyn.SEASONS)
    a = ap.parse_args()
    if a.run:
        run(a.seasons)
    if a.report or a.run:
        report()


if __name__ == "__main__":
    main()
