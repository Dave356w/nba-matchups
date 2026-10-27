#!/usr/bin/env python3
"""Half-life sweep: v9's base margin model, walk-forward, at each rating decay.

  python research/half_life_sweep.py --half-life 25 [--seasons 2017 ... 2027]
  python research/half_life_sweep.py --report

Owner's question (2026-10-27): the shipped half-life of 25 games was chosen
on the original report's few seasons ("20-45 flat", docs/
nba_composite_model_report.pdf §5) and never confirmed on more data
(docs/research_history.md). This re-runs the choice on v9.

Arms, fixed before the run: HALF_LIVES games, plus "inf" (equal weights,
season-to-date; last season still fades in by rho(i) for games 0-9). The
half-life sets every decayed input together, as in production: the
four-factor totals, opponents' 3P luck and own FT% (fade_v8_games), and
the offseason decay of last season's games. Talent and availability terms
do not depend on it. Each fold Y refits the composite weights on
WEIGHT_YEARS before Y and v9's base OLS on the game seasons before Y
(research/dynamic_fit.py's S arm), then scores season Y.

Readings: log loss, Brier and margin MSE against outcomes, each half-life
minus 25 on the same games (negative = better), games 1+ pooled as a
description and by season and stage; and against the close, each basis x
book x season alone. Decision rule, set before the run: 25 stays unless
another half-life is better on games 1+ by more than the 95% interval AND
in at least 7 of the 10 folds; even then it is a candidate for a new model
tag, the owner's decision, not a change. Research only: nothing here touches
the ledgers, the pages or model/*.json. ± is 95%, game-level, paired.
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

OUT = os.path.join(ROOT, "diagnostics", "half_life_sweep")
HALF_LIVES = [10, 15, 20, 25, 35, 45, 60, "inf"]
PRODUCTION = 25
INF = 1e9                                          # equal weights


def hl_value(h):
    return INF if h == "inf" else float(h)


def fold(Y, h):
    """Season Y scored by the walk-forward base fit at half-life h."""
    wts = dyn.fold_weights(Y)

    def games(t):
        tal, tal0 = dyn.talents(t)
        g = nc.fade_v8_games(t, wts, half_life=hl_value(h), talent=tal, talent0=tal0)
        g["date"] = pd.to_datetime(g["date"])
        g["slate_date"] = g["date"].dt.strftime("%Y-%m-%d")
        return g

    past = pd.concat([games(t) for t in dyn.GAME_YEARS if t < Y], ignore_index=True)
    past = past[np.isfinite(past[dyn.FEATS + ["margin"]].to_numpy(float)).all(axis=1)]
    cur = games(Y)
    cur = cur[np.isfinite(cur[dyn.CORE].to_numpy(float)).all(axis=1)].copy()
    b, s = dyn.wls(past)
    cur["m"], cur["p"] = dyn.predict(b, s, cur)
    mn = np.fmin(cur["gp_home"], cur["gp_away"])
    cur["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
    cur["season"], cur["hl"] = Y, str(h)
    dyn.log(f"hl {h} fold {Y}: {len(cur)} games, intercept {b[0]:+.2f} sigma {s:.2f}")
    return cur[["season", "slate_date", "home", "away", "gp_home", "gp_away", "stage",
                "margin", "m", "p", "hl"]]


def report():
    P = pd.concat([pd.read_csv(os.path.join(OUT, f"predictions_hl{h}.csv"))
                   for h in HALF_LIVES], ignore_index=True)
    key = ["season", "slate_date", "home", "away"]
    W = P.pivot_table(index=key + ["stage", "margin"], columns="hl", values=["m", "p"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index().dropna()
    y = (W["margin"] > 0).to_numpy(float)
    base = str(PRODUCTION)
    L = [__doc__.strip().splitlines()[0], "",
         f"Games scored by every half-life: {len(W)}. Each row: half-life - {PRODUCTION} "
         "(negative = better), ± 95%.", ""]

    def cell(g, h, yy):
        d = dyn.ll(g[f"p_{h}"], yy) - dyn.ll(g[f"p_{base}"], yy)
        mse = (g["margin"] - g[f"m_{h}"]) ** 2 - (g["margin"] - g[f"m_{base}"]) ** 2
        return dyn.ci(d), dyn.ci(mse)

    L.append("== 1. Against outcomes, log loss (and margin MSE)")
    for st, g in [("games 1+", W[W["stage"] != "game 0"])] + list(W.groupby("stage")):
        yy = (g["margin"] > 0).to_numpy(float)
        L.append(f"  {st} (n {len(g)}; hl {PRODUCTION} log loss "
                 f"{dyn.ll(g[f'p_{base}'], yy).mean():.4f})")
        for h in HALF_LIVES:
            if str(h) == base:
                continue
            (m, e, _), (mm, me, _) = cell(g, str(h), yy)
            wins = sum(cell(s, str(h), (s["margin"] > 0).to_numpy(float))[0][0] < 0
                       for _, s in g.groupby("season"))
            L.append(f"    hl {str(h):>4s}  {m:+.4f} ± {e:.4f}   MSE {mm:+.2f} ± {me:.2f}   "
                     f"better in {wins} of {g['season'].nunique()} seasons")
    L += ["", "  By season, games 1+ (log loss, hl - 25):"]
    G = W[W["stage"] != "game 0"]
    L.append("    season   n     " + "  ".join(f"{str(h):>16s}" for h in HALF_LIVES
                                              if str(h) != base))
    for s, g in G.groupby("season"):
        yy = (g["margin"] > 0).to_numpy(float)
        L.append(f"    {s}  {len(g):5d}  " + "  ".join(
            f"{cell(g, str(h), yy)[0][0]:+.4f}±{cell(g, str(h), yy)[0][1]:.4f}"
            for h in HALF_LIVES if str(h) != base))

    M = dyn.market_rows()
    J = W.merge(M, on=key, how="inner")
    L += ["", "== 2. Against the close, each basis x book x season alone "
          "(log loss half-life - close; ± 95%)"]
    for (basis, book, s), g in J.groupby(["basis", "close_book", "season"]):
        for st, h_ in [("games 1+", g[g["stage"] != "game 0"])] + list(g.groupby("stage")):
            if len(h_) < 100:
                continue
            yy = (h_["margin"] > 0).to_numpy(float)
            lc = dyn.ll(h_["close_q_home"].to_numpy(float), yy)
            L.append(f"  {basis:13s} {book:7s} {s} {st:10s} n {len(h_):4d}  " + "  ".join(
                f"{h}: {dyn.ci(dyn.ll(h_[f'p_{h}'], yy) - lc)[0]:+.4f}" for h in HALF_LIVES))
            L.append(" " * 50 + "  (± about " +
                     f"{dyn.ci(dyn.ll(h_[f'p_{base}'], yy) - lc)[1]:.4f} each)")
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half-life")
    ap.add_argument("--seasons", nargs="+", type=int, default=dyn.SEASONS)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    h = a.half_life if a.half_life == "inf" else int(a.half_life)
    P = pd.concat([fold(Y, h) for Y in a.seasons], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    P.to_csv(os.path.join(OUT, f"predictions_hl{h}.csv"), index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
