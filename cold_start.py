#!/usr/bin/env python3
"""Early-season model: last-season carryover for games 1-9 (MODEL_TAG v2).

  python cold_start.py fit --years 2023-2026     # writes model/logit_early.json

The composite model abstains until both teams have nc.MIN_GAMES (10) games.
research/cold_start_probe.py backtested replacements against the market on
the same games (2023-24 .. 2025-26, leave one season out). The arm shipped
here is its `carry_rho=0.25`, the leave-season-out choice in 2 of 3 seasons:

  * each team's features are nc's decayed four-factor totals over
    [last season's full log x RHO][this season's games so far], the decay
    counting games ago across the offseason, so last season fades as this
    season's games arrive;
  * P(home) = sigma(a + b*delta + c*b2b_net) with its own logit, fitted on
    every game in the first WINDOW games of the training seasons (the probe's
    protocol), stored in model/logit_early.json;
  * used only when min(games played) is 1..MIN_GAMES-1. Game 0 still
    abstains (the probe's worst bucket: the market knows the offseason, this
    does not). From MIN_GAMES on the v1 model is unchanged.

It did not beat the close in any bucket (about +0.02 log loss per game vs
the market in games 1-19, like the full-season gap); it makes early cards
about as good as mid-season ones, not better than the market.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import nba_composite as nc

RHO = 0.25           # weight on last season's games (offseason discount)
WINDOW = 20          # early-logit training window: min(games played) < WINDOW
FEATURES = ["delta", "b2b_net"]
MODEL_FILE = "logit_early.json"


def cold_features(cur, i, half_life=nc.HALF_LIFE, prior=None, rho=0.0,
                  pre=None, kappa=0.0):
    """Decayed four-factor features from [prior season][preseason][current].

    Rows are ordered oldest -> newest and the decay counts games ago across
    the whole sequence, so last season fades as new games arrive. `rho` and
    `kappa` multiply the prior-season and preseason rows. With rho = kappa =
    0 this equals nc.decayed_features(cur, i). Returns None when there is
    nothing to weight.
    """
    parts, mult = [], []
    if prior is not None and rho > 0 and len(prior):
        parts.append(prior[nc.COLS].to_numpy(float))
        mult.append(np.full(len(prior), float(rho)))
    if pre is not None and kappa > 0 and len(pre):
        parts.append(pre[nc.COLS].to_numpy(float))
        mult.append(np.full(len(pre), float(kappa)))
    if i > 0:
        parts.append(cur[nc.COLS].to_numpy(float)[:i])
        mult.append(np.ones(i))
    if not parts:
        return None
    A = np.vstack(parts)
    m = np.concatenate(mult)
    w = 0.5 ** (np.arange(len(A))[::-1] / half_life) * m
    return nc.features_from_totals((A * w[:, None]).sum(0))


def carry_delta(cur_h, ih, prior_h, cur_a, ia, prior_a, weights,
                rho=RHO, half_life=nc.HALF_LIFE):
    """Composite delta (home - away) from carryover features, or None."""
    fh = cold_features(cur_h, ih, half_life, prior=prior_h, rho=rho)
    fa = cold_features(cur_a, ia, half_life, prior=prior_a, rho=rho)
    if fh is None or fa is None:
        return None
    return float(nc.composite(fh - fa, weights["sd"], weights["w"]))


def early_games(y, weights, logs=None, prior_logs=None, window=WINDOW,
                rho=RHO, min_gp=0):
    """One row per home game of season y with min(games played) in
    [min_gp, window): carryover delta, b2b_net, result, games played."""
    logs = nc.load_logs(y) if logs is None else logs
    prior_logs = nc.load_logs(y - 1) if prior_logs is None else prior_logs
    rows = []
    for tm, c in logs.items():
        for i in np.where(c["home"].to_numpy())[0]:
            r = c.iloc[i]
            opp = r["opp"]
            if opp not in logs:
                continue
            c2 = logs[opp]
            j = np.where(c2["date"].to_numpy() == r["date"].to_datetime64())[0]
            if not len(j):
                continue
            j = int(j[0])
            if not (min_gp <= min(i, j) < window):
                continue
            d = carry_delta(c, i, prior_logs.get(tm), c2, j, prior_logs.get(opp),
                            weights, rho)
            if d is None:
                continue
            rh, ra = nc.rest_days(c, i), nc.rest_days(c2, j)
            rows.append({"year": y, "date": r["date"], "home": tm, "away": opp,
                         "gp_home": i, "gp_away": j,
                         "h_b2b": int(rh == 0), "a_b2b": int(ra == 0),
                         "b2b_net": int(ra == 0) - int(rh == 0), "delta": d,
                         "win": int(r["pts"] > r["opp_pts"])})
    return pd.DataFrame(rows)


def fit_early(years, weights, rho=RHO, window=WINDOW):
    G = pd.concat([early_games(y, weights, rho=rho, window=window) for y in years])
    m = nc.fit_logit(G[FEATURES].to_numpy(float), G["win"], FEATURES)
    m.update({"rho": rho, "window": window, "half_life": nc.HALF_LIFE,
              "n_games": int(len(G)), "years": list(years)})
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--years", nargs="+", default=["2023-2026"])
    a = ap.parse_args(argv)
    weights = nc.load_json("weights.json")
    m = fit_early(nc.parse_years(a.years), weights)
    nc.save_json(m, MODEL_FILE)
    coefs = ", ".join(f"{f}={c:+.4f}" for f, c in zip(m["features"], m["coef"]))
    print(f"saved model/{MODEL_FILE}  intercept={m['intercept']:.3f}  {coefs}  "
          f"rho={m['rho']}  n={m['n_games']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
