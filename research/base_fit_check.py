#!/usr/bin/env python3
"""Base route fit check: is model/logit.json what its recipe says, and does
each of its inputs earn its place?

  python research/base_fit_check.py

Owner's request (2026-11-12): sanity check the base route's fit. The base
margin model is fitted by "Fit model" (nba_composite.py fit-logit --v11) on
every game of its seasons from the game logs, which are not in the repo, so
this rebuilds the same training table (fade_v8_games with the committed
weights, rookies by pick, preseason rosters at game 0) and reports:

  1  the committed composite weights and base fit against a refit here
     (they should reproduce), and the training table's integrity;
  2  each coefficient with its standard error, t and the points a 1-SD
     move is worth;
  3  correlations between inputs and variance inflation;
  4  the fit season by season (sign and size stability, home edge, sigma);
  5  residuals: shape, spread by stage (game 0, games 1-9, 10+), by season
     phase and by predicted margin, and stated against actual win rate;
  6  walk-forward, each input dropped in turn: every fit on seasons strictly
     before the test season (the committed composite weights are used
     throughout, so the rating itself has seen every season: the same for
     every arm), log loss and squared margin error, without minus with.

Descriptive: no pass rule and no candidate. A change to the model's inputs
is the owner's decision, with its own walk-forward test and a new tag.
Research only: nothing here touches the ledgers, the pages or model/*.json.
Writes diagnostics/base_fit_check/RESULTS.txt. ± is 95%, game-level.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import nba_composite as nc                        # noqa: E402
import player_availability as pav                 # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "base_fit_check")
Z95 = 1.959964
STAGES = ((0, 0, "game 0"), (1, 9, "games 1-9"), (10, 999, "games 10+"))


def training_table(model, weights, cache_dir=None):
    """The games fit-logit --v11 fits on: every game of the model's seasons
    with complete inputs (and gp = the lesser of the teams' games played)."""
    feats = model["features"]
    G = pd.concat([nc.fade_v8_games(y, weights, model.get("half_life", nc.HALF_LIFE),
                                    talent=pav.season_talent(y, cache_dir, rookies=True),
                                    talent0=pav.preseason_talent(y, cache_dir=cache_dir))
                   for y in model["years"]], ignore_index=True)
    G = G[np.isfinite(G[feats].to_numpy(float)).all(axis=1)].reset_index(drop=True)
    G["gp"] = np.minimum(G["gp_home"], G["gp_away"])
    return G


def ols(X, y):
    """(beta with intercept first, standard errors, sigma, residuals)."""
    X1 = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ b
    s2 = r @ r / max(1, len(y) - X1.shape[1])
    return b, np.sqrt(np.diag(s2 * np.linalg.pinv(X1.T @ X1))), float(np.sqrt(s2)), r


def vif(X):
    """Variance inflation of each column against the others."""
    X = np.asarray(X, float)
    out = []
    for j in range(X.shape[1]):
        o = np.column_stack([np.ones(len(X)), np.delete(X, j, 1)])
        fit = o @ np.linalg.lstsq(o, X[:, j], rcond=None)[0]
        r2 = 1 - ((X[:, j] - fit) ** 2).sum() / ((X[:, j] - X[:, j].mean()) ** 2).sum()
        out.append(float(1 / max(1e-12, 1 - r2)))
    return out


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ci(d):
    d = np.asarray(d, float)
    return float(d.mean()), float(Z95 * d.std(ddof=1) / np.sqrt(len(d)))


def walk_forward(G, feats, min_train=1):
    """Per-game log loss and squared margin error on each season from the
    fit of the seasons strictly before it (the first `min_train` only train)."""
    years = sorted(G["year"].unique())
    rows = []
    for y in years[min_train:]:
        tr, te = G[G["year"] < y], G[G["year"] == y]
        b, _, sig, _ = ols(tr[feats], tr["margin"])
        m = b[0] + te[feats].to_numpy(float) @ b[1:]
        rows.append(pd.DataFrame({"year": y, "gp": te["gp"].to_numpy(),
                                  "ll": ll(nc.norm_cdf(m / sig), te["win"].to_numpy(float)),
                                  "se": (te["margin"].to_numpy(float) - m) ** 2}))
    return pd.concat(rows, ignore_index=True)


def drop_one(G, feats, extra=()):
    """[(name, n, log-loss diff, ±, sq-err diff, ±, seasons helped, seasons)]
    for games 1+ and games 10+, each input (and each `extra` group) dropped."""
    full = walk_forward(G, feats)
    out = []
    for group in [(f,) for f in feats] + [tuple(e) for e in extra]:
        r = walk_forward(G, [f for f in feats if f not in group])
        for lo, name in ((1, "games 1+"), (10, "games 10+")):
            k = (full["gp"] >= lo).to_numpy()
            dl, ds = (r["ll"] - full["ll"])[k], (r["se"] - full["se"])[k]
            per = dl.groupby(full["year"][k]).mean()
            out.append(("+".join(group), name, int(k.sum()), *ci(dl), *ci(ds),
                        int((per > 0).sum()), int(len(per))))
    return full, out


def report(G, model, weights_refit=None, weights=None):
    feats = model["features"]
    X, y = G[feats].to_numpy(float), G["margin"].to_numpy(float)
    b, se, sig, res = ols(X, y)
    L = ["Base route fit check (research/base_fit_check.py): model/logit.json rebuilt from "
         "the game logs. ± is 95%.", ""]

    L.append("== 1. Reproduction and training table")
    if weights_refit is not None and weights is not None:
        dw = np.abs(np.asarray(weights_refit["w"]) - np.asarray(weights["w"])).max()
        ds = np.abs(np.asarray(weights_refit["sd"]) - np.asarray(weights["sd"])).max()
        L.append(f"  composite weights refitted on {weights['years'][0]}..{weights['years'][-1]}: "
                 f"max |w diff| {dw:.2e}, max |sd diff| {ds:.2e}")
    L.append(f"  base fit: n {len(G)} here vs {model.get('n_games')} committed; max |coef diff| "
             f"{np.abs(b[1:] - np.asarray(model['coef'])).max():.2e}, intercept diff "
             f"{abs(b[0] - model['intercept']):.2e}, sigma {sig:.4f} vs {model['sigma']:.4f}")
    L.append("  games by season: " + ", ".join(f"{int(s)} {n}" for s, n in
                                               G.groupby("year").size().items()))
    L.append(f"  duplicate games {int(G.duplicated(['date', 'home', 'away']).sum())}, tied "
             f"margins {int((y == 0).sum())}, by stage: " + ", ".join(
                 f"{name} {int(G['gp'].between(lo, hi).sum())}" for lo, hi, name in STAGES))

    L += ["", "== 2. Coefficients (points of home margin), ± 1 SE"]
    L.append(f"  {'intercept':<12} {b[0]:+8.3f} ± {se[0]:.3f}  t {b[0] / se[0]:5.1f}")
    for f, c, s, sd in zip(feats, b[1:], se[1:], X.std(0)):
        L.append(f"  {f:<12} {c:+8.3f} ± {s:.3f}  t {c / s:5.1f}   1 SD = {sd:6.2f} -> "
                 f"{c * sd:+.2f} pts")

    L += ["", "== 3. Correlations between inputs, and variance inflation"]
    L += ["  " + ln for ln in G[feats].corr().round(2).to_string().split("\n")]
    L.append("  VIF: " + ", ".join(f"{f} {v:.2f}" for f, v in zip(feats, vif(X))))

    L += ["", "== 4. Fitted season by season"]
    rows = []
    for s, g in G.groupby("year"):
        bb, _, sg, _ = ols(g[feats], g["margin"])
        rows.append(pd.Series([len(g), bb[0], *bb[1:], sg, g["win"].mean()],
                              index=["n", "home", *feats, "sigma", "home won"], name=int(s)))
    T = pd.DataFrame(rows)
    L += ["  " + ln for ln in T.round(3).to_string().split("\n")]
    L.append("  inputs negative in a season: " + (", ".join(
        f"{f} {int((T[f] < 0).sum())} of {len(T)}" for f in feats if (T[f] < 0).any()) or "none"))

    L += ["", "== 5. Residuals (pooled fit)"]
    z = res / res.std()
    L.append(f"  mean {res.mean():+.3f}, sd {res.std():.2f}, skew {(z ** 3).mean():+.2f}, excess "
             f"kurtosis {(z ** 4).mean() - 3:+.2f}; within 1 sigma {np.mean(np.abs(res) < sig):.3f} "
             f"(normal 0.683), within 0.25 sigma {np.mean(np.abs(res) < 0.25 * sig):.3f} (0.197)")
    for lo, hi, name in STAGES:
        k = G["gp"].between(lo, hi).to_numpy()
        if k.any():
            L.append(f"  {name:<10} n {k.sum():>5}  residual sd {res[k].std():.2f}  mean "
                     f"{res[k].mean():+.2f} ± {Z95 * res[k].std() / np.sqrt(k.sum()):.2f}")
    for lo, hi in ((0, 0.1), (0.1, 0.33), (0.33, 0.66), (0.66, 1.01)):
        k = ((G["phase"] >= lo) & (G["phase"] < hi)).to_numpy()
        if k.any():
            L.append(f"  phase {lo:.2f}-{min(hi, 1):.2f} n {k.sum():>5}  residual sd "
                     f"{res[k].std():.2f}  mean {res[k].mean():+.2f} ± "
                     f"{Z95 * res[k].std() / np.sqrt(k.sum()):.2f}")
    pred = y - res
    p, win = nc.norm_cdf(pred / sig), G["win"].to_numpy(float)
    L.append("  stated P(home) against the home win rate (in sample; se if P were right):")
    for lo, hi in ((0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        k = (p >= lo) & (p < hi)
        if k.any():
            L.append(f"    {lo:.1f}-{min(hi, 1):.1f}  n {k.sum():>5}  stated {p[k].mean():.3f}  "
                     f"actual {win[k].mean():.3f}  (se {np.sqrt((p[k] * (1 - p[k])).sum()) / k.sum():.3f})")

    L += ["", "== 6. Walk-forward, each input dropped: without minus with (positive = the "
              "input helps), ± 95%"]
    full, rows = drop_one(G, feats, extra=[("luck_def", "ft_diff")]
                          if {"luck_def", "ft_diff"} <= set(feats) else ())
    for lo, name in ((1, "games 1+"), (10, "games 10+")):
        k = full["gp"] >= lo
        L.append(f"  full model, {name}: n {int(k.sum())}, log loss {full['ll'][k].mean():.4f}, "
                 f"margin RMSE {np.sqrt(full['se'][k].mean()):.3f}")
    for name, stage, n, dl, dlc, ds, dsc, wins, seasons in rows:
        L.append(f"  -{name:<18} {stage:<9} log loss {dl:+.4f} ± {dlc:.4f}   sq. err "
                 f"{ds:+.2f} ± {dsc:.2f}   helps in {wins} of {seasons} seasons")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=None, help="box-score cache (default research/output)")
    a = ap.parse_args(argv)
    weights, model = nc.load_json("weights.json"), nc.load_json("logit.json")
    refit = nc.fit_weights(weights["years"], orb_pct=nc.is_orb_pct(weights))
    G = training_table(model, weights, a.cache)
    text = report(G, model, refit, weights)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
