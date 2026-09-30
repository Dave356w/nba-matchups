#!/usr/bin/env python3
"""Calibration shape: are the model's probabilities too flat at the extremes,
or only late in the season?

  python research/calibration_shape.py --seasons 2025 2026 \
      [--game-years 2016-2019 2021-2026]

In-sample on the reconstructed rows, the outcome's slope on the model's logit
is ~1 overall but ~1.4-1.9 in March-April (too flat) and ~0.8 in October-
February (too steep). This tests that walk-forward: for each test season Y,
composite weights on WEIGHT_YEARS < Y and every logit on GAME_YEARS < Y, the
same training games for every arm (games 10+):

  base       P = sigma(a + b*delta + c*b2b_net)                  (shipped form)
  ext        base + d*delta*|delta|/100        extremes steeper (or flatter)
  phase      base + e*delta*phase              phase = days since opening
                                               night / 175, capped at 1
                                               (nc.logit_inputs, as shipped)
  late       base + e*delta*[date >= Mar 1]
  phase_ext  base + d*delta*|delta|/100 + e*delta*phase

Every feature is known before tip (the date and the season's opening night).
Reported per test season and closing book on identical games: log loss and
Brier (paired ± 95%) of each arm against base and against the close, the
outcome's slope on each arm's logit by phase (1 = calibrated shape), and
market favourites of 80%+ (mean model P vs market q vs actual).

Research only: no change to the model, the ledger or MODEL_TAG. Per-game
output goes to research/output/calibration_shape.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backfill_history as bf  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import nba_composite as nc  # noqa: E402
import walk_forward as wfm  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                   "calibration_shape.csv")
SEASON_DAYS = nc.SEASON_DAYS
BASE = list(nc.LOGIT_FEATURES)
ARMS = {
    "base": BASE,
    "ext": BASE + ["dsq"],
    "phase": BASE + ["d_phase"],
    "late": BASE + ["d_late"],
    "phase_ext": BASE + ["dsq", "d_phase"],
}


def add_shape(games, opening=None):
    """Add the shape features to nc.build_games rows of ONE season.

    Phase is measured from the season's opening night, as in production
    (nc.logit_inputs): by default the `phase` column nc.build_games already
    carries; `opening` recomputes it from that date. Never the earliest date
    in `games` -- for games 10+ that is ~3 weeks after opening night."""
    g = games.copy()
    d = pd.to_datetime(g["date"])
    if opening is not None:
        g["phase"] = [nc.season_phase(x, opening) for x in d]
    elif "phase" not in g:
        raise ValueError("add_shape needs the season's opening night or the "
                         "phase column from nc.build_games")
    g["late"] = ((d.dt.month >= 3) & (d.dt.month <= 6)).astype(int)
    g["dsq"] = g["delta"] * g["delta"].abs() / 100
    g["d_phase"] = g["delta"] * g["phase"]
    g["d_late"] = g["delta"] * g["late"]
    return g


def season_games(y, weights):
    g = nc.build_games(y, weights)
    g = add_shape(g)
    g["slate_date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    return g


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def slope(p, y):
    """Outcome's logistic slope on logit(p), ± SE (1 = calibrated shape)."""
    x = logit(p)
    if len(x) < 30:
        return float("nan"), float("nan")
    m = nc.fit_logit(x, y, l2=0.0)
    X = np.column_stack([np.ones(len(x)), x])
    q = 1 / (1 + np.exp(-X @ np.array([m["intercept"]] + m["coef"])))
    H = X.T @ (X * (q * (1 - q))[:, None])
    return float(m["coef"][0]), float(np.sqrt(np.linalg.inv(H)[1, 1]))


def score(y, weight_years, game_years):
    wy, gy = wfm.before(weight_years, y), wfm.before(game_years, y)
    if not wy or not gy:
        raise SystemExit(f"season {y}: no earlier training seasons "
                         f"(weights {wy}, logit {gy})")
    weights = nc.fit_weights(wy)
    tr = pd.concat([season_games(t, weights) for t in gy], ignore_index=True)
    te = season_games(y, weights)
    fits = {}
    for arm, feats in ARMS.items():
        fm = nc.fit_logit(tr[feats].to_numpy(float), tr["win"], feats)
        te[f"p_{arm}"] = nc.predict(fm, te[feats].to_numpy(float))
        fits[arm] = fm
    return te, fits, dict(weight_years=wy, game_years=gy, n_train=len(tr))


def report(m):
    lines = []
    for (season, book), g in m.groupby(["year", "close_book"]):
        y = g["home_won"].to_numpy(float)
        q = g["close_q_home"].to_numpy(float)
        base = g["p_base"].to_numpy(float)
        lines.append(f"\n{season} {market.BOOK_NAMES.get(book, book)} close · "
                     f"games 10+ (n={len(g)})")
        comps = [("base", "market", base, q)]
        for arm in ARMS:
            if arm != "base":
                a = g[f"p_{arm}"].to_numpy(float)
                comps += [(arm, "base", a, base), (arm, "market", a, q)]
        for na, nb, a, b in comps:
            s = wfm.paired(a, b, y)
            lines.append(
                f"  {na:9s} vs {nb:6s} logloss {s['ll_a']:.4f} vs {s['ll_b']:.4f}  "
                f"diff {s['d_ll']:+.4f} ± {1.96 * s['d_ll_se']:.4f}  "
                f"Brier diff {s['d_br']:+.4f} ± {1.96 * s['d_br_se']:.4f}")
        late = g["late"].to_numpy(bool)
        lines.append("  outcome slope on logit (1 = calibrated shape): "
                     "all / Oct-Feb / Mar-Apr")
        for name, col in [("market", "close_q_home")] + [(a, f"p_{a}") for a in ARMS]:
            p = g[col].to_numpy(float)
            parts = [slope(p[k], y[k]) for k in (np.ones(len(g), bool), ~late, late)]
            lines.append(f"    {name:9s} " + "   ".join(f"{s:.2f} ± {e:.2f}"
                                                       for s, e in parts))
        fav = np.maximum(q, 1 - q)
        home_fav = q >= 0.5
        for lo, hi in ((0.8, 0.9), (0.9, 1.0)):
            k = (fav >= lo) & (fav < hi)
            if not k.any():
                continue
            won = np.where(home_fav, y, 1 - y)[k]
            se = np.sqrt((fav[k] * (1 - fav[k])).sum()) / k.sum()
            arms = "  ".join(
                f"{a} {100 * np.where(home_fav, g[f'p_{a}'], 1 - g[f'p_{a}'])[k].mean():.1f}"
                for a in ARMS)
            lines.append(f"  market fav {100 * lo:.0f}-{100 * hi:.0f}% (n={k.sum()}): "
                         f"market {100 * fav[k].mean():.1f}  actual "
                         f"{100 * won.mean():.1f} ± {100 * se:.1f}  | model: {arms}")
    return "\n".join(lines)


def coef_line(name, fm):
    return f"  {name:9s} intercept {fm['intercept']:+.4f}  " + "  ".join(
        f"{f}={c:+.4f}" for f, c in zip(fm["features"], fm["coef"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--game-years", nargs="+", default=None,
                    help="logit seasons (only those before each test season "
                         "are used); default backfill_history.GAME_YEARS")
    a = ap.parse_args(argv)
    game_years = nc.parse_years(a.game_years) if a.game_years else bf.GAME_YEARS
    recon = ledger.graded(ledger.load(ledger.RECON_PATH))
    recon = recon[["slate_date", "home", "away", "close_q_home", "close_book",
                   "home_won"]]
    allm = []
    for y in a.seasons:
        te, fits, used = score(y, bf.WEIGHT_YEARS, game_years)
        m = te.merge(recon, on=["slate_date", "home", "away"], how="inner")
        print(f"\n=== season {y}: weights {used['weight_years']}, logit "
              f"{used['game_years']} (n={used['n_train']}); test games 10+ "
              f"{len(te)}, matched to reconstructed rows {len(m)}", flush=True)
        for k, fm in fits.items():
            print(coef_line(k, fm))
        allm.append(m)
    m = pd.concat(allm, ignore_index=True)
    print("\n== Same games, one book at a time (negative diff = first is better)")
    print(report(m))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    m.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(m)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
