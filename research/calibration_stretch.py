#!/usr/bin/env python3
"""Calibration stretch: would scaling the model's probabilities away from 50%
have helped, when the scale is learned only from earlier games?

  python research/calibration_stretch.py [--seasons 2017 ... 2026]

Owner's question (2026-11-12). Performance's accuracy-by-stated-chance table
shows the v11 picks winning about 3 points more often than they state on
games 10+ (964 of 978 games a 2025-26 hindsight replay; slope 1.14 ± 0.09 at
1 SE). A slope above 1 in one season is in-sample: this asks whether a
stretch fitted before each game is played would have paid.

The model is P = Phi(m / sigma), sigma the route's residual SD on the
training seasons. The stretch is one number k: P_k = Phi(k * m / sigma)
(k > 1 = more confident, the same as dividing sigma by k; the home edge and
the pick are unchanged). Predictions are margin_model.run's walk-forward
folds (production's inputs and routing as of v8, every fit on seasons
strictly before the test season), so k is learned from out-of-sample
predictions only, by maximum likelihood on win/loss:

  prior    k from every earlier test season's walk-forward predictions
  expand   prior plus this season's games on dates strictly before the game

Each stage has its own k (games 10+; games 1-9), fitted on that stage's
earlier games. A fold without MIN_FIT earlier games keeps k = 1 and is left
out of the comparison (the first test season has none).

Reading, fixed before the run (G7 form). The stretch is a candidate only if,
on games 10+, the prior arm's log loss minus the base's is negative with its
95% interval excluding zero pooled over the compared seasons, Brier agrees in
sign, it is better in a majority of those seasons, and it is not worse
against the close on either large book-season (2024-25 ESPN BET, 2025-26
DraftKings). Anything else reads as "no stretch". expand and games 1-9 are
reported beside it and decide nothing. Passing would make it a candidate
only: a change to the model's sigma is the owner's decision, with a new
model tag and the post-change checks.

Research only: nothing here touches the ledgers, the pages or model/*.json.
Writes diagnostics/calibration_stretch/RESULTS.txt. ± is 95%, game-level
(paired differences).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "research"))

# The root's modules first: research/ holds a live_wp.py of its own, which
# would shadow the root's once margin_model puts research/ ahead on the path.
import build_site  # noqa: E402,F401
import margin_model as mm                         # noqa: E402
import player_availability as pav                 # noqa: E402
import unified as u                               # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "calibration_stretch")
MIN_FIT = 500                      # earlier games needed before k leaves 1
K_LO, K_HI = 0.5, 2.0
STAGES = ((10, 999, "games 10+"), (1, 9, "games 1-9"))
BIG_BOOK_SEASONS = ((2025, "espnbet"), (2026, "dk"))
_GOLD = (np.sqrt(5) - 1) / 2


def fit_k(z, y, min_n=MIN_FIT, tol=1e-4):
    """Maximum-likelihood k of win ~ Phi(k z) on [K_LO, K_HI] (golden
    section; the log likelihood is concave in k). 1.0 below min_n rows."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    ok = np.isfinite(z) & np.isfinite(y)
    z, y = z[ok], y[ok]
    if len(z) < min_n:
        return 1.0

    def loss(k):
        return float(u.ll(mm.norm_cdf(k * z), y).sum())
    a, b = K_LO, K_HI
    c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
    fc, fd = loss(c), loss(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLD * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLD * (b - a)
            fd = loss(d)
    return float((a + b) / 2)


def stretch(d, min_fit=MIN_FIT):
    """One stage's walk-forward rows (year, slate_date, z, win) with k_prior,
    k_expand and their probabilities. k reads only rows of earlier seasons
    (prior) plus this season's earlier dates (expand); `fitted` marks folds
    whose prior k had min_fit earlier games."""
    d = d.sort_values(["year", "slate_date"]).reset_index(drop=True).copy()
    d["k_prior"], d["k_expand"], d["fitted"] = 1.0, 1.0, False
    for Y in sorted(d["year"].unique()):
        te = d["year"].eq(Y)
        pr = d[d["year"] < Y]
        zp, yp = pr["z"].to_numpy(float), pr["win"].to_numpy(float)
        d.loc[te, "k_prior"] = fit_k(zp, yp, min_fit)
        d.loc[te, "fitted"] = len(pr) >= min_fit
        cur = d[te]
        for day in cur["slate_date"].unique():
            seen = cur[cur["slate_date"] < day]
            d.loc[te & d["slate_date"].eq(day), "k_expand"] = fit_k(
                np.r_[zp, seen["z"].to_numpy(float)],
                np.r_[yp, seen["win"].to_numpy(float)], min_fit)
    d["p_base"] = mm.norm_cdf(d["z"].to_numpy(float))
    for arm in ("prior", "expand"):
        d[f"p_{arm}"] = mm.norm_cdf(d[f"k_{arm}"].to_numpy(float) * d["z"].to_numpy(float))
    return d


def diff(z, arm, ref="p_base"):
    """(log loss mean, ±95, Brier mean, ±95) of arm minus ref on z."""
    y = z["win"].to_numpy(float)
    a, r = z[arm].to_numpy(float), z[ref].to_numpy(float)
    return u.mean_ci(u.ll(a, y) - u.ll(r, y)) + u.mean_ci((a - y) ** 2 - (r - y) ** 2)


def stage_lines(s, name):
    L = [f"== {name}: stretched minus base (negative = the stretch helps), ± 95%"]
    c = s[s["fitted"]]
    if not len(c):
        return L + ["  no season with enough earlier games to fit k"], None
    wins = 0
    for Y, z in s.groupby("year"):
        own = fit_k(z["z"], z["win"], min_n=100)
        head = (f"  {u.season_name(Y)}  n {len(z):>5}  own-season k {own:.3f} "
                f"(slope {u.slope(z['p_base'], z['win'].to_numpy(float)):.2f})")
        if not z["fitted"].all():
            L.append(head + "  k = 1: too few earlier games, not compared")
            continue
        dl, dc, db, dbc = diff(z, "p_prior")
        el, ec, _, _ = diff(z, "p_expand")
        wins += dl < 0
        ke = z["k_expand"].to_numpy(float)
        L.append(head + f"  prior k {z['k_prior'].iloc[0]:.3f}: log loss {dl:+.4f} ± {dc:.4f}, "
                 f"Brier {db:+.4f} ± {dbc:.4f}  |  expand k {ke[0]:.3f}->{ke[-1]:.3f}: "
                 f"{el:+.4f} ± {ec:.4f}")
    n_seasons = c["year"].nunique()
    dl, dc, db, dbc = diff(c, "p_prior")
    el, ec, eb, ebc = diff(c, "p_expand")
    y = c["win"].to_numpy(float)
    L.append(f"  pooled   n {len(c):>5}  prior: log loss {dl:+.4f} ± {dc:.4f}, Brier {db:+.4f} "
             f"± {dbc:.4f}, better in {wins} of {n_seasons} seasons  |  expand: {el:+.4f} ± "
             f"{ec:.4f}, Brier {eb:+.4f} ± {ebc:.4f}")
    L.append(f"  pooled slope (1 = calibrated): base {u.slope(c['p_base'], y):.3f}  "
             f"prior {u.slope(c['p_prior'], y):.3f}  expand {u.slope(c['p_expand'], y):.3f}; "
             f"mean k prior {c['k_prior'].mean():.3f}, one k on all compared games "
             f"{fit_k(c['z'], c['win']):.3f} (in-sample)")
    return L, dict(ll=dl, ll_ci=dc, brier=db, wins=int(wins), seasons=int(n_seasons))


def close_lines(s, name):
    """Against the close, each book x season alone; {(season, book): prior
    minus base log loss} for the reading."""
    L, out = [f"== {name} against the close, each book x season alone (± 95%)"], {}
    m = mm.priced(s)
    m = m[m["fitted"]]
    for (season, book), z in m.groupby(["season", "close_book"]):
        if len(z) < 50:
            continue
        y, q = z["win"].to_numpy(float), z["q_close"].to_numpy(float)
        b, bc = u.mean_ci(u.ll(z["p_base"], y) - u.ll(q, y))
        p, pc = u.mean_ci(u.ll(z["p_prior"], y) - u.ll(q, y))
        dl, dc, _, _ = diff(z, "p_prior")
        out[(int(season), book)] = dl
        L.append(f"  {u.BOOK_NAMES.get(book, book)} {u.season_name(season)}  n {len(z):>5}  "
                 f"base - close {b:+.4f} ± {bc:.4f}  stretched - close {p:+.4f} ± {pc:.4f}  "
                 f"stretched - base {dl:+.4f} ± {dc:.4f}")
    if len(L) == 1:
        L.append("  no priced book-season with a fitted k")
    return L, out


def reading(r, close):
    """The pre-set reading on games 10+ (module docstring)."""
    if r is None:
        return ["== Reading", "  NO STRETCH: no season could be compared"]
    checks = [
        ("log loss negative, 95% interval excludes zero", r["ll"] + r["ll_ci"] < 0),
        ("Brier agrees in sign", r["brier"] < 0),
        (f"better in a majority of seasons ({r['wins']} of {r['seasons']})",
         r["wins"] * 2 > r["seasons"]),
    ]
    for key in BIG_BOOK_SEASONS:
        name = f"not worse against the close, {u.BOOK_NAMES[key[1]]} {u.season_name(key[0])}"
        checks.append((name, key in close and close[key] <= 0) if key in close
                      else (name + " (not available)", False))
    L = ["== Reading (fixed before the run; games 10+, prior arm)"]
    L += [f"  {'pass' if ok else 'FAIL'}  {name}" for name, ok in checks]
    L.append("  CANDIDATE: every check passes (owner's decision, new tag, post-change checks)"
             if all(ok for _, ok in checks) else "  NO STRETCH: at least one check fails")
    return L


def report(d):
    """RESULTS text from margin_model.run's frame."""
    d = d[np.isfinite(d["m_M"]) & np.isfinite(d["sigma_M"]) & d["win"].isin([0, 1])].copy()
    d["z"] = d["m_M"] / d["sigma_M"]
    L = ["Calibration stretch (research/calibration_stretch.py): P_k = Phi(k * m / sigma), "
         "k fitted on earlier walk-forward predictions only. ± is 95%.", ""]
    verdict = None
    for lo, hi, name in STAGES:
        s = stretch(d[d["g"].between(lo, hi)])
        sl, r = stage_lines(s, name)
        cl, close = close_lines(s, name)
        L += sl + [""] + cl + [""]
        if lo == 10:
            verdict = reading(r, close)
    return "\n".join(L + verdict)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", nargs="+", type=int, default=u.SEASONS)
    ap.add_argument("--cache", default=pav.DEFAULT_CACHE)
    a = ap.parse_args(argv)
    d, _models = mm.run(a.seasons, a.cache)
    text = report(d)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
