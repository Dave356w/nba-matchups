#!/usr/bin/env python3
"""Matrix interactions: is the replay ROI matrix's pattern (aligned bands
profitable, model good on market underdogs) more than noise?

  python research/matrix_interactions.py

Owner's question (2026-11-18), after reading ML stats' Historical replay
matrix: "when model and market bands align they are profitable; the model
does well against market underdogs; analyze interactions". The question was
prompted by the same matrix it is tested on, so the primary cohort cannot
confirm it: every reading on it is descriptive, and a second book/season
(ESPN BET 2024-25) and the 2026-27 live locks are reported as replication.

Population: the model's pick in every graded v11 game (games 1+, openers
out, voids out), one per game, flat 1u at the source price:
  primary      2025-26 DraftKings replay at the close (the public replay)
  replication  2024-25 ESPN BET replay at the close (other book, season)
  live         2026-27 native five-hour locks at DraftKings lock prices
Replay P is leave-one-season-out (the replayed season is not in its fit);
closes are later than a five-hour lock. Books and seasons are never pooled.

For each pick, null = expected 1u return if the no-vig price were exact
(q x decimal - 1, about minus half the hold); excess = units - null.
Readings (±95%, game-level):
  1  the 4x5 matrix: n, W-L, ROI, excess
  2  aligned (model band = market band, both >= 50%) vs off-diagonal
  3  market underdog picks (q < 50%) vs market favourite picks
  4  continuous: units on gap = p - q, underdog flag and their product;
     and a logistic of won on logit q with logit gap, underdog and product
     (slope 1 on logit q = market calibrated; a positive gap term = the
     model adds information beyond the price)
  5  multiplicity: how often a random relabelling of the same 20 cells
     yields a "best" contrast as large as the one observed
Descriptive research: no rule, filter, display or record changes.
Writes diagnostics/matrix_interactions/RESULTS.txt.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import analysis        # noqa: E402
import ledger          # noqa: E402
import market          # noqa: E402
import wp_calibration  # noqa: E402
from matrix_window_backtest import picks  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "matrix_interactions", "RESULTS.txt")
Z = 1.96
MODEL_LABELS = ["50-60", "60-70", "70-80", "80-100"]
MARKET_LABELS = ["<50", "50-60", "60-70", "70-80", "80-100"]


def enrich(d):
    d = d[d.key.map(lambda k: k[0]) == "pooled"].copy()
    d["mb"] = d.key.map(lambda k: k[1])
    d["qb"] = d.key.map(lambda k: k[2])
    # Null: expected 1u return if the no-vig q were exact.
    d["null"] = d.q * d.apply(lambda r: _dec(r), axis=1) - 1
    d["excess"] = d.units - d.null
    d["gap"] = d.p - d.q
    d["dog"] = (d.q < .5).astype(int)
    d["aligned"] = ((d.qb == d.mb + 1) & (d.qb >= 1)).astype(int)   # market bands are offset by the <50 column
    return d


def _dec(r):
    return r["_dec"]


def mean_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (x.mean() if n else np.nan), np.nan
    return x.mean(), Z * x.std(ddof=1) / np.sqrt(n)


def fmt(x, ci, pct=True):
    s = 100 if pct else 1
    if not np.isfinite(ci):
        return f"{s*x:+6.1f}" + (" " * 9)
    return f"{s*x:+6.1f} ± {s*ci:4.1f}"


def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + X)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtX = np.linalg.inv(X.T @ X)
    # HC1 robust SE
    meat = X.T @ (X * (e ** 2)[:, None]) * len(y) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(XtX @ meat @ XtX))
    return b, se


def logit_fit(X, y, iters=50):
    X = np.column_stack([np.ones(len(y))] + X)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        b = b + np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ b))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))
    return b, se


def lg(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def section(label, d, rng):
    L = [f"== {label} ==", f"picks {len(d)} · W-L {d.won.sum()}-{len(d)-d.won.sum()} · "
         f"ROI {fmt(*mean_ci(d.units))}% · null {100*d.null.mean():+.1f}% · excess {fmt(*mean_ci(d.excess))} pp", ""]
    L.append("1  Matrix (rows model WP of the pick, columns no-vig market WP of the same team)")
    L.append("   cell: n  W-L  ROI%  excess pp ±95%")
    L.append("   " + "model\\mkt".ljust(10) + "".join(c.rjust(30) for c in MARKET_LABELS))
    for mb, ml in enumerate(MODEL_LABELS):
        row = "   " + ml.ljust(10)
        for qb in range(5):
            c = d[(d.mb == mb) & (d.qb == qb)]
            if not len(c):
                row += "—".rjust(30)
                continue
            ex, ci = mean_ci(c.excess)
            cis = f"±{100*ci:.0f}" if np.isfinite(ci) else ""
            row += (f"{len(c)} {c.won.sum()}-{len(c)-c.won.sum()} {100*c.units.mean():+.0f} "
                    f"{100*ex:+.0f}{cis}").rjust(30)
        L.append(row)
    L.append("")

    def contrast(name, a, b):
        ea, ca = mean_ci(d[a].excess)
        eb, cb = mean_ci(d[b].excess)
        diff = ea - eb
        se = np.sqrt((ca / Z) ** 2 + (cb / Z) ** 2)
        return [f"   {name}",
                f"     in : n {a.sum():4d}  ROI {100*d[a].units.mean():+6.1f}%  excess {fmt(ea, ca)} pp",
                f"     out: n {b.sum():4d}  ROI {100*d[b].units.mean():+6.1f}%  excess {fmt(eb, cb)} pp",
                f"     difference {fmt(diff, Z*se)} pp  (z {diff/se:+.2f})"]

    L.append("2  Aligned bands (model band = market band, 50-60/60-70/70-80/80-100) vs the rest")
    L += contrast("aligned", d.aligned.eq(1), d.aligned.eq(0))
    fav = d.dog.eq(0)
    L += contrast("aligned, market favourites only", d.aligned.eq(1), fav & d.aligned.eq(0))
    L.append("")
    L.append("3  Market underdog picks (q < 50%: the model picks the team the market does not)")
    L += contrast("underdog", d.dog.eq(1), d.dog.eq(0))
    L.append("")

    L.append("4  Continuous interactions")
    gap = d.gap.values
    dog = d.dog.values.astype(float)
    b, se = ols([gap, dog, gap * dog], d.excess.values)
    names = ["intercept", "gap p-q", "underdog", "gap x underdog"]
    L.append("   OLS excess units on gap, underdog, gap x underdog (HC1, ±95%)")
    L += [f"     {n:16s} {b[i]:+.3f} ± {Z*se[i]:.3f}  (z {b[i]/se[i]:+.2f})" for i, n in enumerate(names)]
    al = d.aligned.values.astype(float)
    absg = np.abs(gap)
    b, se = ols([absg, al], d.excess.values)
    L.append("   OLS excess on |gap| and aligned (is 'aligned' just small gap?)")
    L += [f"     {n:16s} {b[i]:+.3f} ± {Z*se[i]:.3f}  (z {b[i]/se[i]:+.2f})"
          for i, n in enumerate(["intercept", "|gap|", "aligned"])]
    lq, lgap = lg(d.q.values), lg(d.p.values) - lg(d.q.values)
    b, se = logit_fit([lq, lgap, dog, lgap * dog], d.won.values.astype(float))
    L.append("   Logistic won ~ logit q + (logit p - logit q) + underdog + product (±95%)")
    L += [f"     {n:16s} {b[i]:+.3f} ± {Z*se[i]:.3f}" for i, n in
          enumerate(["intercept", "logit q", "logit gap", "underdog", "gap x underdog"])]
    L.append("     (logit q = 1 and gap = 0: the price is calibrated and the model adds nothing)")
    L.append("")

    # 5. multiplicity: largest |z| of any single cell's excess vs zero, under
    # sign-flipped excess (the null that every cell's excess is centred at 0).
    obs = _max_cell_z(d.excess.values, d.mb.values, d.qb.values)
    sims = []
    ex = d.excess.values - d.excess.values.mean()
    for _ in range(2000):
        sims.append(_max_cell_z(ex * rng.choice([-1, 1], len(ex)), d.mb.values, d.qb.values))
    L.append(f"5  Multiplicity: largest single-cell |z| (cells n >= 10) observed {obs:.2f}; "
             f"share of 2000 sign-flip nulls at least as large {np.mean(np.array(sims) >= obs):.2f}")
    L.append("")
    return L


def _max_cell_z(x, mb, qb):
    best = 0.0
    for a in range(4):
        for c in range(5):
            m = (mb == a) & (qb == c)
            if m.sum() >= 10:
                v = x[m]
                s = v.std(ddof=1) / np.sqrt(len(v))
                if s > 0:
                    best = max(best, abs(v.mean()) / s)
    return best


def with_dec(rows, source, frame):
    """picks() plus each pick's decimal price."""
    dec = {}
    for _, r in rows.iterrows():
        p = pd.to_numeric(r.get("lock_p_home" if source == "lock" else "p_home"), errors="coerce")
        side = "home" if p >= .5 else "away"
        ml = float(pd.to_numeric(r.get(f"{source}_{side}_ml"), errors="coerce"))
        dec[str(r["game_id"])] = 1 + (ml / 100 if ml > 0 else 100 / -ml)
    frame = frame.copy()
    frame["_dec"] = frame.game_id.map(dec)
    return frame


def main():
    recon = ledger.load(os.path.join(ROOT, ledger.RECON_PATH))
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    v11 = recon[recon["model_tag"].astype(str).str.endswith(analysis.FAMILY_TAGS)
                & recon["basis"].eq("reconstructed") & ~recon["game_id"].astype(str).isin(analysis.VOID_LOCKS)]
    rng = np.random.default_rng(20261118)
    cohorts = []
    for label, rows in [("2025-26 DraftKings replay at the close (primary, prompted the question)",
                         v11[(v11.season == 2026) & v11.close_book.eq("dk")]),
                        ("2024-25 ESPN BET replay at the close (replication)",
                         v11[(v11.season == 2025) & v11.close_book.eq("espnbet")])]:
        cohorts.append((label, enrich(with_dec(rows, "close", picks(rows, "close")))))
    live = analysis.model_predictions(native)
    live = live[live["lock_book"].astype(str).eq(analysis.HYPOTHESIS_BOOK)]
    cohorts.append(("2026-27 native five-hour locks, DraftKings lock prices (live, small)",
                    enrich(with_dec(live, "lock", picks(live, "lock")))))
    lines = ["Matrix interactions: replay ROI by model WP x market WP (games 1+, model's pick, flat 1u)",
             "Generated by research/matrix_interactions.py. ± is 95%. excess = units - no-vig null.", ""]
    for label, d in cohorts:
        lines += section(label, d, rng)
    text = "\n".join(lines).rstrip() + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
