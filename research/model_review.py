#!/usr/bin/env python3
"""Model review: where the current model stands against the market and the outcomes (HINDSIGHT).

  python research/model_review.py

The aggregate check behind docs/model_guidelines.md. On the reconstructed
rows (leave-one-season-out; for 2025-26 every fit uses earlier seasons
only), each sportsbook close book x season, never pooled:

  1. Scores: log loss and Brier, model vs the close on the same games, by
     route (games 10+, games 1-9); the outcome's calibration slope on each
     forecast; w, the outcome's weight on the model's disagreement with the
     price (y ~ a + b*logit q + w*(logit P - logit q); 0 = the model adds
     nothing to the price, 1 = trust the model's disagreement in full), at
     the close and at the open.
  2. Where the gap to the close sits (games 10+): model - close log loss by
     month, by the market favourite's no-vig price, by how far the model
     disagrees with the close, and by whether its lean is the favourite.
  3. Claimed vs realised disagreement: on the side the model prefers (P >
     q), the model's claimed gap P - q beside the realised gap y - q, by
     size of the claim, at the close and the open. realised / claimed is
     the share of the model's disagreement the outcomes bore out.
  4. Power: for each pre-registered rule, its hindsight effect (DraftKings
     where it has bets), the per-bet spread of units, bets per season,
     and the bets a native test needs to put that effect (and half of it)
     2 SE from its null.

Research only: nothing here touches the ledgers, the pages or the model.
Writes diagnostics/model_review/RESULTS.txt. ± is 95%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis                                   # noqa: E402
import ledger                                     # noqa: E402
import nba_composite as nc                        # noqa: E402

OUT = ROOT / "diagnostics" / "model_review"
Z95 = 1.96
EPS = 1e-4
BOOK_NAMES = {"dk": "DraftKings", "espnbet": "ESPN BET"}


def season_name(y):
    return f"{int(y) - 1}-{str(int(y))[-2:]}"


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mean_ci(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    return float(d.mean()), float(Z95 * d.std(ddof=1) / np.sqrt(len(d)))


def fit(X, y):
    """Logistic fit with intercept: (coefs, 95% half-widths), or Nones."""
    if len(y) < 30:
        return None, None
    try:
        fm = nc.fit_logit(X, y, l2=0.0)
    except np.linalg.LinAlgError:
        return None, None
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.array([fm["intercept"]] + list(fm["coef"]))
    pr = 1 / (1 + np.exp(-X1 @ b))
    cov = np.linalg.inv(X1.T @ (X1 * (pr * (1 - pr))[:, None]))
    return b, Z95 * np.sqrt(np.diag(cov))


def slope(p, y):
    """Calibration slope of y on logit(p): 1 = calibrated shape."""
    b, ci = fit(logit(p)[:, None], y)
    return (float(b[1]), float(ci[1])) if b is not None else (np.nan, np.nan)


def weight(p, q, y):
    """w: y ~ a + b*logit(q) + w*(logit(p) - logit(q))."""
    lq = logit(q)
    b, ci = fit(np.column_stack([lq, logit(p) - lq]), y)
    return (float(b[2]), float(ci[2])) if b is not None else (np.nan, np.nan)


def frame(recon):
    """Graded reconstructed rows with the close and open no-vig q."""
    g = ledger.graded(recon).copy()
    g["q_close"] = analysis.price_q_home(g, "close")
    g["q_open"] = analysis.price_q_home(g, "open")
    g["p"] = pd.to_numeric(g["p_home"], errors="coerce")
    g["y"] = pd.to_numeric(g["home_won"], errors="coerce")
    opener, g["early"] = analysis.stage(g)          # game 0 (v8) is neither route
    g["late"] = ~opener & ~g["early"]
    g["month"] = pd.to_datetime(g["slate_date"]).dt.month
    ok = g["q_close"].between(0, 1, inclusive="neither") & g["p"].between(0, 1)
    return g[ok]


def groups(g):
    """[(label, rows)] per sportsbook close book x season, never pooled."""
    out = []
    for (season, book), x in g.groupby(["season", "close_book"]):
        out.append((f"{BOOK_NAMES.get(book, book)} {season_name(season)}", x))
    return sorted(out, key=lambda t: t[0])


def gap_line(label, x, width=30):
    if len(x) < 20:
        return f"    {label:<{width}} n={len(x):<5} (too few)"
    d = logloss(x["p"], x["y"]) - logloss(x["q_close"], x["y"])
    m, ci = mean_ci(d)
    z = m / (ci / Z95) if ci > 0 else float("nan")
    return (f"    {label:<{width}} n={len(x):<5} model - close {m:+.4f} ± {ci:.4f}"
            f"  (z {z:+.1f})")


def block_scores(g):
    L = ["== 1. Scores vs the close and the outcomes (log loss / Brier: negative = "
         "model better)"]
    for label, x in groups(g):
        for route, r in (("games 10+", x[x["late"]]), ("games 1-9", x[x["early"]])):
            if len(r) < 30:
                continue
            y = r["y"].to_numpy(float)
            p, qc = r["p"].to_numpy(float), r["q_close"].to_numpy(float)
            dl = logloss(p, y) - logloss(qc, y)
            db = (p - y) ** 2 - (qc - y) ** 2
            ml, mlc = mean_ci(dl)
            mb, mbc = mean_ci(db)
            sp, spc = slope(p, y)
            sq, sqc = slope(qc, y)
            wc, wcc = weight(p, qc, y)
            ro = r[r["q_open"].between(0, 1, inclusive="neither")]
            wo, woc = weight(ro["p"].to_numpy(float), ro["q_open"].to_numpy(float),
                             ro["y"].to_numpy(float))
            L.append(f"  {label} · {route}: n={len(r)}")
            L.append(f"    log loss model {logloss(p, y).mean():.4f}  close "
                     f"{logloss(qc, y).mean():.4f}  model - close {ml:+.4f} ± {mlc:.4f}"
                     f"   Brier {mb:+.4f} ± {mbc:.4f}")
            L.append(f"    calibration slope: model {sp:.2f} ± {spc:.2f}  close "
                     f"{sq:.2f} ± {sqc:.2f}")
            L.append(f"    w (weight on the model's disagreement): at the close "
                     f"{wc:+.2f} ± {wcc:.2f}   at the open {wo:+.2f} ± {woc:.2f} "
                     f"(n={len(ro)})")
    return L


def block_where(g):
    L = ["== 2. Where the gap to the close sits (games 10+, model - close log loss)"]
    for label, x in groups(g):
        x = x[x["late"]]
        if len(x) < 100:
            continue
        L.append(f"  {label}")
        L.append("   by month:")
        for months, name in (((10, 11), "Oct-Nov"), ((12,), "Dec"), ((1,), "Jan"),
                             ((2,), "Feb"), ((3, 4), "Mar-Apr")):
            L.append(gap_line(name, x[x["month"].isin(months)]))
        fav = np.maximum(x["q_close"], 1 - x["q_close"])
        L.append("   by the market favourite's no-vig price:")
        for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)):
            L.append(gap_line(f"fav {lo:.1f}-{hi:.1f}", x[(fav >= lo) & (fav < hi)]))
        dis = np.abs(logit(x["p"]) - logit(x["q_close"]))
        L.append("   by |logit P - logit q| (how far the model disagrees):")
        for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 9)):
            L.append(gap_line(f"{lo:.2f}-{hi if hi < 9 else 'up':}", x[(dis >= lo) & (dis < hi)]))
        same = (x["p"] >= 0.5) == (x["q_close"] >= 0.5)
        L.append("   by the lean vs the market favourite:")
        L.append(gap_line("lean = market favourite", x[same]))
        L.append(gap_line("lean = market underdog", x[~same]))
    return L


def block_claims(g):
    L = ["== 3. Claimed vs realised disagreement, the side with model P > no-vig q "
         "(games 10+; pp of win probability)"]
    for price in ("close", "open"):
        qcol = f"q_{price}"
        L.append(f"  at the {price}:")
        for label, x in groups(g):
            x = x[x["late"] & x[qcol].between(0, 1, inclusive="neither")]
            if len(x) < 100:
                continue
            home = x["p"] > x[qcol]
            claim = np.where(home, x["p"] - x[qcol], x[qcol] - x["p"])
            real = np.where(home, x["y"] - x[qcol], x[qcol] - x["y"])
            L.append(f"    {label}")
            for lo, hi in ((0, 0.04), (0.04, 0.08), (0.08, 0.12), (0.12, 1)):
                k = (claim >= lo) & (claim < hi)
                if k.sum() < 20:
                    continue
                m, ci = mean_ci(real[k])
                c = float(claim[k].mean())
                L.append(f"      claim {100 * lo:>2.0f}-{100 * hi if hi < 1 else 99:>2.0f} pp"
                         f"  n={int(k.sum()):<5} claimed {100 * c:+5.1f}  realised "
                         f"{100 * m:+5.1f} ± {100 * ci:4.1f}  share {m / c:+.2f}")
            m, ci = mean_ci(real)
            c = float(claim.mean())
            L.append(f"      all            n={len(claim):<5} claimed {100 * c:+5.1f}  "
                     f"realised {100 * m:+5.1f} ± {100 * ci:4.1f}  share {m / c:+.2f}"
                     f" (± {ci / c:.2f})")
    return L


def block_power(recon):
    L = ["== 4. Power: native bets each rule needs (z = 2 at 1 SE)",
         "  n2 = (2 x sd / effect)^2, effect = hindsight ROI - null at the "
         "registered price (DraftKings hindsight; H1/H6, which DraftKings never "
         "priced, the registered pooled ESPN BET hindsight); bets per season = "
         "the rule's hindsight bets over every book, per priced season."]
    seasons = pd.to_numeric(recon["season"], errors="coerce").nunique()
    dk = recon[recon["close_book"] == "dk"]

    def bets(g, hyp):
        g = analysis.with_open_spreads(g) if hyp.get("market") == "ats" else g
        return analysis.hypothesis_picks(g, hyp, hyp["hindsight"]) if g is not None else None
    for label, hyp in analysis.hypothesis_groups():
        d, src = bets(dk, hyp), "DraftKings"
        if d is None or not len(d):
            d, src = bets(recon, hyp), "pooled ESPN BET"
        if d is None or not len(d):
            L.append(f"  {label:<10} no hindsight bets")
            continue
        allb = bets(recon, hyp)
        per = len(allb) / seasons if allb is not None and seasons else 0
        u = d["units"].to_numpy(float)
        eff = float(u.mean() - d["units_null"].mean())
        sd = float(u.std(ddof=1))
        n2 = (2 * sd / eff) ** 2 if eff > 0 else float("inf")
        span = (f"(~{n2 / per:.1f} seasons); at half the effect {4 * n2:,.0f} "
                f"(~{4 * n2 / per:.1f})" if per and np.isfinite(n2) else "")
        L.append(f"  {label:<10} effect {100 * eff:+5.1f} pts ({src}, n={len(d)})  "
                 f"sd/bet {sd:.2f}  bets/season {per:>6.0f}  n for 2 SE: {n2:,.0f} "
                 + span)
    return L


def report(recon):
    g = frame(recon)
    L = ["Model review -- reconstructed rows, HINDSIGHT (leave-one-season-out; "
         "2025-26 fits use earlier seasons only)",
         "Model tags: " + ", ".join(sorted(g["model_tag"].astype(str).unique())),
         "Each sportsbook close book x season stands alone. ± is 95%.", ""]
    for block in (block_scores(g), block_where(g), block_claims(g), block_power(recon)):
        L += block + [""]
    return "\n".join(L)


def main():
    recon = ledger.load(ledger.RECON_PATH)
    text = report(recon)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
