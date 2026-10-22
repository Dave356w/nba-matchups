#!/usr/bin/env python3
"""Disagreement shrinkage toward the opening price (HINDSIGHT).

  python research/open_shrinkage.py

docs/model_guidelines.md G3 / next test 3. At the open the outcomes bear
out about a third of the model's disagreement with the price (w = +0.33 /
+0.44); at the close none of it. So read the model through the opener:

  P* = sigma(logit q_open + w * (logit P - logit q_open))

with w fitted on the OTHER priced season(s) (2025-26 rows: fitted on
2024-25, a walk-forward test; 2024-25 rows: fitted on 2025-26, labelled
hindsight). Shrinking never changes which side the model prefers (P* - q
has the sign of P - q); it changes how big its edge looks. So the tests
are whether P*'s edge is honest and whether it prices a bet:

  1. Log loss at the open: P*, P and q_open on the same games; P* - q_open
     is the model's information beyond the opener.
  2. Claimed vs realised on P*'s side (realised / claimed ~ 1 if w is
     right; for the raw P it is ~1/3).
  3. Bets at the posted opening price on P*'s side, by P*'s claimed EV
     (P* x decimal odds - 1): realised ROI beside the claim, the ROI null
     (q x decimal - 1, minus the hold) and the same-game favourite. The
     rule "bet when the claimed EV > 0" has no tuned threshold; the
     blanket value side (every game, raw P's side) is the comparison.
  4. Line move: no-vig q from open to close on the bet side (closing-line
     value).

Each sportsbook book x season stands alone; games 10+ and 1-9 apart.
Research only: nothing here touches the ledgers, the pages, the model or
the frozen hypotheses. Writes diagnostics/open_shrinkage/RESULTS.txt.
± is 95%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import ledger                                     # noqa: E402
import market                                     # noqa: E402
import model_review as mr                         # noqa: E402

OUT = ROOT / "diagnostics" / "open_shrinkage"
Z95 = 1.96
EV_BANDS = ((-np.inf, 0.0, "EV* <= 0"), (0.0, 0.02, "0 to 2%"),
            (0.02, 0.05, "2 to 5%"), (0.05, np.inf, "5% and up"))


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def fit_w(p, q, y, iters=50):
    """w (and its ±95) maximising the likelihood of sigma(logit q + w*d),
    d = logit p - logit q: the share of the model's disagreement to keep."""
    lq, d = mr.logit(q), mr.logit(p) - mr.logit(q)
    y = np.asarray(y, float)
    w = 0.0
    for _ in range(iters):
        mu = sig(lq + w * d)
        g, h = float(((y - mu) * d).sum()), float((mu * (1 - mu) * d * d).sum())
        if h <= 0:
            break
        w += g / h
        if abs(g / h) < 1e-10:
            break
    mu = sig(lq + w * d)
    h = float((mu * (1 - mu) * d * d).sum())
    return w, (Z95 / np.sqrt(h) if h > 0 else float("nan"))


def shrink(p, q, w):
    return sig(mr.logit(q) + w * (mr.logit(p) - mr.logit(q)))


def frame(recon):
    """Graded rows with a usable opening pair, model P and result."""
    g = mr.frame(recon)
    for c in ("open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    ok = (g["q_open"].between(0, 1, inclusive="neither")
          & np.isfinite(g["open_home_ml"]) & np.isfinite(g["open_away_ml"]))
    return g[ok].copy()


def bets(g, p_col):
    """One bet per game on p_col's side at the posted opening price."""
    p, q = g[p_col].to_numpy(float), g["q_open"].to_numpy(float)
    home = p > q
    ml = np.where(home, g["open_home_ml"], g["open_away_ml"]).astype(float)
    ps = np.where(home, p, 1 - p)
    qs = np.where(home, q, 1 - q)
    qcs = np.where(home, g["q_close"], 1 - g["q_close"]).astype(float)
    y = g["y"].to_numpy(float)
    won = np.where(home, y == 1, y == 0)
    dec = np.array([market.decimal_payout(m) for m in ml])
    fav_home = q >= 0.5
    fav_ml = np.where(fav_home, g["open_home_ml"], g["open_away_ml"]).astype(float)
    fav_won = np.where(fav_home, y == 1, y == 0)
    return pd.DataFrame(dict(
        p=ps, q=qs, q_close=qcs, dec=dec, won=won.astype(int),
        units=[market.unit_profit(m, w) for m, w in zip(ml, won)],
        units_null=qs * dec - 1, ev=ps * dec - 1,
        fav_units=[market.unit_profit(m, w) for m, w in zip(fav_ml, fav_won)]))


def roi_line(label, b):
    n = len(b)
    if n < 20:
        return f"      {label:<22} n={n:<5} (too few)"
    u = b["units"].to_numpy(float)
    roi, se = u.mean(), u.std(ddof=1) / np.sqrt(n)
    null = b["units_null"].mean()
    move = 100 * (b["q_close"] - b["q"]).mean()
    return (f"      {label:<22} n={n:<5} claimed EV {100 * b['ev'].mean():+5.1f}%  "
            f"ROI {100 * roi:+5.1f}% ± {100 * Z95 * se:4.1f}  null {100 * null:+4.1f}%  "
            f"ROI - null {100 * (roi - null):+5.1f}  fav {100 * b['fav_units'].mean():+5.1f}%"
            f"  line move {move:+.2f} pp")


def block(label, test, train, w_note):
    y = test["y"].to_numpy(float)
    p, q, qc = (test[c].to_numpy(float) for c in ("p", "q_open", "q_close"))
    w, wc = fit_w(train["p"], train["q_open"], train["y"])
    w_in, w_inc = fit_w(p, q, y)
    ps = shrink(p, q, w)
    t = test.assign(p_star=ps)
    L = [f"  {label}: n={len(test)}   w fitted on {w_note}: {w:+.2f} ± {wc:.2f}"
         f"   (in-sample w {w_in:+.2f} ± {w_inc:.2f})"]
    ll = {k: mr.logloss(v, y) for k, v in (("P*", ps), ("P", p), ("open", q), ("close", qc))}
    L.append("    log loss  " + "  ".join(f"{k} {v.mean():.4f}" for k, v in ll.items()))
    for a, b in (("P*", "open"), ("P", "open"), ("P*", "P"), ("P*", "close")):
        m, ci = mr.mean_ci(ll[a] - ll[b])
        L.append(f"      {a:>2} - {b:<5} {m:+.4f} ± {ci:.4f}")
    for name, col in (("P*", "p_star"), ("P", "p")):
        bb = bets(t, col)
        claim = (bb["p"] - bb["q"]).mean()
        real = (bb["won"] - bb["q"]).mean()
        m, ci = mr.mean_ci(bb["won"] - bb["q"])
        L.append(f"    {name:<2} side: claimed {100 * claim:+.1f} pp  realised "
                 f"{100 * m:+.1f} ± {100 * ci:.1f} pp  share {real / claim:+.2f}")
    bb = bets(t, "p_star")
    L.append("    bets at the posted open, P*'s side, by P*'s claimed EV:")
    for lo, hi, name in EV_BANDS:
        L.append(roi_line(name, bb[(bb["ev"] > lo) & (bb["ev"] <= hi)]))
    L.append(roi_line("rule: EV* > 0", bb[bb["ev"] > 0]))
    L.append(roi_line("blanket value side", bets(t, "p")))
    return L


def report(recon):
    g = frame(recon)
    L = ["Disagreement shrinkage toward the opening price -- reconstructed rows, "
         "HINDSIGHT prices (the open is a real price; the rows are reconstructed). "
         "± is 95%.",
         "P* = sigma(logit q_open + w (logit P - logit q_open)), w fitted on the other "
         "priced season(s): 2025-26 rows are a walk-forward test, 2024-25 rows "
         "hindsight.", ""]
    for route, r in (("games 10+", g[~g["early"]]), ("games 1-9", g[g["early"]])):
        L.append(f"== {route}")
        for label, x in mr.groups(r):
            season = x["season"].iloc[0]
            train = r[r["season"] != season]
            if len(x) < 50 or len(train) < 50:
                continue
            note = ", ".join(sorted({mr.season_name(s) for s in train["season"]}))
            note += " (walk-forward)" if train["season"].max() < season else " (hindsight)"
            L += block(label, x, train, note)
        L.append("")
    return "\n".join(L)


def main():
    text = report(ledger.load(ledger.RECON_PATH))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
