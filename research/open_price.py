#!/usr/bin/env python3
"""Open vs close: is the model's opening-price signal real, or injury timing?

  python research/open_price.py --seasons 2025 2026 [--cache research/output]

At the close the model adds nothing to the market; at the OPEN its value
side gains ~1.3-1.8 pp of no-vig probability by the close and the value-side
ROI is near break-even. But the v4 availability terms use the injury report
>= 30 min before tip, which did not exist when the line opened, so part of
that "signal" may be news the model saw early. This separates the two:

  v4     the shipped routing, walk-forward (backfill_history.
         reconstruct_season(walk_forward=True), availability on covered games)
  base   the same routing with NO injury-report terms (base v4 on games 10+):
         every input is known at the open. Games 1-9 are identical in both.

For each season and closing book, games 10+ and games 1-9 separately, and
each arm at the open and at the close price (the book that supplied the
close, `close_book`):
  * log loss, model vs that price's no-vig q (same games);
  * w: the outcome's weight on logit(P) - logit(q) beside logit(q)
    (0 = the model adds nothing to the price; ~0.27 is about what a value
    bet needs to clear a ~4% hold);
  * value side (P > q): flat 1u ROI beside its null (mean q/breakeven - 1,
    the ROI if q is right; about minus the hold, not zero);
  * CLV: the value side's no-vig probability, open -> close.

Pre-registered cells (CLAUDE.md, fixed 2026-09-30 before any native rows):
  H1  games 1-9, value side with P - q >= 0.08, at the close;
  H2  games 10+, value side that is the favourite, at the open;
  H3  games 10+, value side with P - q >= 0.12, at the open.
H2/H3 were profitable at the open only; if `base` loses them, they were
injury timing. Research only; per-game output in research/output/open_price.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backfill_history as bf  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                   "open_price.csv")
ARMS = ("v4", "base")
PRICES = ("open", "close")
CELLS = (("H1 games 1-9, edge >= 8pp", "close", True, "edge", 0.08),
         ("H2 games 10+, value favourite", "open", False, "fav", None),
         ("H3 games 10+, edge >= 12pp", "open", False, "edge", 0.12))


def home_q(m, price):
    """No-vig home probability at `price` ('open' or 'close')."""
    return np.array([market.devig(h, a) for h, a in
                     zip(m[f"{price}_home_ml"], m[f"{price}_away_ml"])])


def value_bets(m, pcol, price):
    """One value-side bet per game at `price` ('open' or 'close'): the side
    with model P above that price's no-vig q. Rows without a valid pair drop."""
    hm, am = m[f"{price}_home_ml"].to_numpy(float), m[f"{price}_away_ml"].to_numpy(float)
    q = home_q(m, price)
    p = m[pcol].to_numpy(float)
    ok = np.isfinite(q) & np.isfinite(p)
    home = p > q
    ml = np.where(home, hm, am)
    y = np.where(home, m["home_won"], 1 - m["home_won"]).astype(float)
    qo = home_q(m, "open")
    qc = m["close_q_home"].to_numpy(float)
    out = pd.DataFrame(dict(
        P=np.where(home, p, 1 - p), q=np.where(home, q, 1 - q), ml=ml,
        be=market.breakeven_prob(ml), y=y,
        pl=[market.unit_profit(x, w == 1) for x, w in zip(ml, y)],
        move=np.where(home, 1, -1) * (qc - qo),
        early=m["early"].to_numpy(bool)), index=m.index)
    out["edge"] = out["P"] - out["q"]
    out["fav"] = out["ml"] < 0
    return out[ok]


def roi(b):
    """n, ROI, 95% half-width, null (mean q/breakeven - 1) and z vs the null."""
    n = len(b)
    if n < 2:
        return dict(n=n, roi=np.nan, ci=np.nan, null=np.nan, z=np.nan)
    se = b["pl"].std(ddof=1) / np.sqrt(n)
    null = float((b["q"] / b["be"]).mean() - 1)
    r = float(b["pl"].mean())
    return dict(n=n, roi=r, ci=1.96 * se, null=null,
                z=(r - null) / se if se > 0 else np.nan)


def weight(p, q, y):
    """w, ±95%: y ~ a + b*logit(q) + w*(logit(p) - logit(q))."""
    p, q = np.clip(p, 1e-4, 1 - 1e-4), np.clip(q, 1e-4, 1 - 1e-4)
    lq = np.log(q / (1 - q))
    X = np.column_stack([lq, np.log(p / (1 - p)) - lq])
    if len(y) < 30:
        return np.nan, np.nan
    try:
        fm = nc.fit_logit(X, y, l2=0.0)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.array([fm["intercept"]] + fm["coef"])
    pr = 1 / (1 + np.exp(-X1 @ b))
    cov = np.linalg.inv(X1.T @ (X1 * (pr * (1 - pr))[:, None]))
    return float(b[2]), float(1.96 * np.sqrt(cov[2, 2]))


def cell(b, kind, thr):
    if kind == "edge":
        return b[b["edge"] >= thr]
    return b[b["fav"]]


def report(m, arms=ARMS):
    lines = []
    for (season, book), g in m.groupby(["year", "close_book"]):
        lines.append(f"\n{season} {market.BOOK_NAMES.get(book, book)} close book")
        for early in (False, True):
            ge = g[g["early"] == early]
            if len(ge) < 20:
                continue
            lines.append(f"  {'games 1-9' if early else 'games 10+'} (n={len(ge)})")
            for arm in arms:
                if early and arm != arms[0]:
                    continue                 # identical routing in games 1-9
                for price in PRICES:
                    b = value_bets(ge, f"p_{arm}", price)
                    if not len(b):
                        continue
                    gb = ge.loc[b.index]
                    qh = home_q(gb, price)
                    y = gb["home_won"].to_numpy(float)
                    pm = gb[f"p_{arm}"].to_numpy(float)
                    d_ll = float((market.logloss(pm, y) - market.logloss(qh, y)).mean())
                    w, wci = weight(pm, qh, y)
                    r = roi(b)
                    mv = b["move"].to_numpy(float)
                    clv = (f"  CLV {100 * mv.mean():+.2f}pp ± "
                           f"{196 * mv.std(ddof=1) / np.sqrt(len(mv)):.2f}"
                           if price == "open" else "")
                    lines.append(
                        f"    {arm:4s} @{price:5s} logloss−mkt {d_ll:+.4f}  "
                        f"w {w:+.2f} ± {wci:.2f}  value ROI {r['roi']:+.3f} ± "
                        f"{r['ci']:.3f} (null {r['null']:+.3f}, n={r['n']}){clv}")
    lines.append("\n== Pre-registered cells, pooled over seasons and books "
                 "(ROI beside its null)")
    for label, price, early, kind, thr in CELLS:
        for arm in arms:
            if early and arm != arms[0]:
                continue
            b = cell(value_bets(m[m["early"] == early], f"p_{arm}", price), kind, thr)
            r = roi(b)
            lines.append(f"  {label:32s} {arm:4s} @{price:5s} n={r['n']:4d}  ROI "
                         f"{r['roi']:+.3f} ± {r['ci']:.3f}  null {r['null']:+.3f}  "
                         f"z {r['z']:+.2f}")
    return "\n".join(lines)


def score(y, terms):
    """v4 (with terms) and base (without) walk-forward p_home for season y."""
    out = {}
    for arm, t in (("v4", terms), ("base", None)):
        test, _, _ = bf.reconstruct_season(y, terms=t, walk_forward=True)
        test["slate_date"] = pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d")
        out[arm] = (test.drop_duplicates(["slate_date", "home", "away"])
                    .set_index(["slate_date", "home", "away"]))
    f = out["v4"][["year", "win", "route"]].copy()
    f["p_v4"] = out["v4"]["p_home"]
    f["p_base"] = out["base"]["p_home"].reindex(f.index)
    f["early"] = f["route"] == "early"
    return f.reset_index()


def attach(scored, recon):
    r = ledger.graded(recon)
    r = r[["slate_date", "home", "away", "home_won", "close_book", "close_q_home",
           "open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml"]]
    return scored.merge(r, on=["slate_date", "home", "away"], how="inner")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--cache", default=pav.DEFAULT_CACHE)
    a = ap.parse_args(argv)
    archive = pav.ReportArchive(os.path.join(a.cache, "injury_reports"))
    terms = {}
    try:
        for t in [t for t in bf.REPORT_YEARS if t <= max(a.seasons)]:
            terms[t], s = pav.season_terms(t, archive, a.cache)
            print(f"report season {t}: {len(s['covered'])}/{s['games']} games covered",
                  flush=True)
    finally:
        archive.save()
    recon = ledger.load(ledger.RECON_PATH)
    m = pd.concat([attach(score(y, terms), recon) for y in a.seasons],
                  ignore_index=True)
    print(f"\n{len(m)} games matched to reconstructed rows (open and close prices)")
    print("\n== Value side, same games; negative log loss diff = model better")
    print(report(m))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    m.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(m)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
