#!/usr/bin/env python3
"""NFL-style median spread bands on v10's margin: a retrospective ATS test.

  python research/median_spread_band.py

Owner's question (2026-10-28): would the NFL model's production spread map
(nfl-model fixed_bands.py, "median-mov-ats-2026-10-17-v1") help here? It
groups games by the model's absolute predicted margin, replaces each
prediction by the earlier seasons' median signed final margin of that band
(the pick side's margin; losses negative, ties zero), and bets the mapped
spread's side of the market line, taking the market favourite when the
mapped spread equals the line (a pick'em line with a zero map skips).

Fixed before the run:
  forecast   v10's walk-forward margin (diagnostics/v10_candidate,
             predictions_v10h25_*.csv, m_D: every fit on earlier seasons,
             production routing, the per-slate report refit).
  bands      quartile cuts of |margin| on the training seasons (primary;
             count-balanced like the NFL's four bands), and the NFL's literal
             cuts 3 / 7 / 10 points (secondary).
  map        per test season Y, medians from seasons before Y (at least two
             seasons and 30 games a band, as the NFL rule requires).
  selection  the NFL rule above, on every game with main-line spread prices
             (both sides -135..+115, analysis.ATS_MAIN_LINE).
  grading    each book x season x price alone: DraftKings 2025-26 and ESPN
             BET 2024-25, games 10+, at the open (ats_signal/open_spreads.csv)
             and at the close, at the recorded spread prices. ROI against its
             no-vig null, +/- 1 SE; the same-game spread favourite.
  benchmarks the same games: v10's raw margin side (no map), and H5's
             registered rule (13.5 x Phi^-1(P), edge >= 1 point).
Research only: no new rule, record or page. A tracked version would need a
registered amendment (CLAUDE.md keeps exactly three forward tests).
"""
from __future__ import annotations

import glob
import os
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import analysis  # noqa: E402
import market  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "median_spread_band")
PRED = os.path.join(ROOT, "diagnostics", "v10_candidate", "predictions_v10h25_*.csv")
NFL_CUTS = [3.0, 7.0, 10.0]
EPS = 1e-10
MIN_SEASONS, MIN_BAND = 2, 30
CELLS = (("dk", 2026, "DraftKings 2025-26"), ("espnbet", 2025, "ESPN BET 2024-25"))


def load():
    p = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(PRED))], ignore_index=True)
    p = p.drop_duplicates(["season", "slate_date", "home", "away"], keep="last")
    return p.rename(columns={"m_D": "m", "p_D": "p"})[
        ["season", "slate_date", "home", "away", "gp_home", "gp_away", "stage", "margin", "m", "p"]]


def fit(train, cuts):
    """Band cuts and medians of the pick side's final margin, or None."""
    t = train[np.isfinite(train["m"]) & np.isfinite(train["margin"]) & (train["m"] != 0)]
    if t["season"].nunique() < MIN_SEASONS:
        return None
    a = t["m"].abs().to_numpy()
    cuts = list(np.quantile(a, [0.25, 0.5, 0.75])) if cuts == "quartile" else NFL_CUTS
    band = np.digitize(a, cuts)
    counts = np.bincount(band, minlength=len(cuts) + 1)
    if counts.min() < MIN_BAND:
        return None
    signed = np.sign(t["m"].to_numpy()) * t["margin"].to_numpy()
    return dict(cuts=[float(c) for c in cuts], counts=counts.tolist(),
                medians=[float(np.median(signed[band == k])) for k in range(len(cuts) + 1)])


def apply(m, rule):
    m = np.asarray(m, float)
    return np.sign(m) * np.asarray(rule["medians"])[np.digitize(np.abs(m), rule["cuts"])]


def nfl_side(pred, home_line):
    """+1 home, -1 away, 0 skip: the mapped margin against the market's
    (minus the home line); zero edge takes the favourite, a pick'em skips."""
    edge = np.asarray(pred, float) + np.asarray(home_line, float)
    fav = -np.sign(np.asarray(home_line, float))
    return np.where(np.abs(edge) <= EPS, fav, np.sign(edge))


def grade(g, side, label):
    """Flat 1u ATS on `side` (+1 home / -1 away / 0 skip) at g's line and prices."""
    side = np.asarray(side, float)
    keep = side != 0
    g, side = g[keep], side[keep]
    line, hso, aso = (g[c].to_numpy(float) for c in ("line", "hso", "aso"))
    res_home = np.array([market.ats_result(m, x) for m, x in zip(g["margin"], line)])
    home = side > 0
    res = np.where(home, res_home, 1 - res_home)
    ml = np.where(home, hso, aso)
    qh = np.array([market.devig(h, a) for h, a in zip(hso, aso)])
    q = np.where(home, qh, 1 - qh)
    pay = np.array([market.decimal_payout(x) for x in ml])
    units = np.where(res == 0.5, 0.0, np.where(res == 1, pay - 1, -1.0))
    null = q * pay - 1
    fav_home = np.where(line < 0, True, np.where(line > 0, False, hso <= aso))
    fres = np.where(fav_home, res_home, 1 - res_home)
    fpay = np.array([market.decimal_payout(x) for x in np.where(fav_home, hso, aso)])
    fav = np.where(fres == 0.5, 0.0, np.where(fres == 1, fpay - 1, -1.0))
    n = len(units)
    se = units.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    w, lo = int((res == 1).sum()), int((res == 0).sum())
    roi, nu = units.mean(), null.mean()
    return (f"    {label:28s} n={n:4d}  {w}-{lo}-{n - w - lo}  units {units.sum():+7.2f}  "
            f"ROI {100 * roi:+6.1f}% ± {100 * se:4.1f}  (null {100 * nu:+.1f}%, "
            f"z {(roi - nu) / se:+.1f})  cover {100 * w / max(w + lo, 1):.1f}%  "
            f"| fav {100 * fav.mean():+.1f}%")


def market_rows(book, season, price):
    r = pd.read_csv(os.path.join(ROOT, "data", "nba_reconstructed.csv"), dtype={"game_id": str})
    r = r[(r["close_book"] == book) & (r["season"] == season)]
    if price == "open":
        r = analysis.with_open_spreads(r)
    cols = {c: f"{price}_{c}" for c in ("spread", "home_spread_odds", "away_spread_odds")}
    r = r.rename(columns={cols["spread"]: "line", cols["home_spread_odds"]: "hso",
                          cols["away_spread_odds"]: "aso"})
    r = r[r["line"].notna() & r["hso"].between(*analysis.ATS_MAIN_LINE)
          & r["aso"].between(*analysis.ATS_MAIN_LINE)]
    return r[["slate_date", "home", "away", "line", "hso", "aso"]]


def main():
    P = load()
    L = [__doc__.strip().splitlines()[0], ""]
    nd = NormalDist()
    for book, season, name in CELLS:
        rules = {c: fit(P[P["season"] < season], c) for c in ("quartile", "nfl")}
        L.append(f"== {name}, games 10+ (map fitted on seasons {int(P.season.min())}-"
                 f"{season - 1}; walk-forward v10 margin)")
        for c, rule in rules.items():
            L.append(f"  {c:8s} cuts {', '.join(f'{x:.2f}' for x in rule['cuts'])}  "
                     f"band n {rule['counts']}  medians "
                     f"{', '.join(f'{x:+.1f}' for x in rule['medians'])}")
        test = P[(P["season"] == season) & (P["stage"] == "games 10+")]
        for price in ("open", "close"):
            g = test.merge(market_rows(book, season, price), on=["slate_date", "home", "away"])
            L.append(f"  at the {price} ({len(g)} games with main-line prices):")
            for c, rule in rules.items():
                L.append(grade(g, nfl_side(apply(g["m"], rule), g["line"]), f"median band ({c})"))
            L.append(grade(g, nfl_side(g["m"], g["line"]), "v10 margin, no map"))
            h5 = 13.5 * np.array([nd.inv_cdf(x) for x in g["p"]])
            e = h5 + g["line"].to_numpy()
            L.append(grade(g, np.where(np.abs(e) >= 1, np.sign(e), 0), "H5 rule (13.5, >= 1 pt)"))
        L.append("")
    L += ["Hindsight on reconstructed prices; every map fitted on earlier seasons only. "
          "+/- is 1 SE; z = (ROI - null) / SE. Not forward evidence; no rule is registered."]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "RESULTS.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
