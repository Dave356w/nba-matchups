#!/usr/bin/env python3
"""Native line moves toward / against the model, first snapshot to close.

  python research/native_line_moves.py

Native rows (data/nba_ledger.csv) with a frozen first-snapshot P. A sanity
check, not a rule or a test: no thresholds, no selection, nothing written
back to the ledger. Games without a close use the latest pregame line and
are counted apart in the graded line.

  1. Model checks: p_home = Phi(margin_home / sigma) on v9 rows; first P to
     latest P drift by first route; model vs market home margin, spread and
     logit slope (compression); moneyline value side vs H5 side agreement.
  2. Moves signed toward the model: moneyline no-vig q on the value side
     (sign of first P - first q), the spread on H5's side (13.5 x
     Phi^-1(first P) + first spread, analysis.ATS_SIGMA), and the lean's q.
  3. Graded outcomes split by move direction (descriptive; tiny samples).
  4. Overnight lean CLV by edge floor: the lean (first P >= 0.5) where it is
     also the value side, first P - first q above each floor, on closed
     games; by first_model_tag version, v8 (before v9, in no active row)
     apart from v9 or later; the value side on the model's underdog beside
     it. The floors were picked after looking at the data: not a rule.
  5. Close ties: how often the pregame and 5 h lock prices already equal the
     close (CLV is then 0 by tie, not by skill), and Kalshi close coverage.

Games 1-9 and 10+ are shown apart; game 0 is in neither. +/- is 1 SE.
Writes diagnostics/native_line_moves/RESULTS.txt.
"""
import json
import os
import sys
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis  # noqa: E402
import ledger  # noqa: E402

OUT = Path("diagnostics") / "native_line_moves"
ND = NormalDist()
NUM = ["first_p_home", "first_q_home", "pre_q_home", "close_q_home", "p_home",
       "margin_home", "first_spread", "pre_spread", "close_spread",
       "home_pts", "away_pts", "home_won", "gp_home", "gp_away"]
FLOORS = (0.0, 0.01, 0.02, 0.03, 0.05)      # post hoc, descriptive only
TIES = (("pregame", "pre", "snapshot_utc"), ("5 h lock", "lock", "lock_snapshot_utc"))


def version(tag):
    """'v9' from a model tag ending _v9; '' when there is no tag."""
    s = str(tag).rsplit("_", 1)[-1]
    return s if s.startswith("v") else ""


def sigma(path):
    return json.load(open(path))["sigma"]


def moves(df):
    d = df.copy()
    d["ml_dir"] = np.sign(d.first_p_home - d.first_q_home)
    d["ats_edge"] = (analysis.ATS_SIGMA * d.first_p_home.map(ND.inv_cdf)
                     + d.first_spread)          # + = the model likes home ATS
    d["end_q"] = d.close_q_home.fillna(d.pre_q_home)
    d["end_sp"] = d.close_spread.fillna(d.pre_spread)
    d["ml_mv"] = 100 * d.ml_dir * (d.end_q - d.first_q_home)            # pp
    d["sp_mv"] = np.sign(d.ats_edge) * (d.first_spread - d.end_sp)      # pts
    d["lean_mv"] = 100 * np.where(d.first_p_home >= .5, 1, -1) * (d.end_q - d.first_q_home)
    d["lean_edge"] = np.where(d.first_p_home >= .5, 1, -1) * (d.first_p_home - d.first_q_home)
    d["ver"] = d.first_model_tag.map(version)
    return d


def floors(g, L):
    """Section 4 for one stage: closed games, lean CLV vs the close."""
    c = g[g.close_q_home.notna()]
    v9 = c.first_model_tag.astype(str).str.endswith(analysis.EARLY_TAG_SUFFIXES)
    L.append("    overnight lean CLV, lean is the value side (closed; floors post hoc):")
    for t in FLOORS:
        m = c.lean_edge > t
        L.append(line(f"  P - q > {100 * t:.0f}pp, all tags", c.lean_mv[m], "pp"))
        L.append(line("    first tag v9 or later", c.lean_mv[m & v9], "pp"))
        L.append(line(f"    first tag v8 (pre-v9)", c.lean_mv[m & (c.ver == "v8")], "pp"))
    m = c.lean_edge > 0.03
    fav = np.where(c.first_p_home >= .5, c.first_q_home, 1 - c.first_q_home) > .5
    L.append(line("  > 3pp, v9+, market fav", c.lean_mv[m & v9 & fav], "pp"))
    L.append(line("  > 3pp, v9+, market dog", c.lean_mv[m & v9 & ~fav], "pp"))
    m = c.lean_edge < -0.03
    L.append(line("  value on model dog > 3pp", -c.lean_mv[m], "pp"))
    L.append(line("    first tag v9 or later", -c.lean_mv[m & v9], "pp"))
    big = c.lean_edge.abs() > 0.03
    L.append(line("  lean, |P - q| > 3pp", c.lean_mv[big], "pp"))
    L.append(line("  lean, |P - q| <= 3pp", c.lean_mv[~big], "pp"))


def ties(df, L):
    """Section 5: each price's ties with the same book's close."""
    for c in ("pre_home_ml", "pre_away_ml", "close_home_ml", "close_away_ml", "lock_home_ml",
              "lock_away_ml", "lock_spread", "lock_home_spread_odds", "lock_away_spread_odds",
              "pre_home_spread_odds", "pre_away_spread_odds", "close_home_spread_odds",
              "close_away_spread_odds", "pre_spread", "close_spread", "kalshi_close_q_home",
              "first_p_home"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    tip = pd.to_datetime(df.tip_utc, utc=True, errors="coerce")
    for name, p, ts in TIES:
        g = df[df[f"{p}_home_ml"].notna() & df.close_home_ml.notna()
               & (df[f"{p}_book"].astype(str) == df.close_book.astype(str))]
        if not len(g):
            L.append(f"  {name:<9} no rows with a same-book close")
            continue
        h = (tip[g.index] - pd.to_datetime(g[ts], utc=True, errors="coerce")).dt.total_seconds() / 3600
        ml = (g[f"{p}_home_ml"] == g.close_home_ml) & (g[f"{p}_away_ml"] == g.close_away_ml)
        sp = g[f"{p}_spread"].notna() & g.close_spread.notna()
        line_eq = sp & (g[f"{p}_spread"] == g.close_spread)
        all_eq = line_eq & (g[f"{p}_home_spread_odds"] == g.close_home_spread_odds) \
            & (g[f"{p}_away_spread_odds"] == g.close_away_spread_odds)
        L.append(f"  {name:<9} n={len(g):<3} before tip median {h.median():.2f} h, min "
                 f"{60 * h.min():.1f} min; moneyline = close {ml.sum()}/{len(g)}; "
                 f"spread line = close {line_eq.sum()}/{sp.sum()} (prices too {all_eq.sum()})")
    k = df[df.kalshi_close_q_home.notna()]
    L.append(f"  Kalshi close: {len(k)} rows (slates {k.slate_date.min()}..{k.slate_date.max()}), "
             f"{k.first_p_home.notna().sum()} with a first P; no new Kalshi writes, so there is "
             "no second-market close for later bets." if len(k) else
             "  Kalshi close: no rows.")


def line(label, x, unit):
    x = pd.Series(x).dropna()
    n = len(x)
    if not n:
        return f"    {label:<26} no rows"
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return (f"    {label:<26} n={n:<3} {x.mean():+6.2f}{unit} +/- {se:4.2f}   "
            f"toward {(x > 1e-9).sum()} / against {(x < -1e-9).sum()} / flat {(x.abs() <= 1e-9).sum()}")


def model_checks(d, L):
    sb, sa = sigma("model/logit.json"), sigma("model/logit_avail.json")
    v9 = d[d.model_tag.str.contains("v9", na=False) & d.margin_home.notna()]
    s = np.where(v9.model_tag.str.contains("avail"), sa, sb)
    p = [ND.cdf(m / x) for m, x in zip(v9.margin_home, s)]
    L.append(f"  p_home vs Phi(margin_home / sigma): max |diff| {np.max(np.abs(v9.p_home - p)):.5f} "
             f"on {len(v9)} v9 rows (sigma {sb:.2f} base, {sa:.2f} avail)")
    dr = (d.p_home - d.first_p_home).abs()
    for r, g in dr.groupby(d.first_route.fillna("none")):
        L.append(f"  first -> latest P drift, first route {r:<5}: mean |dP| {100 * g.mean():.1f}pp, "
                 f"max {100 * g.max():.1f}pp (n={len(g)})")
    mm = analysis.ATS_SIGMA * d.first_p_home.map(ND.inv_cdf)
    ok = d.first_spread.notna()
    L.append(f"  home margin: model {mm.mean():+.2f} vs market (-first spread) {-d.first_spread.mean():+.2f}; "
             f"corr {np.corrcoef(mm[ok], -d.first_spread[ok])[0, 1]:.2f}")
    lg = lambda x: np.log(x / (1 - x))  # noqa: E731
    b = np.polyfit(lg(d.first_q_home), lg(d.first_p_home), 1)
    L.append(f"  P std {d.first_p_home.std():.3f} vs q std {d.first_q_home.std():.3f}; "
             f"logit(P) on logit(q) slope {b[0]:.2f} (1 = same spread as the market)")
    vq = np.where(d.ml_dir > 0, d.first_q_home, 1 - d.first_q_home)
    L.append(f"  moneyline value side is the underdog in {(vq < .5).sum()} / {len(d)}")
    dis = ok & (d.ml_dir != np.sign(d.ats_edge))
    L.append(f"  moneyline value side and H5 side disagree in {dis.sum()} game(s): "
             + ", ".join(d[dis].away + " @ " + d[dis].home + " " + d[dis].slate_date))


def report(df):
    d = df[df.first_p_home.notna() & df.first_q_home.notna()].copy()
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = moves(d)
    gp = d[["gp_home", "gp_away"]].min(axis=1)
    L = ["Native line moves, first snapshot -> close (latest pregame if not yet closed): DESCRIPTIVE",
         f"Source: {ledger.NATIVE_PATH}; {len(d)} rows with a first P, {d.close_q_home.notna().sum()} closed, "
         f"slates {d.slate_date.min()}..{d.slate_date.max()}. First tags: "
         + ", ".join(f"{t.rsplit('_', 2)[-2] + '_' + t.rsplit('_', 1)[-1]} {n}"
                     for t, n in d.first_model_tag.value_counts().items()),
         "Toward = the line moved to the model's side. +/- 1 SE. Not a rule, not forward evidence of ROI.",
         "", "== Model checks (all rows)"]
    model_checks(d, L)
    for stage, m in (("Games 1-9", gp.between(1, 9)), ("Games 10+", gp >= 10)):
        g = d[m]
        L += ["", f"== {stage}: {len(g)} rows"]
        if not len(g):
            continue
        closed = g.close_q_home.notna()
        L.append(line("moneyline value side q", g.ml_mv, "pp"))
        L.append(line("  closed games only", g.ml_mv[closed], "pp"))
        for t in sorted(set(g.ver) - {""}, key=lambda v: int(v[1:])):
            L.append(line(f"  first tag {t}", g.ml_mv[g.ver == t], "pp"))
        for r, gg in g.groupby(g.first_route.fillna("none")):
            L.append(line(f"  first route {r}", gg.ml_mv, "pp"))
        L.append(line("spread, H5 side", g.sp_mv[g.first_spread.notna()], "pt"))
        for lo, hi in ((0, 1), (1, 3), (3, 99)):
            L.append(line(f"  |edge| {lo}-{hi} pt", g.sp_mv[g.ats_edge.abs().between(lo, hi, inclusive="left")], "pt"))
        L.append(line("lean (model favourite) q", g.lean_mv, "pp"))
        floors(g, L)
        L.append(f"    corr(P - q, home q move) {np.corrcoef(g.first_p_home - g.first_q_home, g.end_q - g.first_q_home)[0, 1]:+.2f}; "
                 f"corr(ATS edge, home line move) {np.corrcoef(g.ats_edge[g.first_spread.notna()], (g.first_spread - g.end_sp)[g.first_spread.notna()])[0, 1]:+.2f}")
        gd = g[g.home_won.notna() & g.close_q_home.notna()]
        won = np.where(gd.ml_dir > 0, gd.home_won == 1, gd.home_won == 0)
        cov = np.sign(gd.home_pts - gd.away_pts + gd.first_spread) * np.sign(gd.ats_edge)
        L.append("    graded, by move direction (descriptive):")
        for lab, mm, ms in (("toward", gd.ml_mv > 1e-9, gd.sp_mv > 0),
                            ("against", gd.ml_mv < -1e-9, gd.sp_mv < 0),
                            ("flat", gd.ml_mv.abs() <= 1e-9, gd.sp_mv == 0)):
            L.append(f"      {lab:<8} value side won {won[mm].sum()}/{mm.sum()}   "
                     f"H5 side vs first line {(cov[ms] > 0).sum()}-{(cov[ms] < 0).sum()}-{(cov[ms] == 0).sum()}")
    L += ["", "== Close ties (all native rows; same book)"]
    ties(df.copy(), L)
    L += ["", "== Rows: date  matchup  first P  q first -> end  ATS edge  spread first -> end  moves (ml pp, spread pt)  first tag"]
    for _, r in d.sort_values(["slate_date", "game_id"]).iterrows():
        L.append(f"  {r.slate_date} {r.away}@{r.home:<4} P {r.first_p_home:.3f}  q {r.first_q_home:.3f} -> {r.end_q:.3f}  "
                 f"edge {r.ats_edge:+5.1f}  {r.first_spread:+5.1f} -> {r.end_sp:+5.1f}  "
                 f"{r.ml_mv:+5.1f} {r.sp_mv:+4.1f}  {r.first_model_tag.rsplit('_', 1)[-1]}"
                 f"{'' if pd.notna(r.close_q_home) else '  (pending)'}")
    return "\n".join(L)


def main():
    text = report(ledger.load(ledger.NATIVE_PATH))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
