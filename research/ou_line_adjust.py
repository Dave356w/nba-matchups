#!/usr/bin/env python3
"""Adjusted closing totals: a grid of  total' = close + a + b x matchup,  walk-forward.

  python research/ou_line_adjust.py      # writes diagnostics/ou_market_residual/RESULTS_line_adjust.txt

Owner's request (2026-11-05), after the post-hoc totals-matchup checks: test
different iterations of the line adjustment. Research only, and the target
is the total against the closing line, which G9 excludes for the production
model. The grid, the baseline and the multiplicity-adjusted rule were
committed before the run.

Data. Games with a closing spread and closing total from the same book
(ESPN BET 2024-25, ESPN BET 2025-26 to 25 Nov, DraftKings 2025-26, native
DraftKings 2026-27; spreads give the implied team points, so 2023-24 is
out). Implied home points (total - spread) / 2, away (total + spread) / 2.
Per team and game, capped at +-CAP: sc = points scored - implied, al =
points allowed - the opponent's implied. Team means are this season's
priced games strictly before the slate, decayed at half-life HL games and
shrunk to 0 with 10 game weights.

Matchup terms:
  sum    m = sc_h + al_a + sc_a + al_h                     (one term)
  split  m1 = sc_h + sc_a (offences), m2 = al_h + al_a     (two terms)
Intercept:
  none   a = 0
  fit    a fitted with b
  roll   a = mean of (total - close) over the last 300 priced games before
         the slate; b fitted on what remains, through the origin
Half-life HL in {5, 25}; cap CAP in {10, 15}. 3 x 2 x 2 x 2 = 24 variants.
Each refitted by least squares before every slate on the reconstructed
priced games before it (expanding; native rows never in a fit); scoring
starts once 500 training games exist.

Baseline: the closing total itself (adjustment 0). Readings per book x
season (scored: ESPN BET 2024-25, DraftKings 2025-26; information only:
ESPN BET 2025-26, DraftKings 2026-27): squared error of the adjusted total
minus the closing total's, paired; the side the adjustment favours when
|adjustment| >= 0.5, at closing over/under prices (pushes out): n, win
rate, ROI and the no-vig null; the share of overs.

Decision rule, set before the run, with 24 variants: a variant is a
candidate only if
  1. its squared error beats the closing total in both scored seasons and
     the two seasons' combined interval at z = 3.08 (95% Bonferroni over 24)
     excludes 0;
  2. its |adjustment| >= 0.5 side has ROI above its null in both scored
     seasons.
Also reported: how many variants beat the close on squared error in both
scored seasons (about a quarter would by chance). Research only: nothing
here touches the ledgers, pages or model/*.json.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "diagnostics", "ou_market_residual")
SHRINK, MIN_TRAIN, ROLL, EDGE, Z_MULTI = 10.0, 500, 300, 0.5, 3.08
SCORED = [(2025, "espnbet"), (2026, "dk")]
INFO = [(2026, "espnbet"), (2027, "dk")]
GRID = list(itertools.product(("none", "fit", "roll"), ("sum", "split"), (5, 25), (10, 15)))


def name(v):
    return f"{v[0]}-{v[1]}-hl{v[2]}-cap{v[3]}"


def games():
    r = pd.read_csv(os.path.join(ROOT, "data", "nba_reconstructed.csv")).assign(basis="reconstructed")
    led = pd.read_csv(os.path.join(ROOT, "data", "nba_ledger.csv")).assign(
        basis="native", close_book="dk")
    cols = ["game_id", "basis", "season", "slate_date", "home", "away", "home_pts",
            "away_pts", "close_book", "close_spread"]
    G = pd.concat([r[cols], led[cols]]).drop_duplicates("game_id")
    o = pd.read_csv(os.path.join(ROOT, "diagnostics", "totals_signal", "totals_odds.csv"))
    G = G.merge(o[["game_id", "book", "close_total", "close_over", "close_under"]],
                on="game_id", how="left")
    lt = led.set_index("game_id")
    for c, lc in (("close_total", "close_ou_line"), ("close_over", "close_ou_over"),
                  ("close_under", "close_ou_under")):
        G[c] = G[c].fillna(G["game_id"].map(lt[lc]))
    G = G[G["book"].isna() | (G["book"] == G["close_book"])]
    G = G.dropna(subset=["close_spread", "close_total", "home_pts", "away_pts"])
    G["imp_h"] = (G["close_total"] - G["close_spread"]) / 2
    G["imp_a"] = (G["close_total"] + G["close_spread"]) / 2
    G["tres"] = G["home_pts"] + G["away_pts"] - G["close_total"]
    return G.sort_values(["season", "slate_date", "game_id"]).reset_index(drop=True)


def team_mean(h, hl):
    if not h:
        return 0.0
    v = np.array(h, float)
    w = 0.5 ** (np.arange(len(v))[::-1] / hl)
    return float((w * v).sum() / (w.sum() + SHRINK))


def features(G):
    """m_sum, m_off, m_def for every (hl, cap)."""
    G = G.copy()
    for hl, cap in itertools.product((5, 25), (10, 15)):
        for k in ("sum", "off", "def"):
            G[f"m_{k}_{hl}_{cap}"] = 0.0
    for _, g in G.groupby("season"):
        hist = {}
        for _, idx in g.groupby("slate_date").groups.items():
            for j in idx:
                th = hist.get(G.at[j, "home"], {})
                ta = hist.get(G.at[j, "away"], {})
                for hl, cap in itertools.product((5, 25), (10, 15)):
                    def m(t, k):
                        return team_mean(t.get((k, cap), []), hl)
                    off = m(th, "sc") + m(ta, "sc")
                    de = m(th, "al") + m(ta, "al")
                    G.at[j, f"m_off_{hl}_{cap}"] = off
                    G.at[j, f"m_def_{hl}_{cap}"] = de
                    G.at[j, f"m_sum_{hl}_{cap}"] = off + de
            for j in idx:
                h, a = G.at[j, "home"], G.at[j, "away"]
                sh = G.at[j, "home_pts"] - G.at[j, "imp_h"]
                sa = G.at[j, "away_pts"] - G.at[j, "imp_a"]
                for t, sc, al in ((h, sh, sa), (a, sa, sh)):
                    d = hist.setdefault(t, {})
                    for cap in (10, 15):
                        d.setdefault(("sc", cap), []).append(float(np.clip(sc, -cap, cap)))
                        d.setdefault(("al", cap), []).append(float(np.clip(al, -cap, cap)))
    return G


def cols(v):
    _, kind, hl, cap = v
    return [f"m_sum_{hl}_{cap}"] if kind == "sum" else [f"m_off_{hl}_{cap}", f"m_def_{hl}_{cap}"]


def walk(G):
    G = G.copy()
    rec = (G["basis"] == "reconstructed").to_numpy()
    order = G["slate_date"].to_numpy()
    tres = G["tres"].to_numpy(float)
    for v in GRID:
        G[f"adj_{name(v)}"] = np.nan
    for d, idx in G.groupby("slate_date").groups.items():
        tr = rec & (order < d)
        if tr.sum() < MIN_TRAIN:
            continue
        y = tres[tr]
        prior = np.flatnonzero(order < d)              # every priced game before the slate
        roll = float(tres[prior[-ROLL:]].mean())
        for v in GRID:
            X = G.loc[tr, cols(v)].to_numpy(float)
            Xc = G.loc[idx, cols(v)].to_numpy(float)
            if v[0] == "fit":
                b = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
                adj = b[0] + Xc @ b[1:]
            elif v[0] == "none":
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                adj = Xc @ b
            else:
                b = np.linalg.lstsq(X, y - roll, rcond=None)[0]
                adj = roll + Xc @ b
            G.loc[idx, f"adj_{name(v)}"] = adj
    return G


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def readings(g, col):
    d = (g["tres"] - g[col]) ** 2 - g["tres"] ** 2
    x = g[(g[col].abs() >= EDGE) & (g["tres"] != 0) & g["close_over"].notna()]
    out = dict(mse=d.mean(), var=d.var(ddof=1) / len(d), n=len(x))
    if len(x) > 1:
        over = (x[col] > 0).to_numpy()
        y = (x["tres"] > 0).to_numpy()
        do, du = dec(x["close_over"]), dec(x["close_under"])
        qo = (1 / do) / (1 / do + 1 / du)
        pay, q = np.where(over, do, du), np.where(over, qo, 1 - qo)
        win = np.where(over, y, ~y)
        roi = np.where(win, pay - 1, -1.0)
        out.update(win=win.mean(), roi=roi.mean(), roi_h=1.96 * roi.std(ddof=1) / np.sqrt(len(roi)),
                   null=(q * pay - 1).mean(), overs=over.mean())
    return out


def report(G):
    L = ["Adjusted closing totals: total' = close + a + b x matchup, 24 variants, walk-forward.",
         "Research only (G9 excludes the market residual as a production target). Grid and",
         "rule fixed before the run. Baseline: the closing total. Squared error vs close in pts^2",
         "(negative = better); ROI ± 95% at |adjustment| >= 0.5, closing prices.", ""]
    R = {}
    for (Y, book) in SCORED + INFO:
        g = G[(G["season"] == Y) & (G["close_book"] == book) & G[f"adj_{name(GRID[0])}"].notna()]
        tag = "scored" if (Y, book) in SCORED else "information only"
        L.append(f"== {book} {Y - 1}-{str(Y)[2:]} ({tag})  n {len(g)}  mean(total - close) "
                 f"{g['tres'].mean():+.2f}")
        for v in GRID:
            r = readings(g, f"adj_{name(v)}")
            R[(Y, book, name(v))] = r
            side = (f"n {r['n']:4d}  won {r['win'] * 100:5.1f}%  ROI {r['roi'] * 100:+6.1f}% ± "
                    f"{r['roi_h'] * 100:5.1f}  null {r['null'] * 100:+.1f}%  overs "
                    f"{r['overs'] * 100:3.0f}%") if "roi" in r else f"n {r['n']:4d}"
            L.append(f"  {name(v):24s} vs close {r['mse']:+6.2f} ± {1.96 * np.sqrt(r['var']):5.2f}  "
                     f"|adj|>=0.5: {side}")
        L.append("")
    L.append("== Decision rule (set before the run; z = 3.08 for 24 variants)")
    both = 0
    for v in GRID:
        s = [R[(Y, b, name(v))] for Y, b in SCORED]
        w = [1 / x["var"] for x in s]
        comb = sum(x["mse"] * wi for x, wi in zip(s, w)) / sum(w)
        half = Z_MULTI * np.sqrt(1 / sum(w))
        beat = all(x["mse"] < 0 for x in s)
        both += beat
        r1 = beat and comb + half < 0
        r2 = all("roi" in x and x["roi"] > x["null"] for x in s)
        L.append(f"  {name(v):24s} squared error {s[0]['mse']:+6.2f} / {s[1]['mse']:+6.2f}, combined "
                 f"{comb:+.2f} ± {half:.2f}: {'pass' if r1 else 'fail'}; ROI vs null "
                 f"{'pass' if r2 else 'fail'} -> {'CANDIDATE' if r1 and r2 else 'not a candidate'}")
    L.append(f"  variants beating the close on squared error in both scored seasons: {both} of "
             f"{len(GRID)} (about {len(GRID) // 4} expected by chance)")
    return L


def posthoc_roll(roll=30):
    """Owner's request after the run (2026-11-05): the rolling intercept over
    the last `roll` games instead of 300. Post-hoc; the registered grid and
    verdict are unchanged."""
    global GRID, ROLL
    keep_grid, keep_roll = GRID, ROLL
    GRID = [v for v in keep_grid if v[0] == "roll"]
    ROLL = roll
    try:
        G = walk(features(games()))
        L = report(G)
    finally:
        GRID, ROLL = keep_grid, keep_roll
    head = [f"POST-HOC (requested after the run): rolling intercept over the last {roll} games",
            "instead of 300; the 8 rolling variants only. Descriptive; the registered rule and",
            "verdict are unchanged (the rule text below is reported for comparison).", ""]
    return head + L


def posthoc_ewma(half_lives=(50, 100, 200, 400)):
    """Owner's request after the run (2026-11-05): an exponentially weighted
    intercept, a = sum(w * (total - close)) / sum(w) over every priced game
    before the slate, w = 0.5 ** (games ago / h); alone (b = 0) and with the
    summed matchup (half-life 5, cap 15) fitted on what remains. Post-hoc."""
    G = features(games())
    order = G["slate_date"].to_numpy()
    tres = G["tres"].to_numpy(float)
    rec = (G["basis"] == "reconstructed").to_numpy()
    col = "m_sum_5_15"
    names = []
    for h in half_lives:
        for kind in ("only", "match"):
            names.append(f"ewma{h}-{kind}")
            G[f"adj_ewma{h}-{kind}"] = np.nan
    for d, idx in G.groupby("slate_date").groups.items():
        tr = rec & (order < d)
        if tr.sum() < MIN_TRAIN:
            continue
        prior = np.flatnonzero(order < d)
        X = G.loc[tr, col].to_numpy(float)
        Xc = G.loc[idx, col].to_numpy(float)
        # each training game's own EWMA intercept as of its slate, so b is
        # fitted on what the intercept left, as it would have been live
        for h in half_lives:
            w = 0.5 ** (np.arange(len(prior))[::-1] / h)
            a = float((w * tres[prior]).sum() / w.sum())
            G.loc[idx, f"adj_ewma{h}-only"] = a
            b = float((X @ (tres[tr] - a)) / (X @ X))
            G.loc[idx, f"adj_ewma{h}-match"] = a + b * Xc
    global GRID
    keep = GRID
    try:
        GRID = []
        L = ["POST-HOC (requested after the run): exponentially weighted intercept, half-life h",
             "games, alone and with the summed matchup (hl 5, cap 15). Descriptive; the registered",
             "grid and verdict are unchanged. Baseline: the closing total.", ""]
        for (Y, book) in SCORED + INFO:
            g = G[(G["season"] == Y) & (G["close_book"] == book) & G[f"adj_{names[0]}"].notna()]
            L.append(f"== {book} {Y - 1}-{str(Y)[2:]}  n {len(g)}  mean(total - close) "
                     f"{g['tres'].mean():+.2f}")
            for nm in names:
                r = readings(g, f"adj_{nm}")
                c = g[f"adj_{nm}"]
                side = (f"n {r['n']:4d}  won {r['win'] * 100:5.1f}%  ROI {r['roi'] * 100:+6.1f}% ± "
                        f"{r['roi_h'] * 100:5.1f}  overs {r['overs'] * 100:3.0f}%") if "roi" in r \
                    else f"n {r['n']:4d}"
                L.append(f"  {nm:16s} adj {c.min():+.2f}..{c.max():+.2f}  vs close {r['mse']:+6.2f} ± "
                         f"{1.96 * np.sqrt(r['var']):5.2f}  |adj|>=0.5: {side}")
            L.append("")
    finally:
        GRID = keep
    return L


def main():
    if "--ewma" in sys.argv:
        text = "\n".join(posthoc_ewma()) + "\n"
        open(os.path.join(OUT, "RESULTS_line_adjust_ewma.txt"), "w").write(text)
        print(text)
        return
    if "--roll30" in sys.argv:
        text = "\n".join(posthoc_roll(30)) + "\n"
        open(os.path.join(OUT, "RESULTS_line_adjust_roll30.txt"), "w").write(text)
        print(text)
        return
    G = walk(features(games()))
    os.makedirs(OUT, exist_ok=True)
    text = "\n".join(report(G)) + "\n"
    open(os.path.join(OUT, "RESULTS_line_adjust.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
