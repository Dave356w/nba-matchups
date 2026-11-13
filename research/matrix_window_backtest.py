#!/usr/bin/env python3
"""Matrix window backtest: how many results per cell make the joint
model-WP x market-WP matrix most stable and most informative?

  python research/matrix_window_backtest.py

Owner's question (2026-11-13), after ML stats gained a Latest 50 / Latest 100
/ All games switch. The cards' Realized WP and the matrix both read a cell's
latest N results; this asks which N, walking forward, carries the most
signal for the next game and moves least from slate to slate.

Walk-forward, exactly as production builds a cell: before each slate, every
game is placed in its cell (model's pick: model-WP band x no-vig market-WP
band of that same team; early and 10+ games pooled, openers their own
cells), and the cell holds results from strictly earlier slates of the
previous and current season, ordered by slate date and game ID. For each
window N the cell's latest N results give

  adj_N     = cell win rate - cell mean no-vig market WP      (Realized WP v2)
  est_N     = clip(q_game + adj_N, 0, 1)                      (the card value)
  roi_N     = the cell's mean flat-1u return at its own prices (the matrix)

Windows: 10, 25, 50, 75, 100, 150, 200, 300 and all (every eligible earlier
result). The market alone (adj = 0) is the baseline. Every window is scored
on the identical games: those whose cell has at least one earlier result.

Measures, per cohort and window:
  Brier, log loss  est_N against the game's result, minus the market's
                   (negative = better); log loss clips est to [0.01, 0.99]
                   and the count of 0%/100% estimates is reported
  signal slope     least squares of (won - q_game) on adj_N: 1 = the cell's
                   gap to its prices carries over in full, 0 = noise
  ROI persistence  slope of the next game's 1u return on roi_N
  jitter           mean |change| of a cell's adj_N between its successive
                   games, in points: lower = more stable
  cell size        mean results behind each estimate

Cohorts (bases, books and seasons never pooled in a score):
  primary    2025-26 DraftKings replay at the close, seeded by 2024-25
             ESPN BET replay (no 2024-25 DraftKings season exists; the seed
             is cross-book, stated)
  secondary  2024-25 ESPN BET replay at the close, no seed (cold start)
  info only  2026-27 native five-hour locks at lock prices, seeded by the
             2025-26 DraftKings replay, as production seeds the cards

Reading, fixed before the run. On the primary cohort the "most stable
informative" window is the N with the lowest Brier difference against the
market; a window "beats the market" only if its 95% interval (game-level,
paired) lies below zero. If none does, the reading is that no window adds
predictive value to the price, and the choice among windows is one of
display stability (jitter, cell size), not accuracy. The other cohorts are
reported beside it and decide nothing. Descriptive research: no display,
rule, record or card value changes; any change to the cards' fixed
latest-50 window is the owner's decision through an amendment.

Caveats: replay P comes from the current fit, whose training seasons may
include the replayed season (flatters the model); closing prices are later
than a five-hour lock. Writes diagnostics/matrix_window/RESULTS.txt. +/- is
95%.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import analysis        # noqa: E402
import ledger          # noqa: E402
import market          # noqa: E402
import wp_calibration  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "matrix_window", "RESULTS.txt")
WINDOWS = (10, 25, 50, 75, 100, 150, 200, 300, None)    # None = all eligible
Z95 = 1.96


def wlabel(n):
    return "all" if n is None else str(n)


def picks(rows, source):
    """One row per game for the model's pick: p (model), q (no-vig market),
    won, ml at the source's price, stage key. `source` 'close' (replay) or
    'lock' (native locks)."""
    out = []
    for _, r in rows.iterrows():
        p = pd.to_numeric(r.get("lock_p_home" if source == "lock" else "p_home"), errors="coerce")
        hml = pd.to_numeric(r.get(f"{source}_home_ml"), errors="coerce")
        aml = pd.to_numeric(r.get(f"{source}_away_ml"), errors="coerce")
        q = market.devig(hml, aml)
        y = pd.to_numeric(r.get("home_won"), errors="coerce")
        gp = np.fmin(pd.to_numeric(r.get("gp_home"), errors="coerce"), pd.to_numeric(r.get("gp_away"), errors="coerce"))
        if not (np.isfinite(p) and 0 <= p <= 1 and np.isfinite(q) and 0 < q < 1 and y in (0, 1) and np.isfinite(gp)):
            continue
        home = p >= .5
        pp, qq = (p, q) if home else (1 - p, 1 - q)
        won = int(y == 1) if home else int(y == 0)
        ml = hml if home else aml
        out.append(dict(game_id=str(r["game_id"]), slate_date=str(r["slate_date"]), season=int(r["season"]),
                        p=float(pp), q=float(qq), won=won, units=float(market.unit_profit(ml, won)),
                        key=("opener" if gp == 0 else "pooled",
                             wp_calibration.band(pp, wp_calibration.MODEL_BOUNDS),
                             wp_calibration.band(qq, wp_calibration.MARKET_BOUNDS))))
    return pd.DataFrame(out)


def walk(seed, target, windows=WINDOWS):
    """Walk forward through `seed` then `target` (each from picks()); score
    only `target` games. A game's cell holds results from strictly earlier
    slates of its own and the previous season. Returns one row per scored
    game with adj/est/roi/n per window (NaN where the cell is empty)."""
    stream = pd.concat([seed.assign(_score=False), target.assign(_score=True)], ignore_index=True)
    stream = stream.sort_values(["slate_date", "game_id"], kind="stable")
    cells, rows = {}, []
    for _, slate in stream.groupby("slate_date", sort=True):
        done = []
        for _, g in slate.iterrows():
            prior = [m for m in cells.get(g.key, []) if g.season - 1 <= m["season"] <= g.season]
            if g._score:
                row = dict(game_id=g.game_id, slate_date=g.slate_date, key=g.key, q=g.q, won=g.won, units=g.units)
                for n in windows:
                    cut = prior if n is None else prior[-n:]
                    k = len(cut)
                    if k:
                        adj = np.mean([m["won"] for m in cut]) - np.mean([m["q"] for m in cut])
                        row[f"adj_{wlabel(n)}"] = adj
                        row[f"est_{wlabel(n)}"] = float(np.clip(g.q + adj, 0, 1))
                        row[f"roi_{wlabel(n)}"] = float(np.mean([m["units"] for m in cut]))
                    else:
                        row[f"adj_{wlabel(n)}"] = row[f"est_{wlabel(n)}"] = row[f"roi_{wlabel(n)}"] = np.nan
                    row[f"n_{wlabel(n)}"] = k
                rows.append(row)
            done.append((g.key, dict(season=g.season, won=g.won, q=g.q, units=g.units)))
        # no result of this slate enters any of its own estimates
        for key, m in done:
            cells.setdefault(key, []).append(m)
    return pd.DataFrame(rows)


def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.var(x) == 0:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    cov = resid.var(ddof=2) * np.linalg.inv(X.T @ X)
    return float(b[1]), float(np.sqrt(cov[1, 1]))


def score(scored, windows=WINDOWS):
    """Per window on the identical games (cell non-empty)."""
    g = scored[scored[f"n_{wlabel(None)}"] > 0] if len(scored) else scored
    y, q = g["won"].to_numpy(float), g["q"].to_numpy(float)
    n = len(g)
    ll = lambda p: -(y * np.log(np.clip(p, .01, .99)) + (1 - y) * np.log(1 - np.clip(p, .01, .99)))  # noqa: E731
    br_q, ll_q = (q - y) ** 2, ll(q)
    out = {"n": n, "market_brier": float(br_q.mean()) if n else np.nan, "rows": []}
    for w in windows:
        lab = wlabel(w)
        if not n:
            break
        est, adj = g[f"est_{lab}"].to_numpy(float), g[f"adj_{lab}"].to_numpy(float)
        db, dl = (est - y) ** 2 - br_q, ll(est) - ll_q
        se = lambda d: float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan  # noqa: E731
        slope, slope_se = _slope(adj, y - q)
        rslope, rslope_se = _slope(g[f"roi_{lab}"].to_numpy(float), g["units"].to_numpy(float))
        jit = []
        for _, c in g.sort_values(["slate_date", "game_id"]).groupby(g["key"].astype(str)):
            a = c[f"adj_{lab}"].to_numpy(float)
            if len(a) > 1:
                jit.extend(np.abs(np.diff(a)))
        out["rows"].append(dict(window=lab, d_brier=float(db.mean()), d_brier_se=se(db),
                                d_ll=float(dl.mean()), d_ll_se=se(dl),
                                certain=int(((est <= 0) | (est >= 1)).sum()),
                                slope=slope, slope_se=slope_se, roi_slope=rslope, roi_slope_se=rslope_se,
                                jitter=float(np.mean(jit)) if jit else np.nan,
                                cell=float(g[f"n_{lab}"].mean())))
    return out


def reading(s):
    if not s["rows"]:
        return "no scored games"
    best = min(s["rows"], key=lambda r: r["d_brier"])
    beats = [r["window"] for r in s["rows"] if r["d_brier"] + Z95 * r["d_brier_se"] < 0]
    stable = min(s["rows"], key=lambda r: r["jitter"])
    txt = (f"lowest Brier vs the market: window {best['window']} ({best['d_brier']:+.4f} ± {Z95 * best['d_brier_se']:.4f}); "
           + (f"beats the market at 95%: {', '.join(beats)}" if beats else "no window beats the market at 95%: the cell adds no predictive value to the price")
           + f"; most stable: window {stable['window']} (jitter {100 * stable['jitter']:.1f} pts)")
    return txt


def render(results):
    lines = ["Matrix window backtest (research/matrix_window_backtest.py). Walk-forward, earlier slates only; ± is 95%.",
             "d Brier / d log loss: Realized WP v2 estimate minus the market (negative = better). slope: (won − q) on the",
             "cell's gap to its prices (1 = carries over, 0 = noise). ROI slope: next 1u return on the cell's trailing ROI.",
             "jitter: mean |change| in a cell's adjustment between its successive games (lower = steadier). cell: mean results used.",
             ""]
    for label, s in results:
        lines.append(f"== {label}: {s['n']} games scored (market Brier {s['market_brier']:.4f})")
        lines.append(f"  {'window':>6} {'d Brier':>17} {'d log loss':>17} {'0/100%':>6} {'slope':>13} {'ROI slope':>14} {'jitter':>7} {'cell':>6}")
        for r in s["rows"]:
            lines.append(f"  {r['window']:>6} {r['d_brier']:+.4f} ± {Z95 * r['d_brier_se']:.4f} {r['d_ll']:+.4f} ± {Z95 * r['d_ll_se']:.4f}"
                         f" {r['certain']:>6} {r['slope']:+.2f} ± {Z95 * r['slope_se']:.2f} {r['roi_slope']:+.3f} ± {Z95 * r['roi_slope_se']:.3f}"
                         f" {100 * r['jitter']:6.1f}p {r['cell']:6.1f}")
        lines.append(f"  reading: {reading(s)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def cohorts(recon, native):
    v11 = recon[recon["model_tag"].astype(str).str.endswith(analysis.FAMILY_TAGS)
                & recon["basis"].eq("reconstructed") & ~recon["game_id"].astype(str).isin(analysis.VOID_LOCKS)]
    espn = picks(v11[(v11.season == 2025) & v11.close_book.eq("espnbet")], "close")
    dk = picks(v11[(v11.season == 2026) & v11.close_book.eq("dk")], "close")
    out = [("2025-26 DraftKings replay at the close, seeded by 2024-25 ESPN BET (primary)", walk(espn, dk)),
           ("2024-25 ESPN BET replay at the close, no seed (secondary)", walk(espn.iloc[:0], espn))]
    live = analysis.model_predictions(native)
    if len(live):
        live = live[live["lock_book"].astype(str).eq(analysis.HYPOTHESIS_BOOK)]
        out.append(("2026-27 native five-hour locks, seeded by 2025-26 DraftKings replay (information only)",
                    walk(dk, picks(live, "lock"))))
    return out


def main():
    recon = ledger.load(os.path.join(ROOT, ledger.RECON_PATH))
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    text = render([(label, score(scored)) for label, scored in cohorts(recon, native)])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
