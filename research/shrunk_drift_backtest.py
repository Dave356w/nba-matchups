#!/usr/bin/env python3
"""Shrunk cell + pooled drift backtest: does shrinking the Realized WP cell
adjustment and adding one pooled, exponentially weighted drift term score
better than Realized WP v3 (latest 200, unshrunk)?

  python research/shrunk_drift_backtest.py

Owner's question (2026-11-13): the 200-result cell barely moves with the live
five-hour locks; can the estimate stay on 200 results and still respond to
recent results? Walk-forward on the matrix window backtest's cohorts and cell
rules (research/matrix_window_backtest.py: earlier slates only, previous +
current season, model's pick, openers their own cells).

Estimators, each added to the game's no-vig market WP q and clipped to 0-1:

  v3       adj = cell win rate - cell mean q, latest 200          (production)
  shrunk   adj * n / (n + SHRINK_GAMES)                           (cell only)
  drift    sum(w * (won - q)) / (sum(w) + DRIFT_PSEUDO) over the latest
           DRIFT_GAMES picks of every cell and stage on earlier slates,
           w = 0.5 ** (age / DRIFT_HALF_LIFE), age 0 = most recent (one number
           per slate, the same for every game)
  v4       shrunk + drift                                         (candidate)

The constants (200 pseudo-games on the cell, half-life 100 picks, 200
pseudo-games and at most 600 picks on the drift) are fixed below. They were
picked on 2026-11-13 from a small grid scored on these same replay seasons,
so every figure here is in-sample to that choice; the v4 forward record
(docs/realized_wp_forward_registration_v4.md) is the out-of-sample test.

Measures: Brier and log loss minus the market's on the identical games
(negative = better, game-level paired 95% interval), the paired v4 - v3
difference, and the largest and mean absolute adjustment. Scored games are
those whose cell has at least one earlier result, as production shows no
estimate for an empty cell.

Caveats as the matrix window backtest: replay P comes from the current fit;
closing prices are later than a five-hour lock; the 2025-26 cohort's seed is
cross-book. Descriptive research; changes nothing by itself. Writes
diagnostics/shrunk_drift/RESULTS.txt. +/- is 95%.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import analysis        # noqa: E402
import ledger          # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "matrix_window_backtest", os.path.join(ROOT, "research", "matrix_window_backtest.py"))
mw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mw)

OUT = os.path.join(ROOT, "diagnostics", "shrunk_drift", "RESULTS.txt")
ESTIMATORS = ("v3", "shrunk", "drift", "v4")
WINDOW_GAMES = 200      # cell results, as Realized WP v3
SHRINK_GAMES = 200      # pseudo-games at zero on the cell adjustment
DRIFT_HALF_LIFE = 100   # picks
DRIFT_PSEUDO = 200      # pseudo-games at zero on the drift
DRIFT_GAMES = 600       # latest pooled picks read by the drift
Z95 = 1.96


def drift(members):
    """Pooled drift from (won, q) pairs ordered oldest to newest."""
    if not members:
        return 0.0
    k = len(members)
    weights = [0.5 ** ((k - 1 - i) / DRIFT_HALF_LIFE) for i in range(k)]
    total = sum(w * (won - q) for w, (won, q) in zip(weights, members))
    return float(total / (sum(weights) + DRIFT_PSEUDO))


def walk(seed, target):
    """Walk forward through `seed` then `target` (each from mw.picks()); score
    only `target` games. One row per scored game with each estimator's
    adjustment; no result of a slate enters any estimate of that slate."""
    stream = pd.concat([seed.assign(_score=False), target.assign(_score=True)], ignore_index=True)
    stream = stream.sort_values(["slate_date", "game_id"], kind="stable")
    cells, pool, rows = {}, [], []
    for _, slate in stream.groupby("slate_date", sort=True):
        done = []
        for _, g in slate.iterrows():
            if g._score:
                ok = lambda m: g.season - 1 <= m["season"] <= g.season  # noqa: E731
                cell = [m for m in cells.get(g.key, []) if ok(m)][-WINDOW_GAMES:]
                recent = [m for m in pool if ok(m)][-DRIFT_GAMES:]
                n = len(cell)
                adj = float(np.mean([m["won"] - m["q"] for m in cell])) if n else 0.0
                shrunk = adj * n / (n + SHRINK_GAMES)
                pooled = drift([(m["won"], m["q"]) for m in recent])
                rows.append(dict(game_id=g.game_id, slate_date=g.slate_date, q=g.q, won=g.won, n=n,
                                 adj_v3=adj, adj_shrunk=shrunk, adj_drift=pooled, adj_v4=shrunk + pooled))
            done.append((g.key, dict(season=g.season, won=g.won, q=g.q)))
        for key, m in done:
            cells.setdefault(key, []).append(m)
            pool.append(m)
    return pd.DataFrame(rows)


def score(scored):
    g = scored[scored["n"] > 0] if len(scored) else scored
    y, q, n = g["won"].to_numpy(float), g["q"].to_numpy(float), len(g)
    ll = lambda p: -(y * np.log(np.clip(p, .01, .99)) + (1 - y) * np.log(1 - np.clip(p, .01, .99)))  # noqa: E731
    half = lambda d: float(Z95 * d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan  # noqa: E731
    out = {"n": n, "market_brier": float(((q - y) ** 2).mean()) if n else np.nan, "rows": [], "paired": None}
    brier = {}
    for name in ESTIMATORS:
        if not n:
            break
        adj = g[f"adj_{name}"].to_numpy(float)
        est = np.clip(q + adj, 0, 1)
        brier[name] = (est - y) ** 2
        db, dl = brier[name] - (q - y) ** 2, ll(est) - ll(q)
        out["rows"].append(dict(name=name, d_brier=float(db.mean()), d_brier_95=half(db),
                                d_ll=float(dl.mean()), d_ll_95=half(dl),
                                mean_abs=float(np.abs(adj).mean()), max_abs=float(np.abs(adj).max())))
    if n:
        d = brier["v4"] - brier["v3"]
        out["paired"] = (float(d.mean()), half(d))
    return out


def render(results):
    lines = ["Shrunk cell + drift backtest (research/shrunk_drift_backtest.py). Walk-forward, earlier slates only; ± is 95%.",
             f"v3: cell adjustment, latest {WINDOW_GAMES}. shrunk: v3 × n/(n+{SHRINK_GAMES}). drift: pooled picks, half-life "
             f"{DRIFT_HALF_LIFE}, {DRIFT_PSEUDO} pseudo-games, latest {DRIFT_GAMES}. v4: shrunk + drift.",
             "d Brier / d log loss: estimate minus the market (negative = better). |adj|: mean and largest adjustment, points.",
             ""]
    for label, s in results:
        lines.append(f"== {label}: {s['n']} games scored (market Brier {s['market_brier']:.4f})")
        lines.append(f"  {'estimate':>8} {'d Brier':>17} {'d log loss':>17} {'mean |adj|':>10} {'max |adj|':>9}")
        for r in s["rows"]:
            lines.append(f"  {r['name']:>8} {r['d_brier']:+.4f} ± {r['d_brier_95']:.4f} {r['d_ll']:+.4f} ± {r['d_ll_95']:.4f}"
                         f" {100 * r['mean_abs']:9.1f}p {100 * r['max_abs']:8.1f}p")
        if s["paired"]:
            lines.append(f"  v4 minus v3, paired Brier: {s['paired'][0]:+.4f} ± {s['paired'][1]:.4f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def cohorts(recon, native):
    v11 = recon[recon["model_tag"].astype(str).str.endswith(analysis.FAMILY_TAGS)
                & recon["basis"].eq("reconstructed") & ~recon["game_id"].astype(str).isin(analysis.VOID_LOCKS)]
    espn = mw.picks(v11[(v11.season == 2025) & v11.close_book.eq("espnbet")], "close")
    dk = mw.picks(v11[(v11.season == 2026) & v11.close_book.eq("dk")], "close")
    out = [("2025-26 DraftKings replay at the close, seeded by 2024-25 ESPN BET (primary)", walk(espn, dk)),
           ("2024-25 ESPN BET replay at the close, no seed (secondary)", walk(espn.iloc[:0], espn))]
    live = analysis.model_predictions(native)
    if len(live):
        live = live[live["lock_book"].astype(str).eq(analysis.HYPOTHESIS_BOOK)]
        out.append(("2026-27 native five-hour locks, seeded by 2025-26 DraftKings replay (information only)",
                    walk(dk, mw.picks(live, "lock"))))
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
