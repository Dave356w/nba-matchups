#!/usr/bin/env python3
"""Walk-forward: bet the model pick only when its cell's ΔROI is positive
both this season and pooled?

  python research/cell_both_positive.py

Owner's question (2026-11-20), after the cards gained "This season" and
"Pooled" cell lines. Rule and reading fixed here, committed before the run.

Walk-forward exactly as the card reads a cell (wp_calibration /
research/matrix_window_backtest.py): before each slate every game sits in
its cell (model pick: model-WP band x no-vig market-WP band of that team,
early and 10+ pooled; openers are not bet, as the public record is games
1+). The cell holds results from strictly earlier slates of the previous and
current season, ordered by slate date and game ID.

  pooled ΔROI   the cell's latest 200 results: mean 1u return minus mean
                no-vig null (q x decimal payout - 1), each at its own price
  season ΔROI   the same over the cell's current-season results only
  RULE          bet 1u on the model pick at the game's own price when both
                are > 0. No season result in the cell yet: no bet.

Measures, per cohort (± 95%, game level): bets, W-L, ROI, null, ΔROI of the
rule's bets; the same for every games-1+ pick (the public record) and for
the picks the rule skips; and the paired question that matters: does the
rule's ΔROI exceed the skipped picks' (difference ± 95%). For context only,
not a search: pooled-only > 0 and season-only > 0.

Cohorts (book x season never pooled):
  PRIMARY  2025-26 DraftKings replay at the close (v11 replay P), seeded by
           2024-25 ESPN BET replay (no 2024-25 DraftKings season exists:
           the seed is cross-book).
  info     2026-27 native five-hour locks (lock P and prices, voids out),
           seeded by 2025-26 DraftKings replay. ~150 games: information only.

Reading, fixed now: the rule shows an edge only if, on the primary cohort,
its ΔROI 95% interval lies above zero AND it beats the skipped picks with
the difference's interval above zero. Otherwise it does not separate from
betting every pick. Replay P comes from the current fit (hindsight). Research
only: never a rule, filter or card change without the owner's amendment.
Writes diagnostics/cell_both_positive/RESULTS.txt.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import analysis        # noqa: E402
import ledger          # noqa: E402
import market          # noqa: E402
import wp_calibration as wc  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "cell_both_positive", "RESULTS.txt")
WINDOW = wc.WINDOW_GAMES
Z95 = 1.96


def picks(rows, source):
    """One row per game: model pick's q, won, 1u return, null, stage, cell."""
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
        pp, qq, ml = (p, q, hml) if home else (1 - p, 1 - q, aml)
        won = int(y == 1) if home else int(y == 0)
        units = market.unit_profit(ml, won)
        if not np.isfinite(units):
            continue
        out.append(dict(game_id=str(r["game_id"]), slate_date=str(r["slate_date"]), season=int(r["season"]),
                        opener=bool(gp == 0), won=won, units=float(units),
                        null=float(qq * market.decimal_payout(ml) - 1),
                        key=("opener" if gp == 0 else "pooled",
                             wc.band(pp, wc.MODEL_BOUNDS), wc.band(qq, wc.MARKET_BOUNDS))))
    return pd.DataFrame(out)


def delta(results):
    return float(np.mean([m["units"] - m["null"] for m in results])) if results else float("nan")


def walk(seed, target):
    """Each target games-1+ pick with its cell's pooled and season ΔROI from
    strictly earlier slates; no result of a slate enters that slate."""
    stream = pd.concat([seed.assign(_score=False), target.assign(_score=True)], ignore_index=True)
    stream = stream.sort_values(["slate_date", "game_id"], kind="stable")
    cells, rows = {}, []
    for _, slate in stream.groupby("slate_date", sort=True):
        done = []
        for _, g in slate.iterrows():
            if g._score and not g.opener:
                held = [m for m in cells.get(g.key, []) if g.season - 1 <= m["season"] <= g.season]
                pooled = held[-WINDOW:]
                season = [m for m in held if m["season"] == g.season]
                rows.append(dict(game_id=g.game_id, units=g.units, null=g.null, won=g.won,
                                 pooled=delta(pooled), season=delta(season), n_pooled=len(pooled), n_season=len(season)))
            done.append((g.key, dict(season=g.season, units=g.units, null=g.null)))
        for key, m in done:
            cells.setdefault(key, []).append(m)
    return pd.DataFrame(rows)


def summary(d):
    if not len(d):
        return dict(n=0)
    u, z = d.units.to_numpy(float), d.null.to_numpy(float)
    dd = u - z
    half = lambda x: Z95 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")  # noqa: E731
    return dict(n=len(d), w=int(d.won.sum()), roi=u.mean(), roi95=half(u), null=z.mean(), delta=dd.mean(), delta95=half(dd))


def fmt(label, s):
    if not s["n"]:
        return f"  {label:34} no bets"
    return (f"  {label:34} {s['n']:4} bets {s['w']:4}-{s['n'] - s['w']:<4} ROI {100 * s['roi']:+6.1f} ± {100 * s['roi95']:4.1f}%"
            f"  null {100 * s['null']:+5.1f}%  ΔROI {100 * s['delta']:+6.1f} ± {100 * s['delta95']:4.1f} pp")


def diff(a, b):
    """Rule bets' ΔROI minus skipped picks' ΔROI, ± 95% (independent groups)."""
    x, y = (a.units - a.null).to_numpy(float), (b.units - b.null).to_numpy(float)
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    return x.mean() - y.mean(), Z95 * math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))


def cohorts(recon, native):
    v11 = recon[recon["model_tag"].astype(str).str.endswith(analysis.FAMILY_TAGS)
                & recon["basis"].eq("reconstructed") & ~recon["game_id"].astype(str).isin(analysis.VOID_LOCKS)]
    espn = picks(v11[(v11.season == 2025) & v11.close_book.eq("espnbet")], "close")
    dk = picks(v11[(v11.season == 2026) & v11.close_book.eq("dk")], "close")
    out = [("2025-26 DraftKings replay at the close, seeded by 2024-25 ESPN BET (PRIMARY)", walk(espn, dk))]
    live = analysis.model_predictions(native)
    if len(live):
        live = live[live["lock_book"].astype(str).eq(analysis.HYPOTHESIS_BOOK)]
        out.append(("2026-27 native five-hour locks, seeded by 2025-26 DraftKings replay (information only)",
                    walk(dk, picks(live, "lock"))))
    return out


def main():
    recon = ledger.load(os.path.join(ROOT, ledger.RECON_PATH))
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    native = native[native.basis.eq("native")]
    lines = ["Cell ΔROI both positive (research/cell_both_positive.py). Walk-forward, earlier slates only; ± is 95%.",
             f"Rule: bet the model pick when its cell's ΔROI > 0 this season AND pooled (latest {WINDOW}). Games 1+.", ""]
    for label, d in cohorts(recon, native):
        both = d.pooled.gt(0) & d.season.gt(0)
        lines.append(f"== {label}: {len(d)} games 1+ picks")
        lines.append(fmt("every pick (public record)", summary(d)))
        lines.append(fmt("RULE: season > 0 and pooled > 0", summary(d[both])))
        lines.append(fmt("skipped by the rule", summary(d[~both])))
        est, h = diff(d[both], d[~both])
        lines.append(f"  rule minus skipped, ΔROI: {100 * est:+.1f} ± {100 * h:.1f} pp")
        s = summary(d[both])
        if label.endswith("(PRIMARY)"):
            edge = s["n"] and s["delta"] - s["delta95"] > 0 and est - h > 0
            lines.append(f"  READING: {'the rule shows an edge' if edge else 'the rule does not separate from betting every pick'}")
        lines.append("  context only (not a search):")
        lines.append(fmt("  pooled > 0 only", summary(d[d.pooled.gt(0)])))
        lines.append(fmt("  season > 0 only", summary(d[d.season.gt(0)])))
        lines.append(f"  no season result in the cell yet (never bet): {int(d.n_season.eq(0).sum())} picks")
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
