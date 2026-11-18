#!/usr/bin/env python3
"""Walk-forward: does betting only when a cell clears break-even pay?

  python research/cell_breakeven_rule.py

Owner (2026-11-18): the matrix and cards are used to judge a cell's ROI
against the price's break-even. This replays that judgement as a bettor
could have made it, before each game, from strictly earlier slates
(research/matrix_window_backtest.walk: model's pick, model-WP x market-WP
cell, games 1+ pooled, previous + current season, slate/game order):

  ROI>0 (N)   bet when the cell's trailing 1u ROI over its latest N > 0
  RWP>BE (N)  bet when Realized WP = q + (cell win rate - cell mean q)
              over the latest N clears this game's break-even 1/decimal
  RWP4>BE     the same with v4's shrinkage n/(n+200) on the cell gap
              (no drift)
For each: ROI of bets taken vs passed, difference ± 95%, and the taken
bets' excess over the no-vig null. Cohorts as in matrix_window_backtest,
never pooled. Chosen after the matrix was read: descriptive only, no rule.
Writes diagnostics/cell_breakeven_rule/RESULTS.txt.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import analysis  # noqa: E402
import ledger    # noqa: E402
from matrix_window_backtest import picks, walk  # noqa: E402
from matrix_interactions import with_dec  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "cell_breakeven_rule", "RESULTS.txt")
Z = 1.96


def mci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan
    return x.mean(), Z * x.std(ddof=1) / np.sqrt(len(x))


def p(x, ci):
    return f"{100*x:+6.1f} ± {100*ci:4.1f}" if np.isfinite(ci) else f"{100*x:+6.1f}" if np.isfinite(x) else "—"


def section(label, seed, target):
    s = walk(seed, target, windows=(50, 100, 200, None))
    s = s.merge(target[["game_id", "_dec"]], on="game_id")
    s = s[s["n_all"] > 0].copy()
    s["be"] = 1 / s._dec
    s["null"] = s.q * s._dec - 1
    rules = {}
    for n in ("50", "200", "all"):
        rules[f"ROI>0 ({n})"] = s[f"roi_{n}"] > 0
        rules[f"RWP>BE ({n})"] = s[f"est_{n}"] > s.be
    adj200 = s["adj_200"] * s["n_200"] / (s["n_200"] + 200)
    rules["RWP4>BE (200, shrunk)"] = (s.q + adj200).clip(0, 1) > s.be
    L = [f"== {label} ==", f"games {len(s)} · all model picks ROI {p(*mci(s.units))}% · null {100*s.null.mean():+.1f}%",
         f"   {'rule':22s} {'bet n':>6s} {'bet ROI %':>15s} {'pass n':>7s} {'pass ROI %':>15s} {'bet - pass pp':>15s} {'bet excess pp':>15s}"]
    for name, m in rules.items():
        a, b = s[m], s[~m]
        ma, ca = mci(a.units)
        mb, cb = mci(b.units)
        se = np.sqrt((ca / Z) ** 2 + (cb / Z) ** 2) if np.isfinite(ca) and np.isfinite(cb) else np.nan
        L.append(f"   {name:22s} {len(a):6d} {p(ma, ca):>15s} {len(b):7d} {p(mb, cb):>15s} "
                 f"{p(ma - mb, Z*se):>15s} {p(*mci(a.units - a.null)):>15s}")
    return L + [""]


def main():
    recon = ledger.load(os.path.join(ROOT, ledger.RECON_PATH))
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    v11 = recon[recon["model_tag"].astype(str).str.endswith(analysis.FAMILY_TAGS)
                & recon["basis"].eq("reconstructed") & ~recon["game_id"].astype(str).isin(analysis.VOID_LOCKS)]
    er = v11[(v11.season == 2025) & v11.close_book.eq("espnbet")]
    dr = v11[(v11.season == 2026) & v11.close_book.eq("dk")]
    espn, dk = with_dec(er, "close", picks(er, "close")), with_dec(dr, "close", picks(dr, "close"))
    live = analysis.model_predictions(native)
    live = live[live["lock_book"].astype(str).eq(analysis.HYPOTHESIS_BOOK)]
    lv = with_dec(live, "lock", picks(live, "lock"))
    L = ["Cell vs break-even, walk-forward (research/cell_breakeven_rule.py). ± is 95%.",
         "Model's pick, flat 1u at the source price; cell from strictly earlier slates.", ""]
    L += section("2025-26 DraftKings replay at the close, seeded by 2024-25 ESPN BET (primary)", espn, dk)
    L += section("2024-25 ESPN BET replay at the close, no seed", espn.iloc[:0], espn)
    L += section("2026-27 live five-hour locks, seeded by 2025-26 DraftKings replay (information only)", dk, lv)
    text = "\n".join(L).rstrip() + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
