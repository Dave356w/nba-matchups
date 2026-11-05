#!/usr/bin/env python3
"""Rolling margin calibration: a continuously refitted stretch on the base model.

  python research/rolling_calibration.py     # writes diagnostics/early_dispersion/RESULTS_rolling.txt

Owner's request (2026-11-05), following research/early_dispersion.py: the
season-start stretch K failed, and refitting the whole model continuously
(research/dynamic_fit.py) barely moves it before game 10. Would a stretch
refitted before every slate on the most recent games, across seasons,
adapt fast enough to help early in the season?

Input: arm B's walk-forward predictions (diagnostics/early_dispersion/
predictions_*.csv; each season's B comes from a fit on earlier seasons
only). Before each slate date d, every arm fits m' = c + k·m_B by least
squares on games 1+ strictly before d (any season, both stages; game 0 is
never in a window and is scored by B), sigma the window's residual SD, and
scores that slate's games 1+ with P = Phi(m' / sigma). Fixed before the run:

  R300  the last 300 games before d, equal weights (about three weeks)
  R600  the last 600 games before d, equal weights
  RW    every game before d, weight 0.5 ** (games ago / 300)

Windows start with fold 2017's games, so scoring and the rule use folds
2019-2026, as in research/early_dispersion.py; fold 2027 (this season so
far) is information only. Readings and decision rule are that script's:
arm - B on identical games, ± 95%, game-level, paired, by stage and fold;
against the close for each basis x book x season x stage; an arm is a
candidate only if games 1-9 log loss is below B with its 95% interval
excluding 0, Brier agreeing, better in at least 6 of the 7 folds; games 10+
not worse than B (point estimate, Brier agreeing); not worse on reconstructed
DraftKings 2025-26 and ESPN BET 2024-25 games 10+; no fold better by more
than 0.008. The path of k through 2026-27 is reported. Research only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
import early_dispersion as ed                     # noqa: E402

ARMS = {"R300": dict(window=300), "R600": dict(window=600), "RW": dict(half_life=300)}
NAMES = ["B"] + list(ARMS)


def stretch(P):
    """Add m_/p_ for each rolling arm; return the k path per arm and slate."""
    P = P.sort_values(["slate_date", "home"]).reset_index(drop=True)
    for a in ARMS:
        P[f"m_{a}"], P[f"p_{a}"] = P["m_B"], P["p_B"]          # game 0 keeps B
    pool = P[P["stage"] != "game 0"]
    m_all, y_all = pool["m_B"].to_numpy(float), pool["margin"].to_numpy(float)
    d_all = pool["slate_date"].to_numpy()
    path = []
    for d, idx in P[P["stage"] != "game 0"].groupby("slate_date").groups.items():
        n = int(np.searchsorted(d_all, d, side="left"))       # games strictly before d
        if n < 300:
            P.loc[idx, [f"m_{a}" for a in ARMS]] = np.nan
            continue
        row = dict(slate_date=d, season=int(P.loc[idx[0], "season"]))
        for a, cfg in ARMS.items():
            if "window" in cfg:
                lo = max(0, n - cfg["window"])
                m, y, w = m_all[lo:n], y_all[lo:n], np.ones(n - lo)
            else:
                m, y = m_all[:n], y_all[:n]
                w = 0.5 ** (np.arange(n)[::-1] / cfg["half_life"])
            X = np.column_stack([np.ones(len(m)), m])
            sw = np.sqrt(w)
            b, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            r = y - X @ b
            sigma = float(np.sqrt((w * r ** 2).sum() / w.sum()))
            mm = b[0] + b[1] * P.loc[idx, "m_B"].to_numpy(float)
            P.loc[idx, f"m_{a}"], P.loc[idx, f"p_{a}"] = mm, ed._PHI(mm / sigma)
            row[f"c_{a}"], row[f"k_{a}"], row[f"sigma_{a}"] = b[0], b[1], sigma
        path.append(row)
    return P, pd.DataFrame(path)


def report():
    P = ed.load()
    P, path = stretch(P)
    ed.ALL = NAMES                       # reuse the readings with these arms
    L = ["Rolling margin calibration on B's walk-forward predictions (2026-11-05).",
         "Arms and rule fixed before the run (research/rolling_calibration.py). "
         "Fold 2027 is information only.", ""]
    L += ed.outcome_lines(P)
    L += ed.fit_lines(P)
    ml, gate = ed.market_lines(P)
    L += ml
    L += ed.rule_lines(P, gate)
    L += ["", "== 5. Fitted stretch k (and intercept c) by season: mean over slates "
          "[min, max]"]
    for Y, g in path.groupby("season"):
        L.append(f"  {Y}  " + "  ".join(
            f"{a} k {g[f'k_{a}'].mean():.3f} [{g[f'k_{a}'].min():.2f}, "
            f"{g[f'k_{a}'].max():.2f}] c {g[f'c_{a}'].mean():+.2f}" for a in ARMS))
    cur = path[path["season"] == path["season"].max()]
    L += ["", f"  {int(cur['season'].iloc[0])} slate by slate:"]
    for _, r in cur.iterrows():
        L.append(f"    {r['slate_date']}  " + "  ".join(
            f"{a} k {r[f'k_{a}']:.2f} c {r[f'c_{a}']:+.2f}" for a in ARMS))
    text = "\n".join(L) + "\n"
    open(os.path.join(ed.OUT, "RESULTS_rolling.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    report()
