#!/usr/bin/env python3
"""Walk-forward: do cells whose ΔROI is below -4 points both this season and
pooled keep losing, and does betting the other side pay?

  python research/cell_both_negative.py

Owner's question (2026-11-20), the mirror of research/cell_both_positive.py
(same cohorts, cells, ordering, games 1+, earlier slates only). Rule and
reading fixed here, committed before the run.

  GROUP   games whose cell ΔROI is < -4 pp both this season and pooled
          (latest 200), ΔROI = mean 1u return minus mean no-vig null, each
          result at its own price. No season result in the cell: not in it.
  model   1u on the model pick in those games
  fade    1u on the other side in those games, at its own saved price

Measures (± 95%, game level): bets, W-L, ROI, null, ΔROI for the model pick
in the group, the model pick on all other games, their difference, and the
fade in the group.

Reading, fixed now, primary cohort only (2025-26 DraftKings replay seeded by
2024-25 ESPN BET): cold cells "stay cold" only if the group's model-pick ΔROI
is below the other games' with the difference's 95% interval below zero; the
fade "pays" only if its ΔROI 95% interval lies above zero. 2026-27 live locks
are information only. Replay P comes from the current fit (in-sample for
2025-26). Research only: no rule, filter or page change.
Writes diagnostics/cell_both_negative/RESULTS.txt.
"""
from __future__ import annotations

import importlib.util
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

_spec = importlib.util.spec_from_file_location("cell_both_positive", os.path.join(ROOT, "research", "cell_both_positive.py"))
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

OUT = os.path.join(ROOT, "diagnostics", "cell_both_negative", "RESULTS.txt")
CUT = -0.04


def fades(rows, source):
    """game_id -> (1u return, no-vig null) on the side the model did not pick."""
    out = {}
    for _, r in rows.iterrows():
        p = pd.to_numeric(r.get("lock_p_home" if source == "lock" else "p_home"), errors="coerce")
        hml = pd.to_numeric(r.get(f"{source}_home_ml"), errors="coerce")
        aml = pd.to_numeric(r.get(f"{source}_away_ml"), errors="coerce")
        q = market.devig(hml, aml)
        y = pd.to_numeric(r.get("home_won"), errors="coerce")
        if not (np.isfinite(p) and np.isfinite(q) and y in (0, 1)):
            continue
        other_home = not (p >= .5)
        ml, qq = (hml, q) if other_home else (aml, 1 - q)
        won = (y == 1) == other_home
        u = market.unit_profit(ml, won)
        if np.isfinite(u):
            out[str(r["game_id"])] = (float(u), float(qq * market.decimal_payout(ml) - 1), int(won))
    return out


def stats(u, z, w):
    u, z = np.asarray(u, float), np.asarray(z, float)
    n = len(u)
    h = lambda x: cb.Z95 * x.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")  # noqa: E731
    return dict(n=n, w=int(np.sum(w)), roi=u.mean() if n else np.nan, roi95=h(u),
                null=z.mean() if n else np.nan, delta=(u - z).mean() if n else np.nan, delta95=h(u - z))


def line(label, s):
    if not s["n"]:
        return f"  {label:30} no bets"
    return (f"  {label:30} {s['n']:4} bets {s['w']:4}-{s['n'] - s['w']:<4} ROI {100 * s['roi']:+6.1f} ± {100 * s['roi95']:4.1f}%"
            f"  null {100 * s['null']:+5.1f}%  ΔROI {100 * s['delta']:+6.1f} ± {100 * s['delta95']:4.1f} pp")


def main():
    recon = ledger.load(os.path.join(ROOT, ledger.RECON_PATH))
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    native = native[native.basis.eq("native")]
    v11 = recon[recon["model_tag"].astype(str).str.endswith(analysis.FAMILY_TAGS) & recon["basis"].eq("reconstructed")]
    live = analysis.model_predictions(native)
    live = live[live["lock_book"].astype(str).eq(analysis.HYPOTHESIS_BOOK)]
    fade_maps = [fades(v11[(v11.season == 2026) & v11.close_book.eq("dk")], "close"), fades(live, "lock")]
    lines = ["Cell ΔROI both below -4 pp (research/cell_both_negative.py). Walk-forward, earlier slates only; ± is 95%.",
             f"Group: the cell's ΔROI < {100 * CUT:+.0f} pp this season AND pooled (latest {cb.WINDOW}). Games 1+.", ""]
    for (label, d), F in zip(cb.cohorts(recon, native), fade_maps):
        cold = d.pooled.lt(CUT) & d.season.lt(CUT)
        g, rest = d[cold], d[~cold]
        lines.append(f"== {label}: {len(d)} games 1+ picks")
        lines.append(line("model pick, cold cells", stats(g.units, g.null, g.won)))
        lines.append(line("model pick, all other games", stats(rest.units, rest.null, rest.won)))
        x, y = (g.units - g.null).to_numpy(float), (rest.units - rest.null).to_numpy(float)
        if len(x) > 1 and len(y) > 1:
            est = x.mean() - y.mean()
            h = cb.Z95 * math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            lines.append(f"  cold minus other, model ΔROI: {100 * est:+.1f} ± {100 * h:.1f} pp")
        else:
            est = h = float("nan")
        f = [F[i] for i in g.game_id if i in F]
        fs = stats([a for a, _, _ in f], [b for _, b, _ in f], [c for _, _, c in f])
        lines.append(line("fade (other side), cold cells", fs))
        if label.endswith("(PRIMARY)"):
            stay = np.isfinite(est) and est + h < 0
            pays = fs["n"] > 1 and fs["delta"] - fs["delta95"] > 0
            lines.append(f"  READING: cold cells {'stay cold' if stay else 'do not stay cold'}; "
                         f"the fade {'pays' if pays else 'does not pay'}")
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
