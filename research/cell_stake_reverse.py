#!/usr/bin/env python3
"""Walk-forward: size each model-pick bet AGAINST its cell's pooled ΔROI?

  python research/cell_stake_reverse.py

Owner's question (2026-11-20), after research/cell_stake_sizing.py lost to
flat on 2025-26 and that season's matrix window ROI slope was negative.
Rule and reading fixed here, committed before the run. Same cohorts, cells and ordering as research/cell_both_positive.py
(games 1+, earlier slates only, latest-200 pooled cell).

  SIZED   stake = clip(1 - pooled ΔROI, 0.5, 1.5) units on the model pick,
          pooled ΔROI as a fraction (+0.08 -> 0.92u); no pooled result -> 1u
  FLAT    1 unit on the same pick

Measures (± 95%, game level): total staked, net units and return per unit
staked for both; the paired per-game difference sized minus flat profit,
(stake - 1) x 1u return, its mean and total.

The rule was chosen after seeing 2025-26 (it is the mirror of a rule that
lost there), so 2025-26 cannot test it: within the caps it gains there what
the forward rule lost. Reading, fixed now: the PRIMARY cohort is 2024-25
ESPN BET replay at the close, no seed (cells from that season only; its
matrix ROI slope was +0.06 ± 0.21). Reverse sizing beats flat only if the
paired mean difference's 95% interval lies above zero there. 2025-26 is
shown, labelled in-sample to the choice; 2026-27 live locks are information
only. Replay P comes from the current fit. Research only.
Writes diagnostics/cell_stake_reverse/RESULTS.txt.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ledger          # noqa: E402

_spec = importlib.util.spec_from_file_location("cell_both_positive", os.path.join(ROOT, "research", "cell_both_positive.py"))
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

OUT = os.path.join(ROOT, "diagnostics", "cell_stake_reverse", "RESULTS.txt")
LO, HI = 0.5, 1.5


def stakes(pooled):
    s = 1 - np.nan_to_num(np.asarray(pooled, float), nan=0.0)
    return np.clip(s, LO, HI)


def main():
    recon = ledger.load(os.path.join(ROOT, ledger.RECON_PATH))
    native = ledger.load(os.path.join(ROOT, ledger.NATIVE_PATH))
    native = native[native.basis.eq("native")]
    lines = ["Reverse cell stake sizing (research/cell_stake_reverse.py). Walk-forward, earlier slates only; ± is 95%.",
             f"Sized: stake = clip(1 - pooled ΔROI, {LO}, {HI}) units (latest {cb.WINDOW}); flat: 1 unit. Games 1+.", ""]
    import analysis
    v11 = recon[recon["model_tag"].astype(str).str.endswith(analysis.FAMILY_TAGS)
                & recon["basis"].eq("reconstructed") & ~recon["game_id"].astype(str).isin(analysis.VOID_LOCKS)]
    espn = cb.picks(v11[(v11.season == 2025) & v11.close_book.eq("espnbet")], "close")
    runs = [("2024-25 ESPN BET replay at the close, no seed (PRIMARY)", cb.walk(espn.iloc[:0], espn))]
    for label, d in cb.cohorts(recon, native):
        runs.append((label.replace(" (PRIMARY)", " (in-sample to the rule's choice)"), d))
    for label, d in runs:
        u = d.units.to_numpy(float)
        s = stakes(d.pooled)
        n = len(u)
        diff = (s - 1) * u
        half = cb.Z95 * diff.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        lines.append(f"== {label}: {n} games 1+ picks")
        lines.append(f"  flat    staked {n:8.1f}u  net {u.sum():+7.2f}u  return per unit {100 * u.mean():+5.1f}%")
        lines.append(f"  sized   staked {s.sum():8.1f}u  net {(s * u).sum():+7.2f}u  return per unit "
                     f"{100 * (s * u).sum() / s.sum():+5.1f}%  (stakes {s.min():.2f}–{s.max():.2f}, "
                     f"{int((s == HI).sum())} at {HI}, {int((s == LO).sum())} at {LO})")
        lines.append(f"  sized minus flat, per game: {diff.mean():+.4f} ± {half:.4f}u; total {diff.sum():+.2f}u")
        if label.endswith("(PRIMARY)"):
            lines.append(f"  READING: {'sizing beats flat' if diff.mean() - half > 0 else 'sizing does not beat flat'}")
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
