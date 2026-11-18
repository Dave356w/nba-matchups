#!/usr/bin/env python3
"""Compare research/suffix_lock_replay.py runs with the frozen 2026-27 locks.

  python research/suffix_lock_compare.py <deployed.csv> <fixed.csv> [<repro.csv>]

Report-name suffix fix gate, "the 2026-27 lock reconstruction" (CLAUDE.md,
registered 2026-11-13). Three replays of every lock, all at the lock time
with the lock's prices and game logs before the slate:
  deployed  the committed model files (old matcher): must reproduce lock P
  repro     the unchanged code refitted today (old matcher; today's BBR
            BPM/draft pages, which differ slightly from the committed fit's)
  fixed     the fix refitted today (split matcher, q_rates_sfx table)
fixed - repro isolates the matcher; fixed - deployed is what activation
through Fit model would roughly do. Voided stale-log locks are listed apart
(they never reproduce: their logs were behind). HINDSIGHT, information only:
locks are never rescored and no value here enters a record.
Writes diagnostics/report_name_suffix/LOCKS.txt.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import analysis  # noqa: E402
import ledger    # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "report_name_suffix", "LOCKS.txt")


def load(path, arm):
    d = pd.read_csv(path, dtype={"game_id": str})
    keep = ["game_id", "p_home", "model_tag", "route", "ou_total", "ou_tag"]
    d = d[[c for c in keep if c in d.columns]].drop_duplicates("game_id", keep="last")
    return d.rename(columns={c: f"{c}_{arm}" for c in d.columns if c != "game_id"})


def stats(x):
    x = np.abs(np.asarray(x, float))
    x = x[np.isfinite(x)]
    if not len(x):
        return "n 0"
    return (f"n {len(x)}  mean {100*x.mean():.2f}  p90 {100*np.quantile(x, .9):.2f}  "
            f"max {100*x.max():.2f} pts;  > 0.5 pt {int((x > .005).sum())}  > 2 pts {int((x > .02).sum())}")


def main():
    paths = sys.argv[1:]
    arms = ["deployed", "fixed", "repro"][:len(paths)]
    L = ledger.load(ledger.NATIVE_PATH)
    L = L[L["season"].astype(str).eq("2027") & L["lock_snapshot_utc"].notna()].copy()
    L["game_id"] = L["game_id"].astype(str)
    m = L[["game_id", "slate_date", "away", "home", "lock_p_home", "lock_model_tag", "home_won",
           "lock_snapshot_utc", "lock_ou_snapshot_utc", "lock_ou_total"]].copy()
    for p, a in zip(paths, arms):
        m = m.merge(load(p, a), on="game_id", how="left")
    void = m["game_id"].isin(analysis.VOID_LOCKS)
    lp = pd.to_numeric(m["lock_p_home"], errors="coerce")
    out = [__doc__.strip().splitlines()[0], "",
           f"2026-27 locks {len(m)} (voided stale-log {int(void.sum())}); replayed: "
           + ", ".join(f"{a} {int(m[f'p_home_{a}'].notna().sum())}" for a in arms), ""]
    v = m[~void]
    d = pd.to_numeric(v["p_home_deployed"], errors="coerce") - pd.to_numeric(v["lock_p_home"], errors="coerce")
    out.append("== 1. Reproduction: deployed replay minus frozen lock P (non-void locks)")
    out.append(f"   |dP|: {stats(d)}")
    out.append(f"   within 0.005: {int((d.abs() <= .005).sum())} of {int(d.notna().sum())}; tag equal: "
               f"{int((v['model_tag_deployed'] == v['lock_model_tag']).sum())}")
    bad = v[d.abs() > .005]
    for r in bad.itertuples():
        out.append(f"     {r.slate_date} {r.away} @ {r.home} ({r.game_id}) lock {float(r.lock_p_home):.4f} "
                   f"replay {float(r.p_home_deployed):.4f}")
    if "ou_total_deployed" in v:
        same = v["lock_ou_snapshot_utc"].astype(str) == v["lock_snapshot_utc"].astype(str)
        dt = (pd.to_numeric(v["ou_total_deployed"], errors="coerce")
              - pd.to_numeric(v["lock_ou_total"], errors="coerce"))[same].abs()
        dt = dt[dt.notna()]
        out.append(f"   O/U total, locks whose O/U lock is the same snapshot: n {len(dt)}, within 0.05 pt "
                   f"{int((dt <= .05).sum())}, max |dT| {dt.max() if len(dt) else float('nan'):.2f}")
    out.append("")
    y = pd.to_numeric(v["home_won"], errors="coerce")
    for a, b, title in [("fixed", "deployed", "fixed minus deployed (what activation would roughly do)"),
                        ("fixed", "repro", "fixed minus repro (the matcher alone, same BBR pages)"),
                        ("repro", "deployed", "repro minus deployed (BBR page drift alone)")]:
        if f"p_home_{a}" not in v or f"p_home_{b}" not in v:
            continue
        pa = pd.to_numeric(v[f"p_home_{a}"], errors="coerce")
        pb = pd.to_numeric(v[f"p_home_{b}"], errors="coerce")
        dd = pa - pb
        flips = int(((pa >= .5) != (pb >= .5))[dd.notna()].sum())
        g = y.isin([0, 1]) & dd.notna()
        br = ((pa - y) ** 2 - (pb - y) ** 2)[g]
        out.append(f"== {title}")
        out.append(f"   |dP|: {stats(dd)};  winner changes {flips}")
        out.append(f"   graded {int(g.sum())}: Brier {br.mean():+.4f} ± {1.96*br.std(ddof=1)/np.sqrt(len(br)):.4f} (95%)")
        big = v[dd.abs() > .02].assign(dp=dd[dd.abs() > .02]).sort_values("dp", key=abs, ascending=False)
        for r in big.itertuples():
            out.append(f"     {r.slate_date} {r.away} @ {r.home}: {b} {float(getattr(r, f'p_home_{b}')):.4f} -> "
                       f"{a} {float(getattr(r, f'p_home_{a}')):.4f} ({100*r.dp:+.2f} pts)")
        if f"ou_total_{a}" in v and f"ou_total_{b}" in v:
            dt = (pd.to_numeric(v[f"ou_total_{a}"], errors="coerce") - pd.to_numeric(v[f"ou_total_{b}"], errors="coerce")).abs()
            dt = dt[dt.notna()]
            if len(dt):
                out.append(f"   O/U total |dT|: n {len(dt)} mean {dt.mean():.2f} max {dt.max():.2f} pts; > 0.5 pt {int((dt > .5).sum())}")
        out.append("")
    if "model_tag_fixed" in v:
        out.append("fixed tags: " + str(v["model_tag_fixed"].value_counts().to_dict()))
    text = "\n".join(out).rstrip() + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
