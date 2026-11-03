#!/usr/bin/env python3
"""Compare a stale_log_replay.py run with the frozen locks (HINDSIGHT).

  python research/stale_log_replay_eval.py <repo> <stale.csv> <complete.csv|-> <out_dir> [slate]

Checks the stale replay against the frozen lock P, total and P(over), then
applies the public tests' registered rules (analysis.hypothesis_picks /
ou_picks at "lock", read unchanged) to the graded rows three ways: as frozen,
with the stale replay and with the complete one. <repo> supplies the graded
ledger (read only). Information only: no record reads these picks.
"""
import os, sys
repo, stale_p, comp_p, out_dir = sys.argv[1:5]
SLATE = sys.argv[5] if len(sys.argv) > 5 else "2026-11-02"
os.chdir(repo); sys.path.insert(0, repo)
import numpy as np, pandas as pd  # noqa: E402
from statistics import NormalDist  # noqa: E402
import analysis, ledger  # noqa: E402

L = ledger.load(ledger.NATIVE_PATH)                       # read only
g0 = L[L["slate_date"].astype(str) == SLATE].copy()
g0["game_id"] = g0["game_id"].astype(str)
runs = {"stale": pd.read_csv(stale_p, dtype={"game_id": str})}
if comp_p != "-":
    runs["complete"] = pd.read_csv(comp_p, dtype={"game_id": str})
z = lambda p: 13.5 * NormalDist().inv_cdf(min(max(p, 1e-9), 1 - 1e-9))  # H5's frozen conversion

def swapped(r):
    g = g0.copy().set_index("game_id")
    r = r.set_index("game_id")
    g.loc[r.index, "lock_p_home"] = r["p_home"]
    g.loc[r.index, "lock_model_tag"] = r["model_tag"]
    for a, b in (("lock_ou_p_over", "ou_p_over"), ("lock_ou_total", "ou_total"), ("lock_ou_tag", "ou_tag")):
        g.loc[r.index, a] = r[b]
    return g.reset_index()

frames = {"frozen": g0, **{k: swapped(v) for k, v in runs.items()}}
runs_ids = set(runs["stale"]["game_id"])
g0 = g0[g0["game_id"].isin(runs_ids)]
cmp = g0.set_index("game_id")[["away", "home", "lock_p_home", "lock_ou_total", "lock_ou_p_over",
                               "lock_report_utc", "lock_model_tag"]]
for k, r in runs.items():
    r = r.set_index("game_id")
    for c in ("p_home", "ou_total", "ou_p_over", "gp_home", "gp_away", "home_b2b", "away_b2b"):
        cmp[f"{k}_{c}"] = r[c]
    cmp[f"{k}_report"] = r["report_utc"]
    cmp[f"{k}_tag_same"] = r["model_tag"].eq(cmp["lock_model_tag"])
if "complete" in runs:   # a side whose games played the complete logs changed
    cmp["b2b_game"] = (cmp.complete_gp_home.ne(cmp.stale_gp_home) | cmp.complete_gp_away.ne(cmp.stale_gp_away))
cmp["stale_dp"] = cmp.stale_p_home - cmp.lock_p_home
if "complete" in runs:
    cmp["shift_p"] = cmp.complete_p_home - cmp.stale_p_home
    cmp["shift_margin"] = [z(c) - z(s) for c, s in zip(cmp.complete_p_home, cmp.stale_p_home)]
    cmp["shift_total"] = cmp.complete_ou_total - cmp.stale_ou_total
cmp.to_csv(os.path.join(out_dir, f"replay_compare_{SLATE}.csv"))
pd.set_option("display.width", 250)
print(f"== reproduction: stale replay vs frozen lock ({SLATE}, {len(cmp)} games)")
print(cmp[["away", "home", "lock_p_home", "stale_p_home", "stale_dp", "lock_ou_total", "stale_ou_total",
           "lock_ou_p_over", "stale_ou_p_over", "stale_tag_same"]].round(5).to_string())
print("max |dP|", float(cmp.stale_dp.abs().max()), " max |d total|", float((cmp.stale_ou_total - cmp.lock_ou_total).abs().max()),
      " max |d p_over|", float((cmp.stale_ou_p_over - cmp.lock_ou_p_over).abs().max()),
      " report same:", bool((pd.to_datetime(cmp.stale_report, utc=True) == pd.to_datetime(cmp.lock_report_utc, utc=True)).all()))
if "complete" in runs:
    print("\n== complete logs vs stale (lock P is home win P; margin via H5's 13.5 x inv-Phi)")
    print(cmp[["away", "home", "b2b_game", "stale_gp_away", "complete_gp_away", "stale_gp_home", "complete_gp_home",
               "complete_away_b2b", "complete_home_b2b", "stale_p_home", "complete_p_home", "shift_p", "shift_margin",
               "stale_ou_total", "complete_ou_total", "shift_total"]].round(4).to_string())

tests = [(k, next(h for h in analysis.active_hypotheses() if h["key"] == k), "ml")
         for k in ("H5·3·E·L·v11", "H2·F·5·E·L·v11", "H4·E·L·v11")]
tests.append(("OU·E·L", next(t for t in analysis.OU_TESTS if t["key"] == "OU·E·L"), "ou"))
picks = []
for v, g in frames.items():
    for key, h, kind in tests:
        d = analysis.ou_picks(g, h, "lock") if kind == "ou" else analysis.hypothesis_picks(g, h, "lock")
        if not len(d):
            continue
        d = d.assign(game_id=d["game_id"].astype(str)).merge(g0[["game_id", "away", "home"]], on="game_id")
        res = d["result"] if "result" in d else d["won"]
        picks += [dict(variant=v, test=key, game=f"{a}@{hm}", side=s, line=d.get("line", pd.Series([np.nan]*len(d))).iloc[i],
                       ml=m, edge=round(float(e), 4), result=float(rs), units=round(float(u), 3))
                  for i, (a, hm, s, m, e, rs, u) in enumerate(zip(d.away, d.home, d.side, d.ml, d.edge, res, d.units))]
P = pd.DataFrame(picks)
P.to_csv(os.path.join(out_dir, f"replay_picks_{SLATE}.csv"), index=False)
print(f"\n== tracked picks by variant ({SLATE}, 5 h lock, frozen lock prices)")
summ = P.groupby(["test", "variant"]).agg(n=("units", "size"), w=("result", lambda r: int((r == 1).sum())),
                                          l=("result", lambda r: int((r == 0).sum())), units=("units", "sum")).round(3)
print(summ.to_string())
key = ["test", "game", "side"]
for a, b in (("frozen", "stale"), ("stale", "complete")):
    if b not in P.variant.unique():
        continue
    x = P[P.variant == a].set_index(key); y = P[P.variant == b].set_index(key)
    print(f"\n-- in {a} not {b}:"); print(x.loc[x.index.difference(y.index)][["edge", "ml", "result", "units"]].to_string() if len(x.index.difference(y.index)) else "   none")
    print(f"-- in {b} not {a}:"); print(y.loc[y.index.difference(x.index)][["edge", "ml", "result", "units"]].to_string() if len(y.index.difference(x.index)) else "   none")
