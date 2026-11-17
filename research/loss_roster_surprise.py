#!/usr/bin/env python3
"""2026-27 losses: roster surprise after the five-hour lock.

  python research/loss_roster_surprise.py

Owner request (2026-11-17), the follow-up angle to the loss feature audit
(diagnostics/loss_feature_audit): did the model lose where the lineup that
actually played differed from what the NBA injury report told it at the lock?

For every 2026-27 game the report route's availability terms are computed
twice with production's function (pav.team_availability, half-life 25,
v11 player values with rookies by pick, arrivals not counted in either):
  lock       p = the player's play probability from the last report at
             least 300 minutes before tip (v11 Questionable weighting, play
             rates from report seasons 2022-23 .. 2025-26, existing matcher):
             what the model's report route sees at the five-hour lock;
  actual     p = 1 if the player logged minutes in this game, else 0
             (hindsight).
surprise = actual - lock (home - away), for av_min and av_bpm, valued in
points of home margin at the report route's fitted coefficients
(model/logit_avail.json; production refits per slate, so approximate).
Positive = the home side was stronger than the lock expected.

Readings on the graded native v11 locks (analysis.model_predictions, voids
out), games 1+, ± 95%:
  1. surprise toward the pick (signed to the picked side) in wins vs losses;
  2. actual - model margin (sigma x Phi^-1(lock P)) on surprise points: a
     slope near 1 means the surprise explains the residual at the model's
     own weights;
  3. the same against the lock market (actual - market margin) and the
     close: if the market residual follows the surprise too, the news came
     after the lock (late scratches); if only the model's does, the market
     knew at the lock and the model did not;
  4. how many losses had a surprise of 2+ points against the pick.

Descriptive only, chosen after seeing this season's losses: never a rule, a
filter or evidence for a model change. Nothing here touches data/, public/
or model/*.json. Writes diagnostics/loss_roster_surprise/RESULTS.txt.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
import player_availability as pav                 # noqa: E402
import questionable_weight as qw                  # noqa: E402
import report_name_suffix as rns                  # noqa: E402
import v10_candidate as v10c                      # noqa: E402
import loss_feature_audit as A                    # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "loss_roster_surprise")
SEASON, LOCK_LEAD = 2027, 300
KEY = ["slate_date", "home", "away"]


def terms():
    box, _, value = v10c.values(SEASON, qw.CFG["rookies"])
    st, cov, _ = rns.statuses(SEASON, LOCK_LEAD, False)
    lock = pav.present_map(st, "q", rns.rates(SEASON, LOCK_LEAD, False))
    rows = []
    feats = {}
    for _, tg in box.groupby("team"):
        t = tg["team"].iloc[0]
        a = pav.team_availability(tg, half_life=qw.CFG["hl"], value=value)
        b = pav.team_availability(tg, half_life=qw.CFG["hl"], value=value, present=lock)
        for gid in a:
            feats[(gid, t)] = (a[gid], b[gid])
    g = box[box["home"]].drop_duplicates("game_id")[["game_id", "date", "team", "opp"]]
    for r in g.itertuples(index=False):
        h, w = feats.get((r.game_id, r.team)), feats.get((r.game_id, r.opp))
        if h is None or w is None:
            continue
        rows.append(dict(slate_date=pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                         home=r.team, away=r.opp, covered=r.game_id in cov,
                         act_min=h[0][0] - w[0][0], act_bpm=h[0][2] - w[0][2],
                         lock_min=h[1][0] - w[1][0], lock_bpm=h[1][2] - w[1][2]))
    return pd.DataFrame(rows)


def ols1(x, y):
    b, ci = A.ols(np.asarray(x, float)[:, None], y)
    return b[1], ci[1]


def main():
    av = json.load(open(os.path.join(ROOT, "model", "logit_avail.json")))
    cm = av["coef"][av["features"].index("av_min")]
    cb = av["coef"][av["features"].index("av_bpm")]
    sig = av["sigma"]
    T = terms()
    m = A.population().drop(columns=["av_min", "av_bpm"], errors="ignore")
    m = m.merge(T, on=KEY, how="left")
    m = m[m["covered"].fillna(False).astype(bool)].copy()
    m["s_min"] = m["act_min"] - m["lock_min"]
    m["s_bpm"] = m["act_bpm"] - m["lock_bpm"]
    m["surprise"] = cm * m["s_min"] + cb * m["s_bpm"]          # home pts
    pick = np.where(m["lock_p_home"].astype(float) >= .5, 1.0, -1.0)
    m["toward_pick"] = m["surprise"] * pick
    P = m["lock_p_home"].astype(float).clip(1e-4, 1 - 1e-4)
    m["model_m"] = sig * A.norm.ppf(P)
    m["lock_mkt_m"] = sig * A.norm.ppf(m["lock_q_home"].astype(float).clip(1e-4, 1 - 1e-4))
    cq = pd.to_numeric(m["close_q_home"], errors="coerce")
    m["close_m"] = sig * A.norm.ppf(cq.clip(1e-4, 1 - 1e-4))
    won = m["prediction_won"].astype(int)
    y = m["margin"].to_numpy(float)
    L = []
    w = L.append
    w("2026-27 losses: roster surprise after the five-hour lock "
      "(research/loss_roster_surprise.py).")
    w("Descriptive; chosen after seeing this season's losses: not a rule or evidence for a")
    w("change. ± 95%. Surprise = actual minus lock availability, in points of home margin at")
    w(f"the report route's fitted weights (av_min {cm:.3f}, av_bpm {cb:.3f}); per-slate refits")
    w("make these approximate. Positive toward the pick = the picked side was stronger than")
    w("the lock report implied.")
    w("")
    w(f"Graded native locks, games 1+, report-covered: n {len(m)}, picks "
      f"{won.sum()}-{(1 - won).sum()}. Surprise |pts|: median "
      f"{m['surprise'].abs().median():.2f}, 90th pct {m['surprise'].abs().quantile(.9):.2f}, "
      f"max {m['surprise'].abs().max():.2f}; zero (no change) in "
      f"{int((m['surprise'].abs() < 0.05).sum())} games.")
    w("")
    w("== 1. Surprise toward the pick (points), wins vs losses")
    a, b = A.mean_ci(m.loc[won == 1, "toward_pick"]), A.mean_ci(m.loc[won == 0, "toward_pick"])
    w(f"  wins   {a[0]:+.2f} ± {a[1]:.2f} (n {int((won == 1).sum())})")
    w(f"  losses {b[0]:+.2f} ± {b[1]:.2f} (n {int((won == 0).sum())})")
    d = b[0] - a[0]
    w(f"  loss − win {d:+.2f} ± {np.hypot(a[1], b[1]):.2f}")
    for st, sub in m.groupby(m["gp"] >= 10):
        ww = sub["prediction_won"].astype(int)
        a2 = A.mean_ci(sub.loc[ww == 1, "toward_pick"])
        b2 = A.mean_ci(sub.loc[ww == 0, "toward_pick"])
        w(f"  {'games 10+' if st else 'games 1-9'}: wins {a2[0]:+.2f} ± {a2[1]:.2f}, "
          f"losses {b2[0]:+.2f} ± {b2[1]:.2f}")
    w("")
    w("== 2-3. Residual on the surprise (slope 1 = explained at the model's own weights)")
    for lab, base in (("actual − model margin", m["model_m"]),
                      ("actual − lock market margin", m["lock_mkt_m"]),
                      ("actual − close market margin", m["close_m"])):
        ok = np.isfinite(base.to_numpy(float))
        s, c = ols1(m.loc[ok, "surprise"], y[ok] - base.to_numpy(float)[ok])
        r = np.corrcoef(m.loc[ok, "surprise"], y[ok] - base.to_numpy(float)[ok])[0, 1]
        w(f"  {lab:30s} slope {s:+.2f} ± {c:.2f}  corr {r:+.2f}  (n {ok.sum()})")
    ok = np.isfinite(m["close_m"].to_numpy(float))
    s, c = ols1(m.loc[ok, "surprise"], (m["close_m"] - m["lock_mkt_m"]).to_numpy(float)[ok])
    w(f"  {'close − lock market (move)':30s} slope {s:+.2f} ± {c:.2f}  "
      "(the market moving with the surprise after the lock)")
    w("")
    w("== 4. Large surprises (|surprise| ≥ 2 points)")
    big = m["surprise"].abs() >= 2
    agp = big & (m["toward_pick"] < 0)
    w(f"  games with a large surprise: {int(big.sum())} of {len(m)}; against the pick "
      f"{int(agp.sum())}, of which lost {int((agp & (won == 0)).sum())}")
    w(f"  losses with a large surprise against the pick: {int((agp & (won == 0)).sum())} of "
      f"{int((won == 0).sum())}")
    w("  largest surprises against the pick:")
    cols = ["slate_date", "away", "home", "prediction", "prediction_won", "toward_pick",
            "s_min", "s_bpm"]
    top = m.loc[m["toward_pick"].nsmallest(8).index, cols]
    for r in top.itertuples(index=False):
        w(f"    {r.slate_date} {r.away}@{r.home} pick {r.prediction} "
          f"{'W' if r.prediction_won == 1 else 'L'}  {r.toward_pick:+.2f} pts "
          f"(Δav_min {r.s_min:+.2f}, Δav_bpm {r.s_bpm:+.2f}, home − away)")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "RESULTS.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
