#!/usr/bin/env python3
"""2026-27 losses by team and by situation.

  python research/loss_context.py

Owner request (2026-11-17), a third angle on this season's losses after the
feature audit (diagnostics/loss_feature_audit) and the roster surprise
(diagnostics/loss_roster_surprise).

Population: the graded native v11 five-hour locks, games 1+, voids out
(analysis.model_predictions, as in loss_feature_audit). Margins in points:
model = sigma x Phi^-1(lock P), market = sigma x Phi^-1(lock no-vig q), at
the report route's sigma (both use the same scale, so the comparison is fair).

1. Teams. Residual r = actual home margin - predicted. Each game gives the
   home team +r and the away team -r; a team's effect is fitted by least
   squares on +1/-1 indicators (sum to zero). Is the spread of team effects
   more than noise? Permutation test: shuffle the residuals across games
   (2,000 draws, seed 0) and compare the variance of the fitted effects. Done
   for the model and the market on the same games; the largest model effects
   are listed beside the market's for those teams.
2. Situations: accuracy and mean residual toward the pick, model beside the
   lock-market favourite on the same games, by whether they agreed, home or
   away pick, pick on a back-to-back, opponent on a back-to-back, model
   probability band, stage.

Descriptive, chosen after seeing this season's losses; ~9 games a team, so a
listed team is a lead to check, never a finding. Never a rule or a filter.
Nothing here touches data/, public/ or model/*.json.
Writes diagnostics/loss_context/RESULTS.txt.
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
import loss_feature_audit as A                    # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "loss_context")
DRAWS, SEED = 2000, 0


def team_effects(m, r):
    teams = sorted(set(m["home"]) | set(m["away"]))
    ix = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(m), len(teams)))
    for k, (h, a) in enumerate(zip(m["home"], m["away"])):
        X[k, ix[h]] += 1
        X[k, ix[a]] -= 1
    # sum-to-zero: append a constraint row
    Xc = np.vstack([X, np.ones(len(teams))])
    rc = np.append(r - r.mean(), 0.0)
    b, *_ = np.linalg.lstsq(Xc, rc, rcond=None)
    games = np.abs(X).sum(0)
    return pd.Series(b, index=teams), pd.Series(games, index=teams), X


def perm_test(m, r):
    eff, n, X = team_effects(m, r)
    obs = eff.var()
    rng = np.random.default_rng(SEED)
    null = np.array([team_effects(m, rng.permutation(r))[0].var() for _ in range(DRAWS)])
    return eff, n, obs, float((null >= obs).mean()), float(np.median(null))


def main():
    av = json.load(open(os.path.join(ROOT, "model", "logit_avail.json")))
    sig = av["sigma"]
    m = A.population().reset_index(drop=True)
    m = m[np.isfinite(pd.to_numeric(m["lock_q_home"], errors="coerce"))].copy()
    P = m["lock_p_home"].astype(float).clip(1e-4, 1 - 1e-4)
    q = m["lock_q_home"].astype(float).clip(1e-4, 1 - 1e-4)
    m["model_m"] = sig * A.norm.ppf(P)
    m["mkt_m"] = sig * A.norm.ppf(q)
    y = m["margin"].to_numpy(float)
    rm = y - m["model_m"].to_numpy(float)
    rk = y - m["mkt_m"].to_numpy(float)
    L = []
    w = L.append
    w("2026-27 losses by team and by situation (research/loss_context.py).")
    w("Descriptive; chosen after seeing this season's losses. ± 95%. Margins in points at the")
    w(f"report route's sigma ({sig:.2f}); market = lock no-vig DraftKings price on the same scale.")
    w("")
    won = m["prediction_won"].astype(int)
    w(f"Graded native locks, games 1+, priced at the lock: n {len(m)}, model picks "
      f"{won.sum()}-{(1 - won).sum()}.")
    w(f"Mean absolute residual: model {np.abs(rm).mean():.2f}, market {np.abs(rk).mean():.2f}; "
      f"RMSE model {np.sqrt((rm ** 2).mean()):.2f}, market {np.sqrt((rk ** 2).mean()):.2f}.")
    w("")
    w(f"== 1. Team effects (sum-to-zero least squares; permutation test, {DRAWS} draws)")
    em, nm, om, pm, medm = perm_test(m, rm)
    ek, _, ok, pk, medk = perm_test(m, rk)
    w(f"  model : variance of team effects {om:6.2f} (null median {medm:6.2f}), p = {pm:.3f}")
    w(f"  market: variance of team effects {ok:6.2f} (null median {medk:6.2f}), p = {pk:.3f}")
    w("  p = share of shuffled seasons with at least this much team spread; small p = the misses")
    w("  cluster by team beyond noise. ~9 games a team: single teams are leads, not findings.")
    w("  Largest model team effects (points per game the team beat the model's margin):")
    order = em.abs().sort_values(ascending=False).index[:8]
    for t in order:
        w(f"    {t}  model {em[t]:+6.2f}  market {ek[t]:+6.2f}  (games {int(nm[t])})")
    w("")
    w("== 2. Situations: model accuracy beside the lock-market favourite, same games")
    pick_home = m["lock_p_home"].astype(float) >= .5
    fav_home = m["lock_q_home"].astype(float) >= .5
    hw = m["home_won"].astype(int) == 1
    model_right = (pick_home == hw)
    mkt_right = (fav_home == hw)
    sgn = np.where(pick_home, 1.0, -1.0)
    m["resid_pick"] = rm * sgn
    b2b_pick = np.where(pick_home, m["home_b2b"], m["away_b2b"]).astype(float) == 1
    b2b_opp = np.where(pick_home, m["away_b2b"], m["home_b2b"]).astype(float) == 1
    pp = np.where(pick_home, m["lock_p_home"].astype(float), 1 - m["lock_p_home"].astype(float))
    groups = [
        ("all", np.ones(len(m), bool)),
        ("model agrees with market favourite", pick_home == fav_home),
        ("model disagrees", pick_home != fav_home),
        ("pick at home", pick_home.to_numpy()),
        ("pick away", ~pick_home.to_numpy()),
        ("pick on a back-to-back", b2b_pick),
        ("opponent on a back-to-back", b2b_opp),
        ("model P 50-60%", pp < .6),
        ("model P 60-70%", (pp >= .6) & (pp < .7)),
        ("model P 70%+", pp >= .7),
        ("games 1-9", (m["gp"] < 10).to_numpy()),
        ("games 10+", (m["gp"] >= 10).to_numpy()),
    ]
    w(f"  {'situation':38s} {'n':>4s} {'model':>7s} {'market':>7s}  "
      f"resid toward pick (actual − model, pts)")
    for lab, sel in groups:
        sel = np.asarray(sel, bool)
        if sel.sum() < 5:
            w(f"  {lab:38s} {sel.sum():4d}  (too few)")
            continue
        mu, ci = A.mean_ci(m.loc[sel, "resid_pick"])
        w(f"  {lab:38s} {sel.sum():4d} {model_right[sel].mean():6.1%} {mkt_right[sel].mean():7.1%}"
          f"  {mu:+5.2f} ± {ci:4.2f}")
    w("  (resid toward pick > 0: the picked side did better than the model said)")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "RESULTS.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
