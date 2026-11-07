#!/usr/bin/env python3
"""Does the model's pregame margin add to the live win chance? (owner's
question, 2026-11-07; live_wp.py, research/live_wp.py).

  python research/live_wp_model.py

The live win chance uses the market's pregame line only:
  z = a(tau) lead / sqrt(tau) + b(tau) mu sqrt(tau)
Arms add the model's disagreement with the line, d = (m - mu) sqrt(tau),
m = the model's predicted home margin:
  M1  + c d                       (one weight)
  M3  + c(tau) d, c piecewise linear on tau = 0, 0.5, 1
Fixed before the run: the same states, line and chronological split as
research/live_wp.py (fitted 2016-17 .. 2021-22, tested 2022-23 .. 2025-26;
2015-16 has no model prediction); the model margin is the walk-forward
prediction under production routing (diagnostics/margin_form, arm B, v10
with report route D; each season fitted on earlier seasons only, so
hindsight, not native). The model adds signal only if an arm's held-out
log loss minus the base is below 0 with its 95% interval. Also shown: the
fitted weights, the pregame analogue (tau = 1) and results by part of the
game and by stage (games 1-9 / 10+). 2026-27 to date is information only.
± is 95% across games (weights: game-clustered). Research only.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("live_wp_fit", ROOT / "research" / "live_wp.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

PRED = ROOT / "diagnostics" / "margin_form" / "predictions_2017-2027.csv"
OUT = ROOT / "diagnostics" / "live_wp_model"
C_KNOTS = [0.0, 0.5, 1.0]
Z95 = 1.96


def model_margins(G):
    p = pd.read_csv(PRED)
    p = p.rename(columns={"slate_date": "date"})[["date", "home", "away", "m_B", "gp_home", "gp_away"]]
    return G.merge(p, on=["date", "home", "away"], how="left")


def design(S, arm):
    X = F.design(S, "spline")
    d = (S.m.values - S.mu.values) * np.sqrt(S.tau.values)
    if arm == "base":
        return X
    if arm == "M1":
        return np.c_[X, d]
    H = np.column_stack([np.interp(S.tau.values, C_KNOTS, np.eye(len(C_KNOTS))[j]) for j in range(len(C_KNOTS))])
    return np.c_[X, H * d[:, None]]


def fit(S, arm):
    return F.probit(design(S, arm), S.y.values, S.w.values, S.game_id.values)


def pred(S, arm, beta):
    return F.Phi(design(S, arm) @ beta)


def main():
    lines = []
    w = lines.append
    G = F.games()
    mu = F.lines(G)
    G = model_margins(G.assign(mu=mu))
    n_all = len(G)
    G = G[np.isfinite(G.mu) & G.m_B.notna()].reset_index(drop=True)
    S, bad = F.states(G, G.mu.values)
    info = G.set_index("game_id")
    S["season"] = info.season.loc[S.game_id].values
    S["m"] = info.m_B.loc[S.game_id].values
    gp = np.fmin(info.gp_home, info.gp_away)
    S["stage"] = np.where(gp.loc[S.game_id].values >= 10, "games 10+",
                          np.where(gp.loc[S.game_id].values >= 1, "games 1-9", "game 0"))
    tr = S[S.season <= F.TRAIN_LAST]
    te = S[(S.season > F.TRAIN_LAST) & (S.season <= F.TEST_LAST)]
    cur = S[S.season > F.TEST_LAST]
    gm = G.drop_duplicates("game_id").set_index("game_id")
    w("Does the model's pregame margin add to the live win chance? Research only; ± is 95%.")
    w("Model margin: walk-forward predictions under production routing (diagnostics/margin_form, arm B):")
    w("hindsight, each season fitted on earlier seasons only. Line: ESPN's archived pregame line (as research/live_wp.py).")
    w("")
    w("== 0. Data")
    w(f"  games in the box files {n_all}; with a line and a model margin {len(G)}; with states {S.game_id.nunique()} "
      f"({bad} dropped: play-by-play final != box final)")
    w(f"  train 2016-17 .. 2021-22: {tr.game_id.nunique()} games; test 2022-23 .. 2025-26: {te.game_id.nunique()}; "
      f"2026-27 to date: {cur.game_id.nunique()}")
    dis = (gm.m_B - gm.mu).abs()
    w(f"  |model margin − line|: median {dis.median():.1f} pts, 90th percentile {dis.quantile(.9):.1f}")
    w("")
    # pregame analogue: probit of the outcome on mu and the disagreement, one row per game
    w("== 1. Pregame (tau = 1): outcome on the line and the model's disagreement, train")
    g0 = gm.loc[gm.season <= F.TRAIN_LAST]
    X = np.c_[g0.mu.values, (g0.m_B - g0.mu).values]
    y = (g0.final > 0).astype(float).values
    beta, V = F.probit(X, y, np.ones(len(y)), g0.index.values)
    se = Z95 * np.sqrt(np.diag(V))
    w(f"  weight on the line {beta[0]:.4f} ± {se[0]:.4f}; on model − line {beta[1]:+.4f} ± {se[1]:.4f} "
      f"(ratio {beta[1] / beta[0]:+.2f}: 0 = the model adds nothing to the line) (games {len(y)})")
    t0 = gm.loc[(gm.season > F.TRAIN_LAST) & (gm.season <= F.TEST_LAST)]
    yt = (t0.final > 0).astype(float).values
    k_line, _ = F.probit(g0.mu.values[:, None], y, np.ones(len(y)), g0.index.values)
    k_model, _ = F.probit(g0.m_B.values[:, None], y, np.ones(len(y)), g0.index.values)
    p_line, p_model = F.Phi(k_line[0] * t0.mu.values), F.Phi(k_model[0] * t0.m_B.values)
    d = F.ll(p_model, yt) - F.ll(p_line, yt)
    w(f"  test seasons, pregame log loss: line alone {F.ll(p_line, yt).mean():.4f}, model alone "
      f"{F.ll(p_model, yt).mean():.4f}; model − line {d.mean():+.4f} ± {Z95 * d.std(ddof=1) / math.sqrt(len(d)):.4f} "
      f"(games {len(yt)})")
    w("")
    w("== 2. Fits (train)")
    fits = {}
    for arm in ("base", "M1", "M3"):
        beta, V = fit(tr, arm)
        fits[arm] = beta
        if arm != "base":
            k = len(beta) - (1 if arm == "M1" else len(C_KNOTS))
            se = Z95 * np.sqrt(np.diag(V))[k:]
            lab = ["c"] if arm == "M1" else [f"c(tau {t:g})" for t in C_KNOTS]
            w(f"  {arm}: " + "   ".join(f"{l} {b:+.4f} ± {s:.4f}" for l, b, s in zip(lab, beta[k:], se))
              + f"   (line weight b(1) {beta[len(F.KNOTS) * 2 - 1]:.4f})")
    w("")
    w("== 3. Test (held-out seasons): arm − base log loss (negative = the model adds)")
    P = {arm: pred(te, arm, fits[arm]) for arm in fits}
    for arm in ("M1", "M3"):
        w(f"  {arm}: all minutes {F.diff_txt(te, P[arm], P['base'])}")
        for part in ("Q1", "Q2", "Q3", "Q4 to 2:00", "last 2:00", "OT"):
            m = (te.part == part).values
            if m.any():
                w(f"      {part:<11} {F.diff_txt(te[m], P[arm][m], P['base'][m])}")
        for stage in ("games 1-9", "games 10+"):
            m = (te.stage == stage).values
            if m.any():
                w(f"      {stage:<11} {F.diff_txt(te[m], P[arm][m], P['base'][m])}  (games {te[m].game_id.nunique()})")
    base_ll = np.mean(F.per_game(te, F.ll(P["base"], te.y.values)))
    w(f"  base log loss {base_ll:.4f}; M1 {np.mean(F.per_game(te, F.ll(P['M1'], te.y.values))):.4f}; "
      f"M3 {np.mean(F.per_game(te, F.ll(P['M3'], te.y.values))):.4f}")
    big = te.assign(dis=(te.m - te.mu).abs())
    m = (big.dis >= 4).values
    w(f"  games where the model and the line differ by ≥ 4 pts ({big[m].game_id.nunique()} games): "
      f"M1 − base {F.diff_txt(big[m], P['M1'][m], P['base'][m])}")
    if len(cur):
        Pc = {arm: pred(cur, arm, fits[arm]) for arm in ("base", "M1")}
        w(f"  2026-27 to date (information only, {cur.game_id.nunique()} games): M1 − base {F.diff_txt(cur, Pc['M1'], Pc['base'])}")
    w("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
