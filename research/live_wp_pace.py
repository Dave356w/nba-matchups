#!/usr/bin/env python3
"""Can teams' scoring patterns modulate the live win chance through its
volatility? (owner's question, 2026-11-07; research/quarter_profiles.py
showed they add nothing to its expected margin.)

  python research/live_wp_pace.py

The score's swings over the time left grow with the points still to be
scored. Arms scale the live z by v^-gamma, v = sqrt(R / R0): R the points
both teams are expected to score in the time left, R0 the same at league
average. Walk-forward, from games strictly before the slate, both teams
with >= 10 earlier games this season:
  VT  pace only: each team's season-to-date points scored and allowed per
      game, the game's expected total = mean of (home scored, away
      allowed) + mean of (away scored, home allowed), spread over the
      quarters in the league's season-to-date shares.
  VQ  team quarter patterns: the same, quarter by quarter (each team's
      season-to-date points scored and allowed in quarter k).
In the current quarter its unplayed share counts; overtime uses Q4's
per-minute rate. Base: the deployed live form (spline a, b) refitted on
2015-16 .. 2021-22 (train); gamma on a grid on train (theory: 1); scored on
2022-23 .. 2025-26 (test) by log loss, each game's states averaged, then
games averaged, ± 95% across games. Fixed before the run: pace helps only
if VT - base is below 0 with its interval; team quarter patterns add only if
VQ - VT is below 0 with its interval. Research only.
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

OUT = ROOT / "diagnostics" / "live_wp_pace"
Z95 = 1.96
MIN_PRIOR = 10
GAMMAS = np.round(np.arange(0, 2.01, 0.1), 2)


def quarter_points(G):
    rows = []
    for gid in G.game_id:
        pl = np.array(F.plays(gid), float)
        ends = []
        for k in (1, 2, 3, 4):
            sel = pl[:, 0] == k if len(pl) else np.array([], bool)
            ends.append(pl[sel][-1, 2:4] if sel.any() else [np.nan, np.nan])
        d = np.diff(np.vstack([[0, 0], ends]), axis=0)
        rows.append(list(d[:, 0]) + list(d[:, 1]))
    cols = [f"h{k}" for k in range(1, 5)] + [f"a{k}" for k in range(1, 5)]
    return G.join(pd.DataFrame(rows, columns=cols, index=G.index))


def prior_features(G):
    """Per game: each team's season-to-date points scored / allowed by quarter
    (earlier games only), its count of earlier games, and the league's
    season-to-date points per team-quarter (earlier dates only)."""
    rows = []
    for side, opp, team in (("h", "a", "home"), ("a", "h", "away")):
        d = G[["game_id", "season", "date", team]].rename(columns={team: "team"}).copy()
        for k in range(1, 5):
            d[f"o{k}"] = G[f"{side}{k}"].to_numpy()
            d[f"d{k}"] = G[f"{opp}{k}"].to_numpy()
        d["side"] = side
        rows.append(d)
    T = pd.concat(rows, ignore_index=True).sort_values(["season", "team", "date", "game_id"])
    g = T.groupby(["season", "team"])
    T["n"] = g.cumcount()
    for c in [f"o{k}" for k in range(1, 5)] + [f"d{k}" for k in range(1, 5)]:
        T["p" + c] = (g[c].cumsum() - T[c]) / T["n"].where(T["n"] > 0)
    # league: per team-quarter points on dates strictly before this one
    day = T.groupby(["season", "date"])[[f"o{k}" for k in range(1, 5)]].agg(["sum", "count"])
    out = {}
    for k in range(1, 5):
        s = day[(f"o{k}", "sum")].groupby(level=0).cumsum() - day[(f"o{k}", "sum")]
        c = day[(f"o{k}", "count")].groupby(level=0).cumsum() - day[(f"o{k}", "count")]
        out[f"L{k}"] = (s / c.where(c > 0))
    L = pd.DataFrame(out)
    h = T[T.side == "h"].set_index("game_id")
    a = T[T.side == "a"].set_index("game_id")
    P = G.set_index("game_id")[["season", "date"]].copy()
    P["n_h"], P["n_a"] = h.n, a.n
    for k in range(1, 5):
        # expected points in quarter k: home = mean(home scored, away allowed), away likewise
        P[f"e{k}"] = (h[f"po{k}"] + a[f"pd{k}"]) / 2 + (a[f"po{k}"] + h[f"pd{k}"]) / 2
    P = P.join(L, on=["season", "date"])
    tot = P[[f"e{k}" for k in range(1, 5)]].sum(axis=1)
    ltot = P[[f"L{k}" for k in range(1, 5)]].sum(axis=1) * 2
    for k in range(1, 5):
        P[f"t{k}"] = tot * (2 * P[f"L{k}"] / ltot)      # pace only: league quarter shares
        P[f"l{k}"] = 2 * P[f"L{k}"]                      # league average, both teams
    return P


def remaining(S, E):
    """Expected points left at each state from per-quarter expectations E
    (n_states x 4): the unplayed share of each regulation quarter; in
    overtime, Q4's per-minute rate over the minutes left."""
    rem = S.tau.to_numpy() * 2880.0
    ot = (S.part == "OT").to_numpy()
    R = np.zeros(len(S))
    for k in range(1, 5):
        f = np.clip((rem - (4 - k) * 720) / 720, 0, 1)
        R += np.where(ot, 0.0, f * E[:, k - 1])
    return np.where(ot, E[:, 3] * rem / 720.0, R)


def main():
    lines = []
    w = lines.append
    G = F.games()
    G["mu"] = F.lines(G)
    G = G[np.isfinite(G.mu)].reset_index(drop=True)
    G = quarter_points(G)
    G = G[G[[f"h{k}" for k in range(1, 5)] + [f"a{k}" for k in range(1, 5)]].notna().all(axis=1)].reset_index(drop=True)
    P = prior_features(G)
    S, bad = F.states(G, G.mu.values)
    S["season"] = P.season.loc[S.game_id].to_numpy()
    keep = ((P.n_h.loc[S.game_id] >= MIN_PRIOR) & (P.n_a.loc[S.game_id] >= MIN_PRIOR)).to_numpy()
    S = S[keep & (S.season <= F.TEST_LAST).to_numpy()].reset_index(drop=True)
    Pi = P.loc[S.game_id]
    E_q = Pi[[f"e{k}" for k in range(1, 5)]].to_numpy()
    E_t = Pi[[f"t{k}" for k in range(1, 5)]].to_numpy()
    E_l = Pi[[f"l{k}" for k in range(1, 5)]].to_numpy()
    R0 = remaining(S, E_l)
    v = {"VT": np.sqrt(remaining(S, E_t) / R0), "VQ": np.sqrt(remaining(S, E_q) / R0)}
    tr = (S.season <= F.TRAIN_LAST).to_numpy()
    te = ~tr
    w("Can teams' scoring patterns modulate the live win chance through its volatility? Research only; ± is 95%.")
    w(f"States of games with both teams ≥ {MIN_PRIOR} earlier games: train {S[tr].game_id.nunique()} games, "
      f"test {S[te].game_id.nunique()} (2022-23 .. 2025-26).")
    gt = Pi.groupby(level=0).first()
    w(f"  expected game total (team season-to-date): mean {gt[[f'e{k}' for k in range(1, 5)]].sum(axis=1).mean():.1f}, "
      f"sd {gt[[f'e{k}' for k in range(1, 5)]].sum(axis=1).std():.1f} pts; v at the half: VT sd "
      f"{v['VT'][np.isclose(S.tau, .5)].std():.3f}, VQ sd {v['VQ'][np.isclose(S.tau, .5)].std():.3f}")
    w("")
    beta, _ = F.probit(F.design(S[tr], "spline"), S.y.values[tr], S.w.values[tr], S.game_id.values[tr])
    z = F.design(S, "spline") @ beta
    y, wt = S.y.to_numpy(), S.w.to_numpy()

    def wll(p, m):
        return float(np.average(F.ll(p[m], y[m]), weights=wt[m]))

    P_base = F.Phi(z)
    P_arm, gam = {}, {}
    w("== 1. Fits (train): gamma on the grid 0 .. 2 (0 = no modulation, 1 = theory)")
    for arm in ("VT", "VQ"):
        best = min(GAMMAS, key=lambda g: wll(F.Phi(z * v[arm] ** -g), tr))
        gam[arm] = best
        P_arm[arm] = F.Phi(z * v[arm] ** -best)
        w(f"  {arm}: gamma {best:.1f}   train log loss {wll(P_arm[arm], tr):.5f} vs base {wll(P_base, tr):.5f}")
        w("      train log loss − base by gamma: " + "  ".join(
            f"{g:.1f}: {wll(F.Phi(z * v[arm] ** -g), tr) - wll(P_base, tr):+.5f}" for g in (0.5, 1.0, 1.5, 2.0)))
    w("  information (not the rule): at the theory value gamma = 1, test "
      + "; ".join(f"{arm} − base {F.diff_txt(S[te], F.Phi(z * v[arm] ** -1.0)[te], P_base[te])}" for arm in ("VT", "VQ")))
    w("")
    w("== 2. Test (held-out): log loss differences, negative = better")
    St = S[te]
    for a, b, lab in ((P_arm["VT"], P_base, "VT − base (pace)"), (P_arm["VQ"], P_base, "VQ − base"),
                      (P_arm["VQ"], P_arm["VT"], "VQ − VT (team quarter patterns)")):
        w(f"  {lab:<34} all minutes {F.diff_txt(St, a[te], b[te])}")
        for part in ("Q1", "Q2", "Q3", "Q4 to 2:00", "last 2:00", "OT"):
            m = (St.part == part).to_numpy()
            if m.any():
                w(f"      {part:<11} {F.diff_txt(St[m], a[te][m], b[te][m])}")
    w(f"  base test log loss {np.mean(F.per_game(St, F.ll(P_base[te], y[te]))):.4f}")
    # extremes: the fastest and slowest tenth of games by expected total
    gt_test = gt.loc[St.game_id.unique()]
    tot = gt_test[[f"e{k}" for k in range(1, 5)]].sum(axis=1)
    for q, lab in ((tot >= tot.quantile(.9), "fastest tenth"), (tot <= tot.quantile(.1), "slowest tenth")):
        m = St.game_id.isin(tot.index[q]).to_numpy()
        w(f"  {lab:<14} ({int(q.sum())} games): VT − base {F.diff_txt(St[m], P_arm['VT'][te][m], P_base[te][m])}")
    w("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
