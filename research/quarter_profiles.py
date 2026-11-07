#!/usr/bin/env python3
"""Team scoring patterns by quarter: real, forecastable, useful live?
(owner's question, 2026-11-07).

  python research/quarter_profiles.py

Data: research/live_wp.py's caches, every regular-season game 2015-16 ..
2026-27 (no 2019-20) with ESPN play-by-play and ESPN's archived pregame
line mu (expected home margin). q_k = the home team's margin in quarter k
(k = 1..4, regulation).

Fixed before the run (95% intervals):
  1  Split of the expected margin: q_k = alpha_k + beta_k mu, by OLS.
  2  A team's quarter shape: its side of r_k = q_k - alpha_k - beta_k mu,
     centred across the four quarters of each game (so the line's total is
     untouched), averaged over its games in a season. It is real only if
     the split-half correlation (odd vs even games, same team-season) is
     above 0 with its interval. Shown beside the uncentred mean residual (a
     team beating its line overall), which an efficient market keeps at 0,
     and the year-to-year correlation.
  3  Forecastable: walk-forward within season, both teams with >= 10
     earlier games; predicted shape = n / (n + n0) x the mean of earlier
     games, n0 and a weight fitted on 2016-17 .. 2021-22 (train), quarter
     margins predicted on 2022-23 .. 2025-26 (test). Passes only if the
     test squared error per game falls with its interval below 0.
  4  Useful live: at the ends of Q1, Q2 and Q3, the deployed live win
     chance (model/live_wp.json) plus c x (the predicted shape's remaining
     quarters, home - away) / sqrt(tau), c fitted on train. Passes only if
     the test log loss falls with its interval below 0. (The deployed fit saw
     the test seasons; the added weight did not.)
Research only: nothing here reaches the model, the page, a test or a record.
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
import live_wp                                    # noqa: E402

OUT = ROOT / "diagnostics" / "quarter_profiles"
Z95 = 1.96
TRAIN_LAST, TEST_LAST = F.TRAIN_LAST, F.TEST_LAST
MIN_PRIOR = 10
N0_GRID = (5, 10, 20, 40, 80, 160)


def quarter_margins(G):
    rows = []
    for gid in G.game_id:
        pl = np.array(F.plays(gid), float)
        if not len(pl):
            rows.append([np.nan] * 4)
            continue
        m = pl[:, 2] - pl[:, 3]
        ends = []
        for k in (1, 2, 3, 4):
            sel = pl[:, 0] == k
            ends.append(m[sel][-1] if sel.any() else np.nan)
        rows.append(list(np.diff(np.r_[0.0, ends])))
    Q = pd.DataFrame(rows, columns=["q1", "q2", "q3", "q4"], index=G.index)
    return G.join(Q)


def ols(X, y):
    X = np.c_[np.ones(len(X)), X]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    V = np.linalg.inv(X.T @ X) * (r @ r) / (len(y) - X.shape[1])
    return b, np.sqrt(np.diag(V))


def corr_ci(x, y):
    r = np.corrcoef(x, y)[0, 1]
    z, s = np.arctanh(r), 1 / math.sqrt(len(x) - 3)
    return r, np.tanh(z - Z95 * s), np.tanh(z + Z95 * s)


def team_rows(G, R):
    """One row per team per game: season, team, date, its residuals r1..r4
    (its own side) and their centred shape s1..s4."""
    h = pd.DataFrame(R, columns=["r1", "r2", "r3", "r4"], index=G.index)
    home = G[["game_id", "season", "date", "home"]].rename(columns={"home": "team"}).join(h)
    away = G[["game_id", "season", "date", "away"]].rename(columns={"away": "team"}).join(-h)
    T = pd.concat([home.assign(side="home"), away.assign(side="away")], ignore_index=True)
    r = T[["r1", "r2", "r3", "r4"]].to_numpy()
    s = r - r.mean(1, keepdims=True)
    for k in range(4):
        T[f"s{k + 1}"] = s[:, k]
    T["rm"] = r.mean(1)
    return T.sort_values(["season", "team", "date", "game_id"]).reset_index(drop=True)


def prior_means(T):
    """Each team-game's mean shape over the team's earlier games that season
    (and the count n)."""
    g = T.groupby(["season", "team"])
    out = {"n": g.cumcount().to_numpy()}
    for k in range(1, 5):
        c = g[f"s{k}"].cumsum() - T[f"s{k}"]
        out[f"p{k}"] = (c / out["n"]).where(out["n"] > 0).to_numpy()
    return pd.DataFrame(out, index=T.index)


def main():
    lines = []
    w = lines.append
    G = F.games()
    G["mu"] = F.lines(G)
    G = G[np.isfinite(G.mu)].reset_index(drop=True)
    G = quarter_margins(G)
    ok = G[["q1", "q2", "q3", "q4"]].notna().all(axis=1)
    G = G[ok].reset_index(drop=True)
    w("Team scoring patterns by quarter: real, forecastable, useful live? Research only; ± is 95%.")
    w(f"Games with a line and four quarters: {len(G)} (2015-16 .. 2026-27, no 2019-20). q_k = home margin in quarter k.")
    w("")
    # ---- 1 ----
    w("== 1. How the expected margin splits by quarter (q_k = alpha_k + beta_k mu, all seasons)")
    A, B = [], []
    for k in range(1, 5):
        b, se = ols(G.mu.values[:, None], G[f"q{k}"].values)
        A.append(b[0]); B.append(b[1])
        w(f"  Q{k}: beta {b[1]:.3f} ± {Z95 * se[1]:.3f}  (share of the line {100 * b[1]:.0f}%)   alpha {b[0]:+.2f} ± {Z95 * se[0]:.2f} pts   "
          f"residual sd {np.std(G[f'q{k}'] - b[0] - b[1] * G.mu):.2f}")
    w(f"  sum of betas {sum(B):.3f} (1 = the quarters add up to the line); sum of alphas {sum(A):+.2f}")
    w("")
    R = np.column_stack([G[f"q{k}"].values - A[k - 1] - B[k - 1] * G.mu.values for k in range(1, 5)])
    T = team_rows(G, R)
    # ---- 2 ----
    w("== 2. Is a team's quarter shape real? Split-half (odd vs even games), same team-season")
    T["half"] = T.groupby(["season", "team"]).cumcount() % 2
    cols = [f"s{k}" for k in range(1, 5)] + ["rm"]
    H = T[T.season <= TEST_LAST].groupby(["season", "team", "half"])[cols].mean().unstack("half")
    for c, lab in zip(cols, ["Q1 shape", "Q2 shape", "Q3 shape", "Q4 shape", "overall vs line"]):
        r, lo, hi = corr_ci(H[(c, 0)], H[(c, 1)])
        rel = 2 * r / (1 + r) if r > 0 else 0.0
        sd_obs = T[T.season <= TEST_LAST].groupby(["season", "team"])[c].mean().std()
        w(f"  {lab:<16} r {r:+.3f} [{lo:+.3f}, {hi:+.3f}]   season reliability {rel:.2f}   "
          f"team-season sd {sd_obs:.2f} pts/game, of which persistent ~{sd_obs * math.sqrt(rel):.2f}   (team-seasons {len(H)})")
    S = T[T.season <= TEST_LAST].groupby(["season", "team"])[cols].mean().reset_index()
    nxt = S.assign(season=S.season + 1).merge(S, on=["season", "team"], suffixes=("_prev", ""))
    nxt = nxt[nxt.season != 2021]                     # no 2019-20 season to follow
    w("  year to year (same team, consecutive seasons):")
    for c, lab in zip(cols, ["Q1 shape", "Q2 shape", "Q3 shape", "Q4 shape", "overall vs line"]):
        r, lo, hi = corr_ci(nxt[c + "_prev"], nxt[c])
        w(f"   {lab:<16} r {r:+.3f} [{lo:+.3f}, {hi:+.3f}]   (pairs {len(nxt)})")
    w("")
    # ---- 3 ----
    w("== 3. Forecastable? Walk-forward within season, both teams ≥ 10 earlier games")
    P = prior_means(T)
    T = T.join(P)
    hm = T[T.side == "home"].set_index("game_id")
    aw = T[T.side == "away"].set_index("game_id")
    D = G.set_index("game_id").join(hm[["n", "p1", "p2", "p3", "p4"]]).join(
        aw[["n", "p1", "p2", "p3", "p4"]], rsuffix="_a")
    D = D[(D.n >= MIN_PRIOR) & (D.n_a >= MIN_PRIOR)].copy()
    tr, te = D[D.season <= TRAIN_LAST], D[(D.season > TRAIN_LAST) & (D.season <= TEST_LAST)]

    def x(d, n0):
        sh = lambda n: n / (n + n0)          # noqa: E731
        return np.column_stack([sh(d.n) * d[f"p{k}"] - sh(d.n_a) * d[f"p{k}_a"] for k in range(1, 5)])

    def resid(d):
        return np.column_stack([d[f"q{k}"] - A[k - 1] - B[k - 1] * d.mu for k in range(1, 5)])

    best = None
    for n0 in N0_GRID:
        X, Y = x(tr, n0).ravel(), resid(tr).ravel()
        g = float(X @ Y / (X @ X))
        sse = float(((Y - g * X) ** 2).sum())
        if best is None or sse < best[2]:
            best = (n0, g, sse)
    n0, gam, _ = best
    X, Y = x(tr, n0).ravel(), resid(tr).ravel()
    se_g = math.sqrt(((Y - gam * X) ** 2).sum() / (len(Y) - 1) / (X @ X))
    w(f"  fitted on train ({len(tr)} games): shrinkage n0 {n0} games, weight {gam:.2f} ± {Z95 * se_g:.2f} (1 = take the shrunk shape at face value)")
    Rt, Xt = resid(te), gam * x(te, n0)
    d = ((Rt - Xt) ** 2).sum(1) - (Rt ** 2).sum(1)
    w(f"  test ({len(te)} games): squared error per game (four quarters) {np.mean((Rt ** 2).sum(1)):.1f} pts² with the line alone; "
      f"shape − line alone {d.mean():+.3f} ± {Z95 * d.std(ddof=1) / math.sqrt(len(d)):.3f} pts²")
    for k in range(4):
        dk = (Rt[:, k] - Xt[:, k]) ** 2 - Rt[:, k] ** 2
        w(f"    Q{k + 1}: {dk.mean():+.3f} ± {Z95 * dk.std(ddof=1) / math.sqrt(len(dk)):.3f}   "
          f"(predicted shape sd {Xt[:, k].std():.2f} pts vs residual sd {Rt[:, k].std():.2f})")
    rem = Rt[:, 2:].sum(1)
    d2 = (rem - Xt[:, 2:].sum(1)) ** 2 - rem ** 2
    w(f"  second half (Q3 + Q4 margin) from halftime: {d2.mean():+.3f} ± {Z95 * d2.std(ddof=1) / math.sqrt(len(d2)):.3f} pts²")
    w("")
    # ---- 4 ----
    w("== 4. Useful live? Deployed live win chance at quarter ends + c x remaining shape / sqrt(tau)")
    fit = live_wp.load()
    Dall = pd.concat([tr, te])
    Xs = gam * x(Dall, n0)
    cum = Dall[["q1", "q2", "q3", "q4"]].cumsum(1).to_numpy()
    rows = []
    for j, tau in ((0, .75), (1, .5), (2, .25)):
        lead = cum[:, j]
        a = np.interp(tau, fit["knots"], fit["a"])
        b = np.interp(tau, fit["knots"], fit["b"])
        z0 = a * lead / math.sqrt(tau) + b * Dall.mu.values * math.sqrt(tau)
        shift = Xs[:, j + 1:].sum(1)
        rows.append(pd.DataFrame(dict(game_id=Dall.index, season=Dall.season.values, tau=tau, z0=z0,
                                      x=shift / math.sqrt(tau), y=(Dall.final.values > 0).astype(float))))
    L = pd.concat(rows, ignore_index=True)
    Ltr, Lte = L[L.season <= TRAIN_LAST], L[(L.season > TRAIN_LAST) & (L.season <= TEST_LAST)]
    c, lo = 0.0, None
    for _ in range(30):                                # probit with offset z0, one weight
        z = Ltr.z0 + c * Ltr.x
        p = np.clip(F.Phi(z), 1e-9, 1 - 1e-9)
        f = F.phi(z)
        g = (Ltr.x * f * (Ltr.y - p) / (p * (1 - p))).sum()
        h = (Ltr.x ** 2 * f * f / (p * (1 - p))).sum()
        c += g / h
    sc = (Ltr.x * F.phi(Ltr.z0 + c * Ltr.x) * (Ltr.y - F.Phi(Ltr.z0 + c * Ltr.x))
          / (F.Phi(Ltr.z0 + c * Ltr.x) * (1 - F.Phi(Ltr.z0 + c * Ltr.x))))
    G_sc = pd.Series(sc.values, index=Ltr.game_id.values).groupby(level=0).sum()
    se_c = math.sqrt((G_sc ** 2).sum()) / h
    w(f"  c {c:+.4f} ± {Z95 * se_c:.4f} (train; the lead's own weight a(tau) is ~0.05)")
    p0, p1 = F.Phi(Lte.z0), F.Phi(Lte.z0 + c * Lte.x)
    dl = pd.Series(F.ll(p1, Lte.y.values) - F.ll(p0, Lte.y.values), index=Lte.game_id.values).groupby(level=0).mean()
    w(f"  test log loss, with − without: {dl.mean():+.5f} ± {Z95 * dl.std(ddof=1) / math.sqrt(len(dl)):.5f} (games {len(dl)})")
    for tau, lab in ((.75, "end Q1"), (.5, "half"), (.25, "end Q3")):
        m = (Lte.tau == tau).values
        dd = F.ll(p1[m], Lte.y.values[m]) - F.ll(p0[m], Lte.y.values[m])
        w(f"    {lab:<7} {dd.mean():+.5f} ± {Z95 * dd.std(ddof=1) / math.sqrt(len(dd)):.5f}")
    w("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
