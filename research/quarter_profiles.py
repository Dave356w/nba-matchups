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
Decayed shapes (owner's request, 2026-11-07, added before their run): the
shape is an exponentially decayed mean of earlier games, half-life 5, 10,
25 or 50 games or none (the season-to-date mean of rule 3), either
restarting each season (within) or carrying on across seasons (across; the
offseason counts as one game). Predicted shape = S / (W + n0) x weight, S
and W the decayed sums of shapes and of weights. Every arm is fitted on
train (n0 on a grid, the weight by least squares) and one arm is chosen by
train squared error; only that arm is scored on test, by rules 3 and 4.
All arms are scored on the same games (both teams >= 10 earlier games this
season); the across arms are also shown on games 1-9 (information only).
Shown first: each arm's predictive correlation with the next game's shape,
by quarter (the decayed form of rule 2).
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
N0_GRID = (2, 5, 10, 20, 40, 80, 160, 320)
HALF_LIVES = (5, 10, 25, 50, None)          # None: no decay (equal weights)


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


def decayed_priors(T, half_life, across):
    """For each team-row: S (decayed sum of the team's earlier shapes s1..s4,
    the most recent weighted 1) and W (the same sum of weights), restarting
    each season (across=False) or carrying on (across=True)."""
    lam = 1.0 if half_life is None else 0.5 ** (1 / half_life)
    keys = ["team"] if across else ["season", "team"]
    Ts = T.sort_values(["team", "season", "date", "game_id"] if across else ["season", "team", "date", "game_id"])
    s = T[["s1", "s2", "s3", "s4"]].to_numpy()
    S, W = np.zeros((len(T), 4)), np.zeros(len(T))
    for pos in Ts.groupby(keys, sort=False).indices.values():
        acc, wacc = np.zeros(4), 0.0
        for i in Ts.index[pos]:
            S[i], W[i] = acc, wacc
            acc = lam * acc + s[i]
            wacc = lam * wacc + 1.0
    return S, W


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
    T["n"] = T.groupby(["season", "team"]).cumcount()
    hrow = T[T.side == "home"].set_index("game_id")["n"]
    arow = T[T.side == "away"].set_index("game_id")["n"]
    idx_h = pd.Series(T.index[T.side == "home"], index=T.game_id[T.side == "home"])
    idx_a = pd.Series(T.index[T.side == "away"], index=T.game_id[T.side == "away"])
    Gi = G.set_index("game_id")
    gids = Gi.index
    nh, na = hrow.loc[gids].to_numpy(), arow.loc[gids].to_numpy()
    ih, ia = idx_h.loc[gids].to_numpy(), idx_a.loc[gids].to_numpy()
    resid = np.column_stack([Gi[f"q{k}"] - A[k - 1] - B[k - 1] * Gi.mu for k in range(1, 5)])
    shape = resid - resid.mean(1, keepdims=True)
    season = Gi.season.to_numpy()
    est = (nh >= MIN_PRIOR) & (na >= MIN_PRIOR)
    early = (np.minimum(nh, na) >= 1) & ~est
    trm, tem = est & (season <= TRAIN_LAST), est & (season > TRAIN_LAST) & (season <= TEST_LAST)
    arms = {}
    for across in (False, True):
        for hl in HALF_LIVES:
            S, W = decayed_priors(T, hl, across)
            arms[(hl, across)] = (S[ih], W[ih], S[ia], W[ia])

    def name(arm):
        hl, across = arm
        return f"{'no decay' if hl is None else f'half-life {hl}'}, {'across seasons' if across else 'within season'}"

    def X(arm, n0):
        Sh, Wh, Sa, Wa = arms[arm]
        return Sh / (Wh + n0)[:, None] - Sa / (Wa + n0)[:, None]

    w("== 2b. Decayed shapes: predictive correlation with the next game's shape (both teams ≥ 10 games, 2016-17 .. 2025-26)")
    allm = est & (season <= TEST_LAST)
    for arm in arms:
        Sh, Wh, Sa, Wa = arms[arm]
        xm = Sh / np.maximum(Wh, 1e-9)[:, None] - Sa / np.maximum(Wa, 1e-9)[:, None]
        cs = []
        for k in range(4):
            r, lo, hi = corr_ci(xm[allm, k], shape[allm, k])
            cs.append(f"Q{k + 1} {r:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        w(f"  {name(arm):<32} " + "  ".join(cs))
    w(f"  (games {int(allm.sum())}; a real, decaying team shape would show correlations above 0)")
    w("")
    w("== 3. Forecastable? Each arm fitted on train (both teams ≥ 10 earlier games); one chosen by train error")
    fits = {}
    for arm in arms:
        best = None
        for n0 in N0_GRID:
            Xa, Y = X(arm, n0)[trm].ravel(), shape[trm].ravel()
            g = float(Xa @ Y / (Xa @ Xa))
            sse = float(((Y - g * Xa) ** 2).sum())
            if best is None or sse < best[2]:
                best = (n0, g, sse)
        n0, g, sse = best
        Xa = X(arm, n0)[trm].ravel()
        Y = shape[trm].ravel()
        se = math.sqrt(((Y - g * Xa) ** 2).sum() / (len(Y) - 1) / (Xa @ Xa))
        base = float((Y ** 2).sum())
        fits[arm] = (n0, g, sse)
        w(f"  {name(arm):<32} n0 {n0:>3}  weight {g:+.2f} ± {Z95 * se:.2f}   train squared error − line alone "
          f"{(sse - base) / trm.sum():+.3f} pts² per game")
    chosen = min(fits, key=lambda a: fits[a][2])
    n0, gam, _ = fits[chosen]
    w(f"  chosen on train: {name(chosen)} (n0 {n0}, weight {gam:.2f})")
    Xt = gam * X(chosen, n0)
    Rt, Pt = shape[tem], Xt[tem]
    d = ((Rt - Pt) ** 2).sum(1) - (Rt ** 2).sum(1)
    w(f"  test ({int(tem.sum())} games): squared error per game (four quarters) − line alone "
      f"{d.mean():+.3f} ± {Z95 * d.std(ddof=1) / math.sqrt(len(d)):.3f} pts²")
    for k in range(4):
        dk = (Rt[:, k] - Pt[:, k]) ** 2 - Rt[:, k] ** 2
        w(f"    Q{k + 1}: {dk.mean():+.3f} ± {Z95 * dk.std(ddof=1) / math.sqrt(len(dk)):.3f}   "
          f"(predicted shape sd {Pt[:, k].std():.2f} pts vs residual sd {resid[tem, k].std():.2f})")
    rem = resid[tem, 2:].sum(1)
    d2 = (rem - Pt[:, 2:].sum(1)) ** 2 - rem ** 2
    w(f"  second half (Q3 + Q4 margin) from halftime: {d2.mean():+.3f} ± {Z95 * d2.std(ddof=1) / math.sqrt(len(d2)):.3f} pts²")
    for arm in [a for a in arms if a[1]]:
        if arm == chosen or arm[0] in (10, 25):
            n0a, ga, _ = fits[arm]
            em = early & (season > TRAIN_LAST) & (season <= TEST_LAST)
            P = ga * X(arm, n0a)[em]
            de = ((shape[em] - P) ** 2).sum(1) - (shape[em] ** 2).sum(1)
            w(f"  information: {name(arm)} on games 1-9 ({int(em.sum())} test games): "
              f"{de.mean():+.3f} ± {Z95 * de.std(ddof=1) / math.sqrt(len(de)):.3f} pts²")
    w("")
    # ---- 4 ----
    w(f"== 4. Useful live? Deployed live win chance at quarter ends + c x remaining shape ({name(chosen)}) / sqrt(tau)")
    fit = live_wp.load()
    cum = Gi[["q1", "q2", "q3", "q4"]].cumsum(axis=1).to_numpy()
    y = (Gi.final.to_numpy() > 0).astype(float)
    rows = []
    for j, tau in ((0, .75), (1, .5), (2, .25)):
        a = np.interp(tau, fit["knots"], fit["a"])
        b = np.interp(tau, fit["knots"], fit["b"])
        rows.append(pd.DataFrame(dict(game_id=gids, season=season, tau=tau, keep=est,
                                      z0=a * cum[:, j] / math.sqrt(tau) + b * Gi.mu.to_numpy() * math.sqrt(tau),
                                      x=Xt[:, j + 1:].sum(1) / math.sqrt(tau), y=y)))
    L = pd.concat(rows, ignore_index=True)
    L = L[L.keep]
    Ltr, Lte = L[L.season <= TRAIN_LAST], L[(L.season > TRAIN_LAST) & (L.season <= TEST_LAST)]
    c = 0.0
    for _ in range(30):                                # probit with offset z0, one weight
        z = Ltr.z0 + c * Ltr.x
        p = np.clip(F.Phi(z), 1e-9, 1 - 1e-9)
        f = F.phi(z)
        g = (Ltr.x * f * (Ltr.y - p) / (p * (1 - p))).sum()
        h = (Ltr.x ** 2 * f * f / (p * (1 - p))).sum()
        c += g / h
    z = Ltr.z0 + c * Ltr.x
    p = np.clip(F.Phi(z), 1e-9, 1 - 1e-9)
    sc = pd.Series((Ltr.x * F.phi(z) * (Ltr.y - p) / (p * (1 - p))).values, index=Ltr.game_id.values)
    se_c = math.sqrt((sc.groupby(level=0).sum() ** 2).sum()) / h
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
