#!/usr/bin/env python3
"""Comebacks: is there a team skill in playing from a live win chance under
50% to over 80%? (owner's question, 2026-11-07.)

  python research/comebacks.py

Data: research/live_wp.py's caches and states (every 30 s of game time,
every 5 s in the last 2 minutes; regulation and overtime), 2015-16 ..
2025-26 (no 2019-20). A team's live win chance p is the deployed live
formula (model/live_wp.json) from its side. A trailing episode: the first
state where p < 0.5 (p0); it "comes back" if a later state has p > 0.8.
For an underdog at tip that state is 30 s in. Leading episodes mirror it.

Fixed before the run (95% intervals):
  1  Calibration: if p is a well-calibrated martingale, a team first under
     50% at p0 later passes 80% with probability ~p0 / 0.8 and wins with
     probability p0. Observed against expected, overall, by p0 band and
     separately for underdogs at tip and favourites who fall behind.
  2  Team skill: per team-season, the mean of (came back - p0 / 0.8) and of
     (won - p0) over its trailing episodes; and of (won - p0') over its
     leading episodes (holding a lead). Real only if the split-half
     correlation (odd vs even episodes, same team-season) is above 0 with
     its interval; also year to year.
  3  Useful live: walk-forward within season, each team's earlier trailing
     and leading residuals (won - p0, shrunk n / (n + n0)); at a state where
     team A trails team B, x = comeback(A) - hold(B) (sign flipped for the
     home side), added as c x to the live z. Base: the live form refitted on
     2015-16 .. 2021-22; n0 and c fitted there; scored on 2022-23 .. 2025-26
     by log loss (games averaged, ± 95%). Passes only if below 0 with its
     interval.
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

OUT = ROOT / "diagnostics" / "comebacks"
Z95 = 1.96
LAST = F.TEST_LAST
N0_GRID = (5, 10, 20, 40, 80, 160)
HI = 0.8


def corr_ci(x, y):
    r = np.corrcoef(x, y)[0, 1]
    z, s = np.arctanh(r), 1 / math.sqrt(len(x) - 3)
    return r, np.tanh(z - Z95 * s), np.tanh(z + Z95 * s)


def episodes(S, p, G):
    """One row per team per game: first state under 0.5 (trailing) and
    first state over 0.5 (leading), whether it later passed 0.8 (or fell
    under 0.2), and the result, from that team's side."""
    gid = S.game_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(gid[1:] != gid[:-1]) + 1]
    ends = np.r_[starts[1:], len(gid)]
    info = G.set_index("game_id")
    rows = []
    for a, b in zip(starts, ends):
        g = gid[a]
        ph = p[a:b]
        y = float(S.y.iat[a])
        row = info.loc[g]
        for side, team, opp, pp, yy in (("home", row.home, row.away, ph, y), ("away", row.away, row.home, 1 - ph, 1 - y)):
            fav = pp[0] >= 0.5
            under = np.flatnonzero(pp < 0.5)
            over = np.flatnonzero(pp > 0.5)
            rec = dict(game_id=g, season=row.season, date=row.date, team=team, opp=opp, side=side, won=yy,
                       fav_tip=fav, p_tip=pp[0])
            if len(under):
                i = under[0]
                rec.update(trail=True, p0=pp[i], back=bool((pp[i:] > HI).any()), t_idx=a + i)
            else:
                rec.update(trail=False, p0=np.nan, back=False, t_idx=-1)
            if len(over):
                j = over[0]
                rec.update(lead=True, q0=pp[j], blown=bool((pp[j:] < 1 - HI).any()), l_idx=a + j)
            else:
                rec.update(lead=False, q0=np.nan, blown=False, l_idx=-1)
            rows.append(rec)
    E = pd.DataFrame(rows)
    E["e80"] = E.back - E.p0 / HI
    E["ew"] = E.won - E.p0
    E["hw"] = E.won - E.q0
    return E


def main():
    lines = []
    w = lines.append
    G = F.games()
    G["mu"] = F.lines(G)
    G = G[np.isfinite(G.mu) & (G.season <= LAST)].reset_index(drop=True)
    S, bad = F.states(G, G.mu.values)
    S["season"] = G.set_index("game_id").season.loc[S.game_id].to_numpy()
    fit = live_wp.load()
    t = S.tau.to_numpy()
    a = np.interp(t, fit["knots"], fit["a"])
    b = np.interp(t, fit["knots"], fit["b"])
    p = F.Phi(a * S.lead.to_numpy() / np.sqrt(t) + b * S.mu.to_numpy() * np.sqrt(t))
    E = episodes(S, p, G)
    T = E[E.trail]
    w("Comebacks: is there a team skill in going from a live win chance under 50% to over 80%? Research only; ± is 95%.")
    w(f"Games {S.game_id.nunique()} (2015-16 .. 2025-26, no 2019-20); team-games {len(E)}; trailing episodes {len(T)}.")
    w("Live win chance: the deployed live formula from each team's side; states every 30 s (5 s in the last 2 minutes).")
    w("")
    # ---- 1 ----
    w("== 1. Calibration (league-wide): came back past 80% and won, against the live chance at the first state under 50%")
    w(f"  {'group':<34}{'episodes':>9}{'mean p0':>9}{'expected >80%':>15}{'observed >80%':>15}{'expected win':>14}{'observed win':>14}")

    def calib(d, lab):
        n = len(d)
        o80, e80 = d.back.mean(), (d.p0 / HI).mean()
        ow, ew = d.won.mean(), d.p0.mean()
        se80 = math.sqrt(o80 * (1 - o80) / n)
        sew = math.sqrt(ow * (1 - ow) / n)
        w(f"  {lab:<34}{n:>9}{d.p0.mean():>9.3f}{100 * e80:>14.1f}%{100 * o80:>9.1f}% ±{100 * Z95 * se80:4.1f}"
          f"{100 * ew:>13.1f}%{100 * ow:>8.1f}% ±{100 * Z95 * sew:4.1f}")
    calib(T, "all trailing episodes")
    calib(T[~T.fav_tip], "underdogs at tip")
    calib(T[T.fav_tip], "favourites who fell behind")
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5)):
        calib(T[(T.p0 >= lo) & (T.p0 < hi)], f"p0 {lo:.1f}–{hi:.1f}")
    late = int(((T.won == 1) & ~T.back).sum())
    w(f"  Past 80% runs below p0 / 0.8 because the states are sampled: {late} of {int(T.won.sum())} comeback wins never show")
    w(f"  over 80% on the grid (decided in the last seconds); counting them gives {100 * (T.back.sum() + late) / len(T):.1f}%, the rest")
    w("  is a jumpy score sampled every 30 s. The win rate, the clean test, matches p0.")
    w("")
    # ---- 2 ----
    w("== 2. Team skill: split-half (odd vs even episodes, same team-season) and year to year")
    for col, src, lab in (("e80", "trail", "comeback past 80% − expected"), ("ew", "trail", "comeback win − p0"),
                          ("hw", "lead", "holding a lead: win − p0'")):
        d = E[E[src]].sort_values(["season", "team", "date", "game_id"]).copy()
        d["half"] = d.groupby(["season", "team"]).cumcount() % 2
        H = d.groupby(["season", "team", "half"])[col].mean().unstack("half").dropna()
        r, lo, hi = corr_ci(H[0], H[1])
        m = d.groupby(["season", "team"])[col].agg(["mean", "size"]).reset_index()
        nxt = m.assign(season=m.season + 1).merge(m, on=["season", "team"], suffixes=("_prev", ""))
        nxt = nxt[nxt.season != 2021]
        ry, loy, hiy = corr_ci(nxt["mean_prev"], nxt["mean"])
        w(f"  {lab:<30} split-half r {r:+.3f} [{lo:+.3f}, {hi:+.3f}] (team-seasons {len(H)}, ~{m['size'].mean():.0f} episodes each)   "
          f"year to year r {ry:+.3f} [{loy:+.3f}, {hiy:+.3f}] (pairs {len(nxt)})")
    tm = T.groupby(["season", "team"]).agg(n=("back", "size"), back=("back", "sum"), exp=("p0", lambda s: (s / HI).sum()))
    tm["diff"] = tm.back - tm.exp
    w(f"  team-season comebacks past 80% minus expected: sd {tm['diff'].std():.2f} episodes; "
      f"sd expected from chance alone {math.sqrt((T.p0 / HI * (1 - T.p0 / HI)).groupby([T.season, T.team]).sum().mean()):.2f}")
    w("")
    # ---- 3 ----
    w("== 3. Useful live? Walk-forward comeback / hold records added to the live z")
    E = E.sort_values(["season", "team", "date", "game_id"]).reset_index(drop=True)
    g = E.groupby(["season", "team"])
    for col, flag, out in (("ew", "trail", "c"), ("hw", "lead", "h")):
        val = E[col].where(E[flag], 0.0).fillna(0.0)
        cnt = E[flag].astype(float)
        E[f"{out}_sum"] = val.groupby([E.season, E.team]).cumsum() - val
        E[f"{out}_n"] = cnt.groupby([E.season, E.team]).cumsum() - cnt
    eh = E[E.side == "home"].set_index("game_id")
    ea = E[E.side == "away"].set_index("game_id")
    gid = S.game_id.to_numpy()
    tr = (S.season <= F.TRAIN_LAST).to_numpy()
    te = ~tr
    beta, _ = F.probit(F.design(S[tr], "spline"), S.y.values[tr], S.w.values[tr], gid[tr])
    z = F.design(S, "spline") @ beta
    ptr = F.Phi(z)
    home_trails = ptr < 0.5

    def x_of(n0):
        def sk(e, out):
            return (e[f"{out}_sum"] / (e[f"{out}_n"] + n0)).reindex(gid).to_numpy()
        # home trails: home comeback - away hold; home leads: -(away comeback - home hold)
        return np.where(home_trails, sk(eh, "c") - sk(ea, "h"), -(sk(ea, "c") - sk(eh, "h")))

    y, wt = S.y.to_numpy(), S.w.to_numpy()
    best = None
    for n0 in N0_GRID:
        x = x_of(n0)
        c = 0.0
        for _ in range(25):
            zz = z[tr] + c * x[tr]
            pp = np.clip(F.Phi(zz), 1e-9, 1 - 1e-9)
            f = F.phi(zz)
            gr = (wt[tr] * x[tr] * f * (y[tr] - pp) / (pp * (1 - pp))).sum()
            hh = (wt[tr] * x[tr] ** 2 * f * f / (pp * (1 - pp))).sum()
            c += gr / hh
        llv = np.average(F.ll(F.Phi(z[tr] + c * x[tr]), y[tr]), weights=wt[tr])
        if best is None or llv < best[2]:
            best = (n0, c, llv, x)
    n0, c, _, x = best
    w(f"  fitted on train: n0 {n0}, c {c:+.3f} (on the z scale per unit of win-probability residual)")
    St = S[te]
    d = F.diff_txt(St, F.Phi(z[te] + c * x[te]), F.Phi(z[te]))
    w(f"  test (2022-23 .. 2025-26, {St.game_id.nunique()} games): log loss with − without {d}")
    for part in ("Q1", "Q2", "Q3", "Q4 to 2:00", "last 2:00", "OT"):
        m = (St.part == part).to_numpy()
        if m.any():
            w(f"      {part:<11} {F.diff_txt(St[m], F.Phi(z[te][m] + c * x[te][m]), F.Phi(z[te][m]))}")
    w("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
