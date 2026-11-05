#!/usr/bin/env python3
"""Box-score projector with last season as a prior, and with fitted weights.

  python research/projector_prior_fit.py   # -> diagnostics/projector_backtest/RESULTS_prior_fit.txt

Owner's request (2026-11-05), after research/projector_backtest.py: add last
season as a prior and fit the weights. Arms, grid and readings fixed before
the run.

Inputs per team before each game, from the projector's own team-game rows
(research/projector_backtest.prepare_team_game_rows on the BBR logs):
offensive rating, defensive rating and pace, each as the season-to-date
mean, the home-or-away split mean and the last-10 mean over games strictly
before the game (the projector's three windows; a window with no games
drops out and the others are renormalised, as weighted_average does).

Prior (P1, P2): last season's full-season team mean of each metric,
regressed toward that season's league mean, prior = lg + 0.6 (team - lg)
(as the totals model's prior), worth K games:
  metric = (n * blend + K * prior) / (n + K), n = games played this season.
A team without a last season keeps the projector's blend.

Arms:
  P0  as supplied: blend 0.50 / 0.40 / 0.10, offence share 0.55, no prior,
      both teams with >= 5 prior games (reproduces projector_backtest)
  P1  P0 + prior (K = 10), every game from game 1
  P2  prior + fitted: blend weights on the 0.1 simplex grid, offence share
      w in {0.45, 0.50, 0.55, 0.60, 0.65}, K in {0, 5, 10, 20}; then
      margin' = a + b * margin and total' = c + d * total by least squares;
      everything fitted on the seasons before the scored one (walk-forward,
      2017-18 on), minimising margin MSE + total MSE on games 1+.
Projection (all arms): home = (w ORtg_h + (1 - w) DRtg_a) / 100 x pace,
away = ((1 - w) ORtg_a + w DRtg_h) / 100 x pace, pace = mean of the two
teams' paces (w = 0.55 is the supplied split).

Readings, each book x season alone (ESPN BET 2024-25, DraftKings 2025-26,
ESPN BET 2025-26; DraftKings 2026-27 information only), on the games P0
predicts (both teams >= 5 prior games) and, for P1 / P2, also games 1-4:
margin and total RMSE, winner accuracy and bias beside the production
models and the closing lines; picks against the closing spread and total
(edge's sign, |edge| >= 3, >= 6) at the closing prices against the no-vig
null; P2's fitted parameters per season. Research only: nothing here
touches the ledgers, pages or model/*.json.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
import projector_backtest as pb                   # noqa: E402

OUT = pb.OUT
METRICS = ["off_rating", "def_rating", "pace"]
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
SCORED = [2025, 2026, 2027]
REGRESS, K_P1 = 0.6, 10.0
BLENDS = [(a / 10, b / 10, round(1 - a / 10 - b / 10, 1))
          for a in range(11) for b in range(11 - a)]
OFF_SHARES = (0.45, 0.50, 0.55, 0.60, 0.65)
KS = (0.0, 5.0, 10.0, 20.0)


def windows(y):
    """Per game (home, away): the three window means of each metric, the
    windows' availability, games played, the prior and the result."""
    tg = pb.prepare_team_game_rows(pb.league_game_log_from_bbr(y))
    pairs = pb.build_game_pairs(tg)
    by_team = {t: g.sort_values(["game_date", "game_id"]).reset_index(drop=True)
               for t, g in tg.groupby("team_abbr")}
    pos = {}
    for t, g in by_team.items():
        for i, gid in enumerate(g["game_id"]):
            pos[(gid, t)] = i
    rows = []
    for _, gm in pairs.iterrows():
        rec = dict(game_id=gm["game_id"], game_date=pd.Timestamp(gm["game_date"]).strftime("%Y-%m-%d"),
                   home_abbr=gm["home_abbr"], away_abbr=gm["away_abbr"],
                   actual_margin=gm["actual_home_score"] - gm["actual_away_score"],
                   actual_total=gm["actual_home_score"] + gm["actual_away_score"])
        for side, t, is_home in (("h", gm["home_abbr"], True), ("a", gm["away_abbr"], False)):
            g = by_team[t]
            i = pos[(gm["game_id"], t)]
            prev = g.iloc[:i]
            rec[f"n_{side}"] = i
            split = prev[prev["is_home"] == is_home]
            last = prev.tail(10)
            for m in METRICS:
                rec[f"{m}_s_{side}"] = prev[m].mean() if i else np.nan
                rec[f"{m}_p_{side}"] = split[m].mean() if len(split) else np.nan
                rec[f"{m}_l_{side}"] = last[m].mean() if i else np.nan
        rows.append(rec)
    D = pd.DataFrame(rows)
    # season-end team means and league means: the next season's prior
    end = tg.groupby("team_abbr")[METRICS].mean()
    lg = tg[METRICS].mean()
    return D, end, lg


def blend(D, side, m, wts):
    vals = [D[f"{m}_{k}_{side}"].to_numpy(float) for k in ("s", "p", "l")]
    num = np.zeros(len(D))
    den = np.zeros(len(D))
    for w, v in zip(wts, vals):
        ok = np.isfinite(v)
        num += np.where(ok, w * np.nan_to_num(v), 0.0)
        den += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def metric(D, side, m, wts, K):
    b = blend(D, side, m, wts)
    if K <= 0:
        return b
    n = D[f"n_{side}"].to_numpy(float)
    pr = D[f"{m}_prior_{side}"].to_numpy(float)
    has_b = np.isfinite(b)
    has_p = np.isfinite(pr)
    out = np.where(has_b & has_p, (n * np.nan_to_num(b) + K * np.nan_to_num(pr)) / (n + K),
                   np.where(has_b, b, pr))
    return out


def project(D, wts, w, K):
    o_h, d_h, p_h = (metric(D, "h", m, wts, K) for m in METRICS)
    o_a, d_a, p_a = (metric(D, "a", m, wts, K) for m in METRICS)
    pace = (np.where(np.isfinite(p_h), p_h, 100.0) + np.where(np.isfinite(p_a), p_a, 100.0)) / 2
    home = (w * np.nan_to_num(o_h) + (1 - w) * np.nan_to_num(d_a)) / 100 * pace
    away = ((1 - w) * np.nan_to_num(o_a) + w * np.nan_to_num(d_h)) / 100 * pace
    ok = np.isfinite(o_h) & np.isfinite(d_a) & np.isfinite(o_a) & np.isfinite(d_h)
    return np.where(ok, home - away, np.nan), np.where(ok, home + away, np.nan)


def load_all():
    data, ends = {}, {}
    for y in SEASONS:
        D, end, lg = windows(y)
        data[y] = D
        ends[y] = (end, lg)
    for y in SEASONS:
        prev = ends.get(y - 1)
        for side, col in (("h", "home_abbr"), ("a", "away_abbr")):
            for m in METRICS:
                if prev is None:
                    data[y][f"{m}_prior_{side}"] = np.nan
                    continue
                end, lg = prev
                team = data[y][col].map(end[m])
                data[y][f"{m}_prior_{side}"] = lg[m] + REGRESS * (team - lg[m])
    return data


def fit(train, margin_intercept=True, fixed_hca=None):
    """Grid over blend, offence share and K; then the linear calibration
    (through the origin for the margin when margin_intercept is False: no
    added home-court constant)."""
    best = None
    for wts, w, K in itertools.product(BLENDS, OFF_SHARES, KS):
        ms, ts, am, at = [], [], [], []
        for D in train:
            mg, tt = project(D, wts, w, K)
            ok = np.isfinite(mg) & (np.fmin(D["n_h"], D["n_a"]) >= 1)
            ms.append(mg[ok]); ts.append(tt[ok])
            am.append(D["actual_margin"].to_numpy()[ok]); at.append(D["actual_total"].to_numpy()[ok])
        mg, tt, am, at = map(np.concatenate, (ms, ts, am, at))
        if fixed_hca is not None:              # a fixed home-court constant
            bm = np.array([float(mg @ (am - fixed_hca) / (mg @ mg)), float(fixed_hca)])
        elif margin_intercept:
            bm = np.polyfit(mg, am, 1)
        else:
            bm = np.array([float(mg @ am / (mg @ mg)), 0.0])
        bt = np.polyfit(tt, at, 1)
        loss = np.mean((am - np.polyval(bm, mg)) ** 2) + np.mean((at - np.polyval(bt, tt)) ** 2)
        if best is None or loss < best[0]:
            best = (loss, wts, w, K, bm, bt)
    return best


def dec(a):
    return pb.dec(a)


def bets(edge, res, ph, pa):
    out = []
    for thr in (0, 3, 6):
        sel = (np.abs(edge) >= thr) & (res != 0) & np.isfinite(ph) & np.isfinite(pa)
        if sel.sum() < 2:
            out.append(f"|edge|>={thr}: n {int(sel.sum())}")
            continue
        side = edge[sel] > 0
        win = np.where(side, res[sel] > 0, res[sel] < 0)
        dh, da = dec(ph[sel]), dec(pa[sel])
        qh = (1 / dh) / (1 / dh + 1 / da)
        pay = np.where(side, dh, da)
        q = np.where(side, qh, 1 - qh)
        roi = np.where(win, pay - 1, -1.0)
        out.append(f"|edge|>={thr}: n {int(sel.sum()):4d} won {win.mean() * 100:5.1f}% ROI "
                   f"{roi.mean() * 100:+6.1f}% ± {1.96 * roi.std(ddof=1) / np.sqrt(len(roi)) * 100:4.1f} "
                   f"null {(q * pay - 1).mean() * 100:+.1f}%")
    return out


def rmse(x):
    return float(np.sqrt(np.nanmean(np.square(x))))


def report_block(G, label):
    L = [f"== {label}  games {len(G)}"]
    am, at = G["actual_margin"], G["actual_total"]
    for nm, mg, tt in (("P0 supplied", G["m_P0"], G["t_P0"]), ("P1 + prior", G["m_P1"], G["t_P1"]),
                       ("P2 + prior, fitted", G["m_P2"], G["t_P2"])):
        L.append(f"  {nm:20s} margin RMSE {rmse(mg - am):6.2f} winner {((mg > 0) == (am > 0)).mean():.3f} "
                 f"bias {(mg - am).mean():+.2f} SD {mg.std():.2f} | total RMSE {rmse(tt - at):6.2f} "
                 f"bias {(tt - at).mean():+.2f} SD {tt.std():.2f}")
    for nm, mg in (("production v11", G["prod_margin"]), ("closing spread", -G["close_spread"])):
        if mg.notna().any():
            L.append(f"  {nm:20s} margin RMSE {rmse(mg - am):6.2f} winner {((mg > 0) == (am > 0)).mean():.3f} "
                     f"bias {(mg - am).mean():+.2f} SD {mg.std():.2f}")
    for nm, tt in (("T* production totals", G["prod_total"]), ("closing total", G["close_total"])):
        if tt.notna().any():
            L.append(f"  {nm:20s} total RMSE {rmse(tt - at):6.2f} bias {(tt - at).mean():+.2f} SD {tt.std():.2f}")
    s = G.dropna(subset=["close_spread"])
    t = G.dropna(subset=["close_total"])
    for arm in ("P0", "P1", "P2"):
        if len(s):
            e = (s[f"m_{arm}"] + s["close_spread"]).to_numpy()
            r = (s["actual_margin"] + s["close_spread"]).to_numpy()
            L.append(f"  {arm} ATS  " + "  ".join(bets(e, r, s["close_home_spread_odds"].to_numpy(float),
                                                       s["close_away_spread_odds"].to_numpy(float))))
        if len(t):
            e = (t[f"t_{arm}"] - t["close_total"]).to_numpy()
            r = (t["actual_total"] - t["close_total"]).to_numpy()
            L.append(f"  {arm} O/U  " + "  ".join(bets(e, r, t["close_over"].to_numpy(float),
                                                       t["close_under"].to_numpy(float))))
    return L


def posthoc_no_hca():
    """Owner's request after the run (2026-11-05): P2 without the added
    home-court constant (margin calibrated through the origin), beside P2,
    walk-forward on the same games. Post-hoc; descriptive."""
    data = load_all()
    L = ["POST-HOC (requested after the run): P2 without the added home-court constant (P2b:",
         "margin' = b x margin, refitted through the origin with its own grid choice) and with a",
         "fixed +1 point (P2c: margin' = 1 + b x margin), beside P2.",
         "Same games: both teams with >= 1 prior game; walk-forward; ROI ± 95%.", ""]
    for y in SCORED:
        D = data[y].copy()
        out = {}
        for nm, icpt, hca in (("P2", True, None), ("P2b", False, None), ("P2c", True, 1.0)):
            loss, wts, w, K, bm, bt = fit([data[t] for t in SEASONS if 2017 <= t < y], icpt, hca)
            mg, tt = project(D, wts, w, K)
            D[f"m_{nm}"], D[f"t_{nm}"] = np.polyval(bm, mg), np.polyval(bt, tt)
            out[nm] = f"blend {wts}, offence {w}, K {K}, margin' = {bm[1]:+.2f} + {bm[0]:.3f} x margin"
        D = D[np.fmin(D["n_h"], D["n_a"]) >= 1]
        D = D.merge(pb.odds_and_production(y), on=["game_date", "home_abbr", "away_abbr"], how="left")
        D = D.drop_duplicates(["game_date", "home_abbr", "away_abbr"])
        L.append(f"== {y - 1}-{str(y)[2:]}" + (" (information only)" if y == 2027 else "") +
                 f"  games {len(D)}")
        for nm in ("P2", "P2b", "P2c"):
            L.append(f"  {nm:4s} fitted: {out[nm]}")
        am = D["actual_margin"]
        for nm, mg in (("P2", D["m_P2"]), ("P2b", D["m_P2b"]), ("P2c", D["m_P2c"]),
                       ("production v11", D["prod_margin"]),
                       ("closing spread", -D["close_spread"])):
            ok = mg.notna() & am.notna()
            L.append(f"  {nm:15s} n {ok.sum():4d} margin RMSE {rmse((mg - am)[ok]):6.2f} winner "
                     f"{((mg[ok] > 0) == (am[ok] > 0)).mean():.3f} bias {(mg - am)[ok].mean():+.2f} "
                     f"home share of projected wins {(mg[ok] > 0).mean():.3f} (actual {(am[ok] > 0).mean():.3f})")
        for lo, hi in ((-99, -3), (-3, 0), (0, 3), (3, 99)):
            row = []
            for nm in ("P2", "P2b", "P2c"):
                sel = (D[f"m_{nm}"] > lo) & (D[f"m_{nm}"] <= hi)
                row.append(f"{nm} n {sel.sum():4d} proj {D.loc[sel, f'm_{nm}'].mean():+6.2f} actual "
                           f"{am[sel].mean():+6.2f} home win {(am[sel] > 0).mean():.3f}")
            L.append(f"    projected ({lo:+d}, {hi:+d}]: " + " | ".join(row))
        for book in [b for b in D["close_book"].dropna().unique()]:
            B = D[D["close_book"] == book].dropna(subset=["close_spread"])
            r = (B["actual_margin"] + B["close_spread"]).to_numpy()
            for nm in ("P2", "P2b", "P2c"):
                e = (B[f"m_{nm}"] + B["close_spread"]).to_numpy()
                L.append(f"  {book} {nm:4s} ATS  " + "  ".join(bets(
                    e, r, B["close_home_spread_odds"].to_numpy(float),
                    B["close_away_spread_odds"].to_numpy(float))))
        L.append("")
    text = "\n".join(L) + "\n"
    (OUT / "RESULTS_prior_fit_no_hca.txt").write_text(text)
    print(text)


def main():
    if "--no-hca" in sys.argv:
        return posthoc_no_hca()
    data = load_all()
    L = ["Projector with last season as a prior (P1) and with fitted weights (P2), point-in-time,",
         "walk-forward fits on earlier seasons. Arms and readings fixed before the run. ROI ± 95%.", ""]
    for y in SCORED:
        D = data[y].copy()
        D["m_P0"], D["t_P0"] = project(D, (0.5, 0.4, 0.1), 0.55, 0.0)
        p0_off = np.fmin(D["n_h"], D["n_a"]) < 5        # the supplied 5-game warmup
        D.loc[p0_off, ["m_P0", "t_P0"]] = np.nan
        D["m_P1"], D["t_P1"] = project(D, (0.5, 0.4, 0.1), 0.55, K_P1)
        loss, wts, w, K, bm, bt = fit([data[t] for t in SEASONS if 2017 <= t < y])
        mg, tt = project(D, wts, w, K)
        D["m_P2"], D["t_P2"] = np.polyval(bm, mg), np.polyval(bt, tt)
        L.append(f"P2 fitted for {y - 1}-{str(y)[2:]} on 2016-17..{y - 2}-{str(y - 1)[2:]}: blend season/split/"
                 f"last10 {wts}, offence share {w}, prior K {K}, margin' = {bm[1]:+.2f} + {bm[0]:.3f} x margin, "
                 f"total' = {bt[1]:+.1f} + {bt[0]:.3f} x total")
        D = D.merge(pb.odds_and_production(y), on=["game_date", "home_abbr", "away_abbr"], how="left")
        D = D.merge(pb.tstar_totals(y), on=["game_date", "home_abbr", "away_abbr"], how="left")
        D.to_csv(OUT / f"predictions_prior_fit_{y}.csv", index=False, float_format="%.4g")
        warm = np.fmin(D["n_h"], D["n_a"])
        for book in [b for b in D["close_book"].dropna().unique()]:
            B = D[D["close_book"] == book]
            tag = "" if y < 2027 else " (information only)"
            L += report_block(B[np.fmin(B["n_h"], B["n_a"]) >= 5],
                              f"{y - 1}-{str(y)[2:]} {book}, both teams >= 5 games (P0's games){tag}")
            early = B[(np.fmin(B["n_h"], B["n_a"]) >= 1) & (np.fmin(B["n_h"], B["n_a"]) < 5)]
            if len(early) >= 20:
                L += report_block(early, f"{y - 1}-{str(y)[2:]} {book}, games 1-4 (P1 / P2 only; P0 abstains)")
        del warm
        L.append("")
    text = "\n".join(L) + "\n"
    (OUT / "RESULTS_prior_fit.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
