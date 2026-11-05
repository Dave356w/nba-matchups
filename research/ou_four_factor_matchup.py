#!/usr/bin/env python3
"""Four-factor matchups against the market O/U line: walk-forward test.

  python research/ou_four_factor_matchup.py   # writes diagnostics/ou_market_residual/RESULTS_four_factor.txt

Owner's request (2026-11-05): fit the four-factor matchup between the two
teams against the market's total to signal an O/U pick. Research only: the
target is the final total against the market line, which G9 excludes for
the production model. Arms, prices and the rule were committed before the
run.

Inputs per team before each game (games strictly before the slate; the
production rating's window: last season's totals x 0.25 fading out by game
10, half-life 25, ORB%), from the cached BBR logs:
  offence  eFG%, TOV per 100, ORB%, FTA/FGA
  defence  the same four allowed to opponents
  pace     possessions per game (decayed the same way)
Matchup features for a game (league-centred: each minus the league mean of
the same feature over that slate's teams):
  efg  = (eFG_h + eFG_allowed_a) + (eFG_a + eFG_allowed_h)
  tov  = (TOV_h + TOV_forced_a) + (TOV_a + TOV_forced_h)
  orb  = (ORB_h + ORB_allowed_a) + (ORB_a + ORB_allowed_h)
  ftr  = (FTr_h + FTr_allowed_a) + (FTr_a + FTr_allowed_h)
  pace = pace_h + pace_a
  x_k  = the products off_h,k x allowed_a,k + off_a,k x allowed_h,k of the
         centred factors (a strong offence meeting a weak defence)

Target: final total minus the line, at the open and at the close
separately. Arms (fixed before the run), least squares or ridge refitted
before every slate on the priced reconstructed games before it (expanding;
ESPN BET 2023-24 on; native rows never in a fit; scoring after 500 games),
features standardised on the training rows:
  F0  intercept only
  F1  intercept + efg, tov, orb, ftr, pace                 (OLS)
  F2  F1 + the four products x_k                           (ridge, lambda 50)
  F3  F1 without an intercept                              (OLS)
Scored: ESPN BET 2024-25 and DraftKings 2025-26; information only: ESPN BET
2025-26 and native DraftKings 2026-27. Baseline: the line itself.

Readings per book x season x price: squared error of line + adjustment
minus the line's, paired, ± 95%; the side the adjustment favours when
|adjustment| >= 0.5 (and >= 1.0) at that price's over/under prices: n, win
rate, ROI ± 95%, null; overs share; the fitted standardised coefficients
at each season's end.

Decision rule, set before the run (4 arms x 2 prices = 8 tests, z = 2.73
Bonferroni): an arm is a candidate at a price only if
  1. squared error beats the line in both scored seasons, with the
     combined interval at z = 2.73 excluding 0;
  2. its |adjustment| >= 0.5 side has ROI above its null in both scored
     seasons.
Passing makes it a candidate for the owner's decision and an amendment
(G9), not a change. Research only: nothing here touches the ledgers, pages
or model/*.json.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import nba_composite as nc                        # noqa: E402
import totals                                     # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "ou_market_residual")
SIG = os.path.join(ROOT, "diagnostics", "totals_signal")
MIN_TRAIN, Z_MULTI, LAMBDA = 500, 2.73, 50.0
SCORED = [(2025, "espnbet"), (2026, "dk")]
INFO = [(2026, "espnbet"), (2027, "dk")]
BASE = ["efg", "tov", "orb", "ftr", "pace"]
PROD = ["x_efg", "x_tov", "x_orb", "x_ftr"]
ARMS = {"F0": ([], True, 0.0), "F1": (BASE, True, 0.0), "F2": (BASE + PROD, True, LAMBDA),
        "F3": (BASE, False, 0.0)}
PRICES = ("open", "close")


# --------------------------------------------------------------- data --
def lines():
    h = pd.read_csv(os.path.join(SIG, "totals_history_2024.csv")).assign(basis="reconstructed")
    o = pd.read_csv(os.path.join(SIG, "totals_odds.csv"))
    ids = pd.concat([pd.read_csv(os.path.join(ROOT, "data", f))[
        ["game_id", "season", "slate_date", "home", "away", "home_pts", "away_pts"]]
        for f in ("nba_reconstructed.csv", "nba_ledger.csv")]).drop_duplicates("game_id")
    o = o.merge(ids, on="game_id")
    keep = ["game_id", "basis", "season", "slate_date", "home", "away", "home_pts", "away_pts",
            "book"] + [f"{p}_{k}" for p in PRICES for k in ("total", "over", "under")]
    L = pd.concat([h[keep], o[keep]], ignore_index=True)
    L = L.dropna(subset=["home_pts", "away_pts"])
    L["total"] = L["home_pts"] + L["away_pts"]
    return L.sort_values(["season", "slate_date", "game_id"]).reset_index(drop=True)


def team_inputs(seasons):
    """{(season, date, team): (off4, def4, pace)} before each of the team's games."""
    out = {}
    for y in seasons:
        logs = nc.load_logs(y)
        try:
            prior = nc.load_logs(y - 1)
        except Exception:                          # noqa: BLE001
            prior = {}
        for tm, lg in logs.items():
            lg = lg.sort_values("date").reset_index(drop=True)
            pr = prior.get(tm)
            poss = totals.possessions(lg)
            ppr = totals.possessions(pr) if pr is not None and len(pr) else np.array([])
            for i in range(len(lg)):
                rho = nc.fade_rho(i)
                t = nc.fade_totals(lg, i, pr, rho)
                if t is None:
                    continue
                f = nc.features_from_totals(np.array([t[c] for c in nc.COLS], float), orb_pct=True)
                # signed features -> raw: off (eFG, TOV, ORB%, FTr), allowed (same four)
                off = np.array([f[0], -f[1], f[2], f[3]])
                allowed = np.array([-f[4], f[5], -f[6], -f[7]])
                parts, w = [], []
                if rho > 0 and len(ppr):
                    parts.append(ppr)
                    w.append(np.full(len(ppr), rho))
                if i:
                    parts.append(poss[:i])
                    w.append(np.ones(i))
                v = np.concatenate(parts)
                ww = 0.5 ** (np.arange(len(v))[::-1] / nc.HALF_LIFE) * np.concatenate(w)
                pace = float((ww * v).sum() / ww.sum())
                out[(y, pd.Timestamp(lg.loc[i, "date"]).strftime("%Y-%m-%d"), tm)] = (off, allowed, pace)
    return out


def features(L):
    T = team_inputs(sorted(L["season"].unique()))
    rows = []
    for r in L.itertuples(index=False):
        h, a = T.get((r.season, r.slate_date, r.home)), T.get((r.season, r.slate_date, r.away))
        rows.append((h, a))
    keep = [i for i, (h, a) in enumerate(rows) if h is not None and a is not None]
    L = L.iloc[keep].reset_index(drop=True)
    rows = [rows[i] for i in keep]
    oh = np.array([h[0] for h, _ in rows])
    ah = np.array([h[1] for h, _ in rows])
    oa = np.array([a[0] for _, a in rows])
    aa = np.array([a[1] for _, a in rows])
    ph = np.array([h[2] for h, _ in rows])
    pa = np.array([a[2] for _, a in rows])
    # league-centre each team input by the slate's mean over its teams
    both_off = np.vstack([oh, oa])
    both_all = np.vstack([ah, aa])
    dates2 = np.concatenate([L["slate_date"].values] * 2)
    mo = pd.DataFrame(both_off).groupby(dates2).transform("mean").to_numpy()[:len(L)]
    ma = pd.DataFrame(both_all).groupby(dates2).transform("mean").to_numpy()[:len(L)]
    oh, oa, ah, aa = oh - mo, oa - mo, ah - ma, aa - ma
    for k, nm in enumerate(["efg", "tov", "orb", "ftr"]):
        L[nm] = (oh[:, k] + aa[:, k]) + (oa[:, k] + ah[:, k])
        L[f"x_{nm}"] = oh[:, k] * aa[:, k] + oa[:, k] * ah[:, k]
    mp = pd.Series(np.concatenate([ph, pa])).groupby(dates2).transform("mean").to_numpy()[:len(L)]
    L["pace"] = (ph - mp) + (pa - mp)
    return L


# ---------------------------------------------------------------- fit --
def walk(L):
    L = L.copy()
    rec = (L["basis"] == "reconstructed").to_numpy()
    dates = L["slate_date"].to_numpy()
    coefs = []
    for p in PRICES:
        L[f"res_{p}"] = L["total"] - L[f"{p}_total"]
        for arm in ARMS:
            L[f"adj_{arm}_{p}"] = np.nan
    for d, idx in L.groupby("slate_date").groups.items():
        for p in PRICES:
            tr = rec & (dates < d) & L[f"res_{p}"].notna().to_numpy()
            if tr.sum() < MIN_TRAIN:
                continue
            y = L.loc[tr, f"res_{p}"].to_numpy(float)
            row = dict(slate_date=d, season=int(L.at[idx[0], "season"]), price=p)
            for arm, (feats, icpt, lam) in ARMS.items():
                if not feats:
                    a = float(y.mean())
                    L.loc[idx, f"adj_{arm}_{p}"] = a
                    continue
                X = L.loc[tr, feats].to_numpy(float)
                mu, sd = X.mean(0), X.std(0)
                sd[sd == 0] = 1.0
                Z = (X - mu) / sd
                Zc = (L.loc[idx, feats].to_numpy(float) - mu) / sd
                if icpt:
                    A = np.column_stack([np.ones(len(Z)), Z])
                    pen = np.diag([0.0] + [lam] * Z.shape[1])
                    b = np.linalg.solve(A.T @ A + pen, A.T @ y)
                    adj = b[0] + Zc @ b[1:]
                else:
                    b = np.linalg.lstsq(Z, y, rcond=None)[0]
                    adj = Zc @ b
                L.loc[idx, f"adj_{arm}_{p}"] = adj
                row.update({f"{arm}_b{k}": v for k, v in enumerate(b)})
            coefs.append(row)
    return L, pd.DataFrame(coefs)


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def readings(g, arm, p, edge):
    col, res = f"adj_{arm}_{p}", f"res_{p}"
    g = g[g[col].notna() & g[res].notna()]
    d = (g[res] - g[col]) ** 2 - g[res] ** 2
    out = dict(mse=d.mean(), var=d.var(ddof=1) / max(len(d), 1), n_all=len(d))
    x = g[(g[col].abs() >= edge) & (g[res] != 0)].dropna(subset=[f"{p}_over", f"{p}_under"])
    out["n"] = len(x)
    if len(x) > 1:
        over = (x[col] > 0).to_numpy()
        y = (x[res] > 0).to_numpy()
        do, du = dec(x[f"{p}_over"]), dec(x[f"{p}_under"])
        qo = (1 / do) / (1 / do + 1 / du)
        pay, q = np.where(over, do, du), np.where(over, qo, 1 - qo)
        win = np.where(over, y, ~y)
        roi = np.where(win, pay - 1, -1.0)
        out.update(win=win.mean(), roi=roi.mean(), roi_h=1.96 * roi.std(ddof=1) / np.sqrt(len(roi)),
                   null=(q * pay - 1).mean(), overs=over.mean())
    return out


def report(L, C):
    out = ["Four-factor matchups against the market O/U line (2026-11-05). Research only (G9:",
           "the target is the total against the line). Arms, prices and rule fixed before the",
           "run. Baseline: the line. Squared error vs line in pts^2 (negative = better); ROI ± 95%.",
           ""]
    R = {}
    for (Y, book) in SCORED + INFO:
        g = L[(L["season"] == Y) & (L["book"] == book)]
        tag = "scored" if (Y, book) in SCORED else "information only"
        out.append(f"== {book} {Y - 1}-{str(Y)[2:]} ({tag})  n {len(g)}")
        for p in PRICES:
            out.append(f"  @{p}  mean(total - line) {g[f'res_{p}'].mean():+.2f}")
            for arm in ARMS:
                for edge in (0.5, 1.0):
                    r = readings(g, arm, p, edge)
                    if edge == 0.5:
                        R[(Y, book, arm, p)] = r
                    side = (f"n {r['n']:4d} won {r['win'] * 100:5.1f}% ROI {r['roi'] * 100:+6.1f}% ± "
                            f"{r['roi_h'] * 100:5.1f} null {r['null'] * 100:+.1f}% overs "
                            f"{r['overs'] * 100:3.0f}%") if "roi" in r else f"n {r['n']:4d}"
                    lead = (f"    {arm} vs line {r['mse']:+6.2f} ± {1.96 * np.sqrt(r['var']):5.2f}"
                            if edge == 0.5 else " " * 33)
                    out.append(f"{lead}  |adj|>={edge}: {side}")
        out.append("")
    out.append("== Standardised coefficients at each season's last slate (intercept first where fitted)")
    for (Y, p), c in C.groupby(["season", "price"]):
        last = c.iloc[-1]
        cells = []
        for arm, (feats, icpt, _) in ARMS.items():
            k = len(feats) + (1 if icpt else 0)
            if not feats:
                continue
            cells.append(f"{arm} [" + ", ".join(f"{last.get(f'{arm}_b{j}', np.nan):+.2f}"
                                                 for j in range(k)) + "]")
        out.append(f"  {Y} @{p}: " + "  ".join(cells))
    out += ["", f"== Decision rule (set before the run; z = {Z_MULTI} for 8 tests)"]
    for p in PRICES:
        for arm in list(ARMS)[1:]:
            s = [R[(Y, b, arm, p)] for Y, b in SCORED]
            w = [1 / x["var"] for x in s]
            comb = sum(x["mse"] * wi for x, wi in zip(s, w)) / sum(w)
            half = Z_MULTI * np.sqrt(1 / sum(w))
            r1 = all(x["mse"] < 0 for x in s) and comb + half < 0
            r2 = all("roi" in x and x["roi"] > x["null"] for x in s)
            out.append(f"  {arm} @{p:5s} squared error {s[0]['mse']:+.2f} / {s[1]['mse']:+.2f}, combined "
                       f"{comb:+.2f} ± {half:.2f}: {'pass' if r1 else 'fail'}; ROI vs null "
                       f"{'pass' if r2 else 'fail'} -> {'CANDIDATE' if r1 and r2 else 'not a candidate'}")
    return out


def main():
    L, C = walk(features(lines()))
    os.makedirs(OUT, exist_ok=True)
    text = "\n".join(report(L, C)) + "\n"
    open(os.path.join(OUT, "RESULTS_four_factor.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
