#!/usr/bin/env python3
"""Opponent-adjusted four factors: walk-forward test on the margin model's base route.

  python research/opponent_adjusted.py --run [--seasons 2017 ... 2027]
  python research/opponent_adjusted.py --report

Owner's request (2026-11-05): every model feature should account for the
teams faced. The four-factor rating reads a team's own and its opponents'
raw totals, so a team that has met strong defences looks worse on offence
than it is. An additive strength-of-schedule term was ruled out on games
10+ (research/team_quality.py, -0.0000 / -0.0002); a per-factor adjustment,
and games 1-9, where schedules are least balanced, were never tested.

Adjustment, per factor k (eFG%, turnovers, ORB%, FTA/FGA, signed so higher
is better, as features_from_totals). Each team-game row gives the team's
offence in that game, u = mu + o_team - dq_opp + e. Before each slate date
the league-wide (mu, o, dq) are fitted by weighted ridge on this season's
games before the date (weights 0.5 ** (days ago / 51), about 25 games;
ridge LAMBDA game weights on o and dq). A team's correction is the decayed
mean (its own half-life 25 over its games, as the rating window) of
  offence  +dq of the opponents it faced (a hard schedule hid offence),
  defence  +o  of the opponents it faced (good offences hid defence),
using the current estimates of those opponents. The rating is v10's faded
rating (last season x rho fading out by game 10): last season's full
schedule is balanced, so only this season's share s of the window's
weight is corrected: f_adj = f + s * c. delta_adj = composite(f_adj home -
f_adj away) under each fold's weights; d_phase follows delta.

Arms, all fixed before the run. Every fold Y refits on the game seasons
before Y (research/dynamic_fit.py's S protocol, the v10h25 feature build of
research/v10_candidate.py: ORB% composite, drafted rookies by pick,
half-life 25), OLS on the final home margin, P = Phi(m / sigma):
  B    the base route, nba_composite.V6_FEATURES
  A5   delta and d_phase from the adjusted rating, LAMBDA = 5
  A20  the same with LAMBDA = 20 (more shrinkage of the opponent estimates)
  S5   B plus sos5 = delta_adj(5) - delta as its own fitted term

Limitation, stated before the run: base route only (the injury-report
route needs the report archive, not in this container); a change would
need its own report-route check.

Readings, each arm minus B on identical games (negative = better), ± 95%,
game-level, paired: log loss, Brier and margin MSE against outcomes, by
stage (game 0, 1-9, 10+) and fold; dispersion on games 1-9 (actual margin
on arm margin); against the close for each basis x book x season x stage
(reconstructed rows and this season's rebuilt rows, hindsight). Fold 2027
is information only.

Decision rule, set before the run, folds 2017-2026 (9 folds):
  1. games 1+ log loss arm - B below 0 with its 95% interval excluding 0,
     Brier agreeing in sign, better in at least 6 of the 9 folds;
  2. games 1-9 and games 10+ each not above 0 (point estimates);
  3. reconstructed DraftKings 2025-26 and ESPN BET 2024-25, games 10+:
     arm - B not above 0, Brier agreeing;
  4. no fold better than -0.008 (the leakage tripwire).
Passing makes an arm a candidate for the owner's decision and the usual
amendment, not a change. Research only: nothing here touches the ledgers,
the pages or model/*.json.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
import nba_composite as nc                        # noqa: E402
import dynamic_fit as dyn                         # noqa: E402
import orb_pct                                    # noqa: E402
import v10_candidate as v10c                      # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "opponent_adjusted")
CFG = v10c.CONFIGS["v10h25"]
FEATS = list(nc.V6_FEATURES)
LAMBDAS = (5.0, 20.0)
DAY_HALF_LIFE = 51.0
ARMS = ["B", "A5", "A20", "S5"]
RULE_FOLDS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
_PHI = np.vectorize(NormalDist().cdf)
Z95 = 1.96


# ------------------------------------------------------- adjustments --
def game_u(row):
    """A team-game's signed offence vector (4) from its 16 totals."""
    return nc.features_from_totals(np.array([row[c] for c in nc.COLS], float),
                                   orb_pct=True)[:4]


def season_corrections(y):
    """{(date, team): (s, {lam: c (8,)}), f (8,)} for every team before each
    of its games in season y: this season's share of the faded window, the
    correction per LAMBDA and the raw faded features."""
    logs = nc.load_logs(y)
    try:                     # the prior season as fade_v8_games reads it (2020 included)
        prior = nc.load_logs(y - 1)
    except Exception:        # noqa: BLE001
        prior = {}
    teams = sorted(logs)
    ix = {t: k for k, t in enumerate(teams)}
    rows = []
    for t, lg in logs.items():
        for i in range(len(lg)):
            r = lg.iloc[i]
            if r["opp"] in ix:
                rows.append((pd.Timestamp(r["date"]), t, r["opp"], *game_u(r)))
    obs = pd.DataFrame(rows, columns=["date", "team", "opp", "u0", "u1", "u2", "u3"])
    U = obs[["u0", "u1", "u2", "u3"]].to_numpy(float)
    n = len(teams)
    X = np.zeros((len(obs), 1 + 2 * n))
    X[:, 0] = 1.0
    X[np.arange(len(obs)), 1 + obs["team"].map(ix).to_numpy()] = 1.0
    X[np.arange(len(obs)), 1 + n + obs["opp"].map(ix).to_numpy()] = -1.0
    dates = sorted(obs["date"].unique())
    est = {}                                    # (date, lam) -> (o (n,4), dq (n,4))
    for d in dates:
        past = (obs["date"] < d).to_numpy()
        for lam in LAMBDAS:
            if not past.any():
                est[(d, lam)] = (np.zeros((n, 4)), np.zeros((n, 4)))
                continue
            age = (d - obs.loc[past, "date"]).dt.days.to_numpy(float)
            w = 0.5 ** (age / DAY_HALF_LIFE)
            Xp = X[past]
            A = Xp.T @ (Xp * w[:, None])
            A[1:, 1:] += lam * np.eye(2 * n)
            b = np.linalg.solve(A, Xp.T @ (U[past] * w[:, None]))
            est[(d, lam)] = (b[1:1 + n], b[1 + n:])
    out = {}
    for t, lg in logs.items():
        lg = lg.reset_index(drop=True)
        pr = prior.get(t)
        for i in range(len(lg)):
            d = pd.Timestamp(lg.loc[i, "date"])
            tot = nc.fade_totals(lg, i, pr, nc.fade_rho(i))
            if tot is None:
                continue
            f = nc.features_from_totals(np.array([tot[c] for c in nc.COLS], float), orb_pct=True)
            # this season's share of the window's weight (fade_totals' weights)
            np_ = len(pr) if pr is not None and nc.fade_rho(i) > 0 else 0
            ages = np.arange(np_ + i)[::-1]
            wts = 0.5 ** (ages / nc.HALF_LIFE)
            wts[:np_] *= nc.fade_rho(i)
            s = float(wts[np_:].sum() / wts.sum()) if wts.sum() > 0 else 0.0
            cs = {}
            if i == 0:
                cs = {lam: np.zeros(8) for lam in LAMBDAS}
            else:
                opps = [ix.get(o) for o in lg.loc[:i - 1, "opp"]]
                gw = 0.5 ** (np.arange(i)[::-1] / nc.HALF_LIFE)
                keep = np.array([o is not None for o in opps])
                oi = np.array([o for o in opps if o is not None], int)
                gw = gw[keep]
                for lam in LAMBDAS:
                    o_est, dq_est = est[(d, lam)]
                    c_off = (gw[:, None] * dq_est[oi]).sum(0) / gw.sum()
                    c_def = (gw[:, None] * o_est[oi]).sum(0) / gw.sum()
                    cs[lam] = np.concatenate([c_off, c_def])
            out[(d.strftime("%Y-%m-%d"), t)] = (s, cs, f)
    return out


_corr = {}


def corrections(y):
    if y not in _corr:
        _corr[y] = season_corrections(y)
        dyn.log(f"corrections {y}: {len(_corr[y])} team-games")
    return _corr[y]


def games(t, wts):
    """Season t's base-route games plus the adjusted deltas under wts."""
    g = v10c.games(t, wts, CFG)
    C = corrections(t)
    opening = nc.season_opening(nc.load_logs(t))
    raw, adj = [], {lam: [] for lam in LAMBDAS}
    for r in g.itertuples(index=False):
        ch, ca = C.get((r.slate_date, r.home)), C.get((r.slate_date, r.away))
        if ch is None or ca is None:
            raw.append(np.nan)
            for lam in LAMBDAS:
                adj[lam].append(np.nan)
            continue
        raw.append(nc.composite(ch[2] - ca[2], wts["sd"], wts["w"]))
        for lam in LAMBDAS:
            fh = ch[2] + ch[0] * ch[1][lam]
            fa = ca[2] + ca[0] * ca[1][lam]
            adj[lam].append(nc.composite(fh - fa, wts["sd"], wts["w"]))
    g["delta_check"] = raw
    g["phase"] = [nc.season_phase(d, opening) for d in g["date"]]
    for lam in LAMBDAS:
        k = int(lam)
        g[f"delta_{k}"] = adj[lam]
        g[f"d_phase_{k}"] = g[f"delta_{k}"] * g["phase"]
    g["sos5"] = g["delta_5"] - g["delta"]
    return g


def swap(df, k):
    """df with delta / d_phase replaced by the LAMBDA-k adjusted versions."""
    out = df.copy()
    out["delta"], out["d_phase"] = df[f"delta_{k}"], df[f"d_phase_{k}"]
    return out


def fit_predict(past, cur, feats):
    p = past[np.isfinite(past[feats + ["margin"]].to_numpy(float)).all(axis=1)]
    b, s = dyn.wls(p, feats=feats)
    return dyn.predict(b, s, cur, feats)


def fold(Y):
    wts = dyn.fold_weights(Y)
    G = {t: games(t, wts) for t in dyn.GAME_YEARS if t <= Y}
    past = pd.concat([G[t] for t in G if t < Y], ignore_index=True)
    cur = G[Y][np.isfinite(G[Y][dyn.CORE + ["delta_5", "delta_20"]].to_numpy(float))
               .all(axis=1)].copy()
    cur["m_B"], cur["p_B"] = fit_predict(past, cur, FEATS)
    for k in (5, 20):
        cur[f"m_A{k}"], cur[f"p_A{k}"] = fit_predict(swap(past, k), swap(cur, k), FEATS)
    cur["m_S5"], cur["p_S5"] = fit_predict(past, cur, FEATS + ["sos5"])
    mn = np.fmin(cur["gp_home"], cur["gp_away"])
    cur["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
    cur["season"] = Y
    gap = (cur["delta_check"] - cur["delta"]).abs().max()
    dyn.log(f"fold {Y}: {len(cur)} games; max |rebuilt delta - delta| {gap:.4f}")
    return cur[["season", "slate_date", "home", "away", "gp_home", "gp_away", "stage",
                "margin", "delta", "delta_5", "delta_20", "sos5"] +
               [c for a in ARMS for c in (f"m_{a}", f"p_{a}")]]


def run(seasons):
    orb_pct.use("pct" if CFG["orb"] else "per100")
    P = pd.concat([fold(Y) for Y in seasons], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    P.to_csv(os.path.join(OUT, f"predictions_{min(seasons)}-{max(seasons)}.csv"),
             index=False, float_format="%.6g")


# -------------------------------------------------------------- report --
def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ci(d):
    d = np.asarray(d, float)
    return d.mean(), Z95 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan


def paired(g, arm):
    y = g["y"].to_numpy()
    return (ll(g[f"p_{arm}"], y) - ll(g["p_B"], y),
            (g[f"p_{arm}"] - y) ** 2 - (g["p_B"] - y) ** 2,
            (g[f"m_{arm}"] - g["margin"]) ** 2 - (g["m_B"] - g["margin"]) ** 2)


def load():
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(
        os.path.join(OUT, "predictions_*.csv")))], ignore_index=True)
    P = P.drop_duplicates(["season", "slate_date", "home", "away"], keep="last")
    P["y"] = (P["margin"] > 0).astype(float)
    return P


def outcome_lines(P):
    L = ["== 1. Against outcomes, folds 2017-2026: arm - B (log loss / Brier / margin MSE, "
         "± 95%; negative = arm better)"]
    R = P[P["season"].isin(RULE_FOLDS)]
    for st, g in [("games 1+", R[R["stage"] != "game 0"])] + \
            [(s, R[R["stage"] == s]) for s in ("game 0", "games 1-9", "games 10+")]:
        L.append(f"  {st:10s} n {len(g):5d}  B log loss {ll(g['p_B'], g['y']).mean():.4f}")
        for arm in ARMS[1:]:
            d_ll, d_br, d_mse = paired(g, arm)
            wins = sum(paired(x, arm)[0].mean() < 0 for _, x in g.groupby("season"))
            m, h = ci(d_ll)
            mb, hb = ci(d_br)
            mm, hm = ci(d_mse)
            L.append(f"    {arm:4s} {m:+.4f} ± {h:.4f}  Brier {mb:+.4f} ± {hb:.4f}  "
                     f"MSE {mm:+.2f} ± {hm:.2f}  better in {wins} of {g['season'].nunique()}")
    L += ["", "  By fold, games 1+ / games 1-9 (log loss arm - B, ± 95%):"]
    for Y, g in P[P["stage"] != "game 0"].groupby("season"):
        e = g[g["stage"] == "games 1-9"]
        cells = []
        for arm in ARMS[1:]:
            m, h = ci(paired(g, arm)[0])
            me, he = ci(paired(e, arm)[0])
            cells.append(f"{arm} {m:+.4f} ± {h:.4f} / {me:+.4f} ± {he:.4f}")
        L.append(f"    {Y} n {len(g):4d}  " + "  ".join(cells))
    return L


def dispersion_lines(P):
    L = ["", "== 2. Dispersion on games 1-9, folds 2017-2026 (actual = a + b·arm margin, "
         "± 95%; b > 1 = compressed)"]
    g = P[P["season"].isin(RULE_FOLDS) & (P["stage"] == "games 1-9")]
    for arm in ARMS:
        X = np.column_stack([np.ones(len(g)), g[f"m_{arm}"].to_numpy(float)])
        y = g["margin"].to_numpy(float)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        se = np.sqrt(np.diag(r @ r / (len(y) - 2) * np.linalg.inv(X.T @ X)))
        L.append(f"  {arm:4s} SD margin {g[f'm_{arm}'].std():.2f}  slope {b[1]:.3f} ± "
                 f"{Z95 * se[1]:.3f}")
    d = P[P["stage"] != "game 0"]
    L.append(f"  adjustment size, games 1-9: SD(delta_5 - delta) "
             f"{(d.loc[d['stage'] == 'games 1-9', 'sos5']).std():.2f}, games 10+ "
             f"{(d.loc[d['stage'] == 'games 10+', 'sos5']).std():.2f} (win-% points; "
             f"SD delta {d['delta'].std():.2f})")
    return L


def market_lines(P):
    rec = pd.read_csv(os.path.join(ROOT, "data", "nba_reconstructed.csv")).assign(
        basis="reconstructed")
    reb = pd.read_csv(os.path.join(ROOT, "data", "nba_rebuilt.csv")).assign(basis="rebuilt")
    M = pd.concat([rec, reb], ignore_index=True)
    M = M[M["close_q_home"].notna()][["basis", "season", "slate_date", "home", "away",
                                      "close_book", "close_q_home"]]
    W = P.merge(M, on=["season", "slate_date", "home", "away"], how="inner")
    L = ["", "== 3. Against the close, each basis x book x season x stage alone "
         "(hindsight; arm - close log loss, ± 95%; [arm - B, Brier])"]
    gate = {}
    for (basis, book, Y, st), g in W.groupby(["basis", "close_book", "season", "stage"]):
        if st == "game 0" or len(g) < 30:
            continue
        y = g["y"].to_numpy()
        lc = ll(g["close_q_home"], y)
        m, h = ci(ll(g["p_B"], y) - lc)
        L.append(f"  {basis:13s} {book:7s} {Y} {st:10s} n {len(g):4d}  B {m:+.4f} ± {h:.4f}")
        for arm in ARMS[1:]:
            ma, ha = ci(ll(g[f"p_{arm}"], y) - lc)
            d_ll, d_br, _ = paired(g, arm)
            L.append(f"      {arm:4s} {ma:+.4f} ± {ha:.4f}  [{d_ll.mean():+.4f}, "
                     f"{d_br.mean():+.4f}]")
            gate[(basis, book, Y, st, arm)] = (d_ll.mean(), d_br.mean())
    return L, gate


def rule_lines(P, gate):
    L = ["", "== 4. Decision rule (set before the run), folds 2017-2026"]
    R = P[P["season"].isin(RULE_FOLDS) & (P["stage"] != "game 0")]
    for arm in ARMS[1:]:
        d_ll, d_br, _ = paired(R, arm)
        m, h = ci(d_ll)
        folds = [paired(x, arm)[0].mean() for _, x in R.groupby("season")]
        wins = sum(f < 0 for f in folds)
        r1 = m + h < 0 and d_br.mean() < 0 and wins >= 6
        st = {s: paired(R[R["stage"] == s], arm)[0].mean() for s in ("games 1-9", "games 10+")}
        r2 = all(v <= 0 for v in st.values())
        books = [gate.get(("reconstructed", "dk", 2026, "games 10+", arm)),
                 gate.get(("reconstructed", "espnbet", 2025, "games 10+", arm))]
        r3 = all(b is not None and b[0] <= 0 and b[1] <= 0 for b in books)
        r4 = min(folds) >= -0.008
        L.append(f"  {arm:4s} 1 games 1+ {m:+.4f} ± {h:.4f}, Brier {d_br.mean():+.4f}, "
                 f"{wins}/{len(folds)} folds: {'pass' if r1 else 'fail'}; 2 games 1-9 "
                 f"{st['games 1-9']:+.4f}, 10+ {st['games 10+']:+.4f}: "
                 f"{'pass' if r2 else 'fail'}; 3 priced books: {'pass' if r3 else 'fail'}; "
                 f"4 tripwire (best fold {min(folds):+.4f}): {'pass' if r4 else 'fail'} -> "
                 f"{'CANDIDATE' if r1 and r2 and r3 and r4 else 'not a candidate'}")
    return L


def report():
    P = load()
    L = ["Opponent-adjusted four factors: walk-forward, base route, v10h25 build (2026-11-05).",
         "Arms and rule fixed before the run (research/opponent_adjusted.py). "
         "Fold 2027 is information only.", ""]
    L += outcome_lines(P)
    L += dispersion_lines(P)
    ml, gate = market_lines(P)
    L += ml
    L += rule_lines(P, gate)
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "RESULTS.txt"), "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seasons", nargs="*", type=int,
                    default=[t for t in dyn.GAME_YEARS if t >= 2017])
    a = ap.parse_args()
    if a.run:
        run(a.seasons)
    if a.report:
        report()


if __name__ == "__main__":
    main()
