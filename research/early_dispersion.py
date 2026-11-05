#!/usr/bin/env python3
"""Early-season dispersion: walk-forward test of early weighting and a margin stretch.

  python research/early_dispersion.py --run [--seasons 2017 ... 2027]
  python research/early_dispersion.py --report

Owner's request (2026-11-05), after research/early_dispersion_check.py
(diagnostics/early_dispersion/RESULTS_season.txt): v11 trails the 2026-27
close by about +0.045 log loss on games 1-9 against roughly zero in the two
earlier priced early seasons. Its margins are compressed (SD 5.3 points
against the close's 6.9; outcome calibration slope 1.66 ± 0.40, 1 SE), and
on 91 games the close weights this season's form and current-roster talent
more, relative to last season's rating, than the model does. This tests,
on earlier seasons only, whether either fix would have helped.

Arms, all fixed before the run. Every fold Y refits on the game seasons
before Y (research/dynamic_fit.py's S protocol, the v10h25 feature build of
research/v10_candidate.py: ORB% composite, drafted rookies by pick,
half-life 25), OLS on the final home margin, P = Phi(m / sigma) with sigma
the fit's residual SD. e = max(0, 1 - (gp_home + gp_away) / 20), the
average share of last season left in the fade (1 at game 0, 0 from game
10). delta_now = the composite difference on this season's games only
(half-life 25, 0 before a team's first game).

  B    the base route: nba_composite.V6_FEATURES.
  E1   B + talent_diff·e        (current-roster talent counts more early)
  E2   E1 + delta_now·e         (this season's own form counts more early)
  E3   E2 + delta·e             (and the faded rating gets its own early slope)
  K    B's margin stretched by stage: m' = c_s + k_s·m_B, sigma_s the residual
       SD, (c_s, k_s, sigma_s) fitted on B's walk-forward predictions of the
       folds before Y for stage s (games 1-9; games 10+). Game 0 keeps B.

Limitation, stated before the run: the base route only. Production scores
covered games 1+ from 2024 on with the injury-report route; that archive is
not used here, so this measures the rating and talent weights, not the
report route's own fit. Any change would need its own report-route check.

Readings, each arm minus B on identical games (negative = better), ± 95%,
game-level, paired: log loss, Brier and margin MSE against outcomes, by
stage and fold; K is scored from fold 2019 (two earlier folds), so the
comparison and the rule use folds 2019-2026 for every arm. Against the
close, each book x season x stage alone (reconstructed rows, hindsight).
Fold 2027 (this season so far, against the rebuilt rows' DraftKings close)
is information only and enters no rule.

Decision rule, set before the run (with the G7 gate,
docs/model_guidelines.md): an arm is a candidate for a new model tag only if,
on folds 2019-2026,
  1. games 1-9 log loss arm - B is below 0 with its 95% interval excluding 0,
     Brier agreeing in sign, better in at least 6 of the 8 folds (as
     registered; 2019-2026 holds 7 folds, no 2019-20 season: read as 6 of 7,
     noted after the run, no verdict depends on it);
  2. games 10+ log loss arm - B is not above 0 (point estimate), Brier agreeing;
  3. on the large priced book-seasons (reconstructed DraftKings 2025-26 and
     ESPN BET 2024-25, games 10+), arm - B is not above 0, Brier agreeing;
  4. no fold improves by more than 0.008 (the leakage tripwire).
Passing makes the arm a candidate for the owner's decision and the usual
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

OUT = os.path.join(ROOT, "diagnostics", "early_dispersion")
CFG = v10c.CONFIGS["v10h25"]
FEATS = list(nc.V6_FEATURES)
ARMS = {"E1": FEATS + ["tal_e"],
        "E2": FEATS + ["tal_e", "now_e"],
        "E3": FEATS + ["tal_e", "now_e", "delta_e"]}
ALL = ["B", "E1", "E2", "E3", "K"]
RULE_FOLDS = list(range(2019, 2027))
STAGES = ("games 1-9", "games 10+")
_PHI = np.vectorize(NormalDist().cdf)
Z95 = 1.96


# ------------------------------------------------------------- inputs --
_logs = {}


def logs(t):
    if t not in _logs:
        _logs[t] = nc.load_logs(t)
    return _logs[t]


def games(t, wts):
    """Season t's games with the base terms plus the early interactions."""
    g = v10c.games(t, wts, CFG)
    L = logs(t)
    pct = nc.is_orb_pct(wts)

    def feats(tm, i):
        return nc.decayed_features(L[tm], i, nc.HALF_LIFE, orb_pct=pct) if i > 0 else None

    now = []
    for r in g.itertuples(index=False):
        fh, fa = feats(r.home, r.gp_home), feats(r.away, r.gp_away)
        now.append(nc.composite(fh - fa, wts["sd"], wts["w"])
                   if fh is not None and fa is not None else 0.0)
    g["delta_now"] = now
    g["e"] = np.clip(1.0 - (g["gp_home"] + g["gp_away"]) / 20.0, 0.0, 1.0)
    # a missing talent term is neutral (0) in prediction, as B fills it
    g["tal_e"] = (g["talent_diff"] * g["e"]).where(g["talent_diff"].notna(), 0.0)
    g["now_e"] = g["delta_now"] * g["e"]
    g["delta_e"] = g["delta"] * g["e"]
    return g


def fit_predict(past, cur, feats):
    p = past[np.isfinite(past[feats + ["margin"]].to_numpy(float)).all(axis=1)]
    b, s = dyn.wls(p, feats=feats)
    return dyn.predict(b, s, cur, feats)


def fold(Y):
    wts = dyn.fold_weights(Y)
    G = {t: games(t, wts) for t in dyn.GAME_YEARS if t <= Y}
    past = pd.concat([G[t] for t in G if t < Y], ignore_index=True)
    cur = G[Y][np.isfinite(G[Y][dyn.CORE].to_numpy(float)).all(axis=1)].copy()
    cur["m_B"], cur["p_B"] = fit_predict(past, cur, FEATS)
    for arm, feats in ARMS.items():
        cur[f"m_{arm}"], cur[f"p_{arm}"] = fit_predict(past, cur, feats)
    mn = np.fmin(cur["gp_home"], cur["gp_away"])
    cur["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
    cur["season"] = Y
    dyn.log(f"fold {Y}: {len(cur)} games, {len(past)} training rows")
    return cur[["season", "slate_date", "home", "away", "gp_home", "gp_away", "stage",
                "margin", "e", "talent_diff", "delta", "delta_now"] +
               [c for a in ["B"] + list(ARMS) for c in (f"m_{a}", f"p_{a}")]]


def add_stretch(P):
    """Arm K: B's margin restretched per stage on the earlier folds' B rows."""
    P = P.sort_values(["season", "slate_date"]).copy()
    P["m_K"], P["p_K"] = np.nan, np.nan
    for Y in sorted(P["season"].unique()):
        prior = P[P["season"] < Y]
        if prior["season"].nunique() < 2:
            continue
        cur = P["season"] == Y
        g0 = cur & (P["stage"] == "game 0")
        P.loc[g0, "m_K"], P.loc[g0, "p_K"] = P.loc[g0, "m_B"], P.loc[g0, "p_B"]
        for s in STAGES:
            tr = prior[prior["stage"] == s]
            X = np.column_stack([np.ones(len(tr)), tr["m_B"].to_numpy(float)])
            b, *_ = np.linalg.lstsq(X, tr["margin"].to_numpy(float), rcond=None)
            sigma = float(np.std(tr["margin"].to_numpy(float) - X @ b, ddof=2))
            idx = cur & (P["stage"] == s)
            if not idx.any():
                continue
            m = b[0] + b[1] * P.loc[idx, "m_B"].to_numpy(float)
            P.loc[idx, "m_K"], P.loc[idx, "p_K"] = m, _PHI(m / sigma)
            dyn.log(f"K fold {Y} {s}: c {b[0]:+.2f} k {b[1]:.3f} sigma {sigma:.2f} "
                    f"(n {len(tr)})")
    return P


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


def load():
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(
        os.path.join(OUT, "predictions_*.csv")))], ignore_index=True)
    P = P.drop_duplicates(["season", "slate_date", "home", "away"], keep="last")
    P = add_stretch(P)
    P["y"] = (P["margin"] > 0).astype(float)
    return P


def paired(g, arm):
    y = g["y"].to_numpy()
    d_ll = ll(g[f"p_{arm}"], y) - ll(g["p_B"], y)
    d_br = (g[f"p_{arm}"] - y) ** 2 - (g["p_B"] - y) ** 2
    d_mse = (g[f"m_{arm}"] - g["margin"]) ** 2 - (g["m_B"] - g["margin"]) ** 2
    return d_ll, d_br, d_mse


def ci(d):
    d = np.asarray(d, float)
    return d.mean(), Z95 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan


def outcome_lines(P):
    L = ["== 1. Against outcomes, folds 2019-2026: arm - B (log loss / Brier / margin "
         "MSE, ± 95%; negative = arm better)"]
    R = P[P["season"].isin(RULE_FOLDS)]
    for st in ("game 0",) + STAGES:
        g = R[R["stage"] == st]
        L.append(f"  {st:10s} n {len(g):5d}  B log loss {ll(g['p_B'], g['y']).mean():.4f}")
        for arm in ALL[1:]:
            d_ll, d_br, d_mse = paired(g, arm)
            wins = sum(paired(x, arm)[0].mean() < 0 for _, x in g.groupby("season"))
            m, h = ci(d_ll)
            mb, hb_ = ci(d_br)
            mm, hm = ci(d_mse)
            L.append(f"    {arm:3s} {m:+.4f} ± {h:.4f}  Brier {mb:+.4f} ± {hb_:.4f}  "
                     f"MSE {mm:+.2f} ± {hm:.2f}  better in {wins} of {g['season'].nunique()}")
    L.append("")
    L.append("  By fold, games 1-9 (log loss arm - B, ± 95%):")
    L.append("    season    n  " + "  ".join(f"{a:>17s}" for a in ALL[1:]))
    for Y, g in P[P["stage"] == "games 1-9"].groupby("season"):
        cells = []
        for arm in ALL[1:]:
            if g[f"p_{arm}"].isna().any():
                cells.append(f"{'--':>17s}")
                continue
            m, h = ci(paired(g, arm)[0])
            cells.append(f"{m:+.4f} ± {h:.4f}")
        L.append(f"    {Y}  {len(g):4d}  " + "  ".join(cells))
    return L


def fit_lines(P):
    """Dispersion of each arm on games 1-9: SD of margins, outcome ~ margin slope."""
    L = ["", "== 2. Dispersion on games 1-9, folds 2019-2026 (actual margin = a + b·arm "
         "margin, ± 95%; b > 1 = arm compressed)"]
    g = P[P["season"].isin(RULE_FOLDS) & (P["stage"] == "games 1-9")]
    for arm in ALL:
        X = np.column_stack([np.ones(len(g)), g[f"m_{arm}"].to_numpy(float)])
        y = g["margin"].to_numpy(float)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        se = np.sqrt(np.diag(r @ r / (len(y) - 2) * np.linalg.inv(X.T @ X)))
        L.append(f"  {arm:3s} SD margin {g[f'm_{arm}'].std():.2f}  slope {b[1]:.3f} ± "
                 f"{Z95 * se[1]:.3f}  intercept {b[0]:+.2f}")
    return L


def market_rows():
    rec = pd.read_csv(os.path.join(ROOT, "data", "nba_reconstructed.csv"))
    rec = rec.assign(basis="reconstructed")
    reb = pd.read_csv(os.path.join(ROOT, "data", "nba_rebuilt.csv")).assign(basis="rebuilt")
    M = pd.concat([rec, reb], ignore_index=True)
    M = M[M["close_q_home"].notna()]
    return M[["basis", "season", "slate_date", "home", "away", "close_book", "close_q_home"]]


def market_lines(P):
    L = ["", "== 3. Against the close, each basis x book x season x stage alone "
         "(hindsight; arm - close log loss, ± 95%; [arm - B])"]
    W = P.merge(market_rows(), on=["season", "slate_date", "home", "away"], how="inner")
    gate = {}
    for (basis, book, Y, st), g in W.groupby(["basis", "close_book", "season", "stage"]):
        if st == "game 0" or len(g) < 30:
            continue
        y = g["y"].to_numpy()
        lc = ll(g["close_q_home"], y)
        m, h = ci(ll(g["p_B"], y) - lc)
        L.append(f"  {basis:13s} {book:7s} {Y} {st:10s} n {len(g):4d}  B {m:+.4f} ± {h:.4f}")
        for arm in ALL[1:]:
            if g[f"p_{arm}"].isna().any():
                continue
            ma, ha = ci(ll(g[f"p_{arm}"], y) - lc)
            d_ll, d_br, _ = paired(g, arm)
            L.append(f"      {arm:3s} {ma:+.4f} ± {ha:.4f}  [{d_ll.mean():+.4f}, Brier "
                     f"{d_br.mean():+.4f}]")
            gate[(basis, book, Y, st, arm)] = (d_ll.mean(), d_br.mean())
    return L, gate


def rule_lines(P, gate):
    L = ["", "== 4. Decision rule (set before the run), folds 2019-2026",
         "  (the rule's '6 of 8 folds' miscounted: 2019-2026 holds 7 folds; read 6 of 7)"]
    R = P[P["season"].isin(RULE_FOLDS)]
    for arm in ALL[1:]:
        e = R[R["stage"] == "games 1-9"]
        d_ll, d_br, _ = paired(e, arm)
        m, h = ci(d_ll)
        wins = sum(paired(x, arm)[0].mean() < 0 for _, x in e.groupby("season"))
        r1 = m + h < 0 and d_br.mean() < 0 and wins >= 6
        late = R[R["stage"] == "games 10+"]
        l_ll, l_br, _ = paired(late, arm)
        r2 = l_ll.mean() <= 0 and l_br.mean() <= 0
        books = [gate.get(("reconstructed", "dk", 2026, "games 10+", arm)),
                 gate.get(("reconstructed", "espnbet", 2025, "games 10+", arm))]
        r3 = all(b is not None and b[0] <= 0 and b[1] <= 0 for b in books)
        best = min(paired(x, arm)[0].mean() for _, x in R.groupby("season"))
        r4 = best >= -0.008
        verdict = "CANDIDATE" if r1 and r2 and r3 and r4 else "not a candidate"
        L.append(f"  {arm:3s} 1 games 1-9 {m:+.4f} ± {h:.4f}, Brier {d_br.mean():+.4f}, "
                 f"{wins}/{e['season'].nunique()} folds: {'pass' if r1 else 'fail'}; 2 games 10+ "
                 f"{l_ll.mean():+.4f}: {'pass' if r2 else 'fail'}; 3 priced books: "
                 f"{'pass' if r3 else 'fail'}; 4 tripwire (best fold {best:+.4f}): "
                 f"{'pass' if r4 else 'fail'} -> {verdict}")
    return L


def report():
    P = load()
    L = ["Early-season dispersion: walk-forward, base route, v10h25 build (2026-11-05).",
         "Arms and rule fixed before the run (see research/early_dispersion.py). "
         "Fold 2027 is information only.", ""]
    L += outcome_lines(P)
    L += fit_lines(P)
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
