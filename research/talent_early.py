#!/usr/bin/env python3
"""Early-season talent weight: walk-forward test (talent_early_v1).

  python research/talent_early.py --run [--seasons 2017 ... 2026]
  python research/talent_early.py --report

Registered in docs/talent_weight_test_registration.md (2026-11-17T16:24:42Z,
commit c71a8d2), before this code. Arms, folds, readings and pass rule are
that file's; nothing here selects them.

  prod  production v11 routing and features, refitted per fold.
  tE    prod + talent_early = talent_diff x max(0, 1 - g/10),
        g = min(gp_home, gp_away), in both routes' least-squares margin fits.

Base route: season Y fitted on the base seasons before Y (static per
season, as production's file), composite weights per fold
(dynamic_fit.fold_weights). Report route (covered games 1+, seasons with
earlier report seasons): refitted before each slate on the report seasons
before Y plus this season's earlier covered games, trained on the od rule
at 30 min and scored with Questionable weighting at the 5 h lock (300 min),
play rates from report seasons before Y, existing matcher -- production's
v11 report route (report_name_suffix's machinery, arm q). Each game is
scored by the route production would use; both arms on identical games.
2026-27 is never a test season or in a fit.

Research only: nothing here touches data/, public/ or model/*.json.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
import dynamic_fit as dyn                         # noqa: E402
import orb_pct                                    # noqa: E402
import questionable_weight as qw                  # noqa: E402
import report_name_suffix as rns                  # noqa: E402
import v10_candidate as v10c                      # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "talent_early")
REG = "docs/talent_weight_test_registration.md (registered 2026-11-17T16:24:42Z, c71a8d2)"
SEASONS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
REPORT_FROM = 2023                       # first report season (2022-23)
LOCK_LEAD = 300                          # 5 h lock, minutes
KEY = ["slate_date", "home", "away"]
BASE = list(dyn.FEATS)
AVAIL = list(dyn.AVAIL_FEATS)
ARMS = {"prod": (BASE, AVAIL),
        "tE": (BASE + ["talent_early"], AVAIL + ["talent_early"])}
FADE = 10                                # production fade_games
COEFS = ("talent_early", "talent_diff", "delta", "d_phase")


def add_te(g):
    gp = np.fmin(g["gp_home"], g["gp_away"]).astype(float)
    g["talent_early"] = g["talent_diff"] * np.clip(1 - gp / FADE, 0, None)
    return g


def complete(df, feats):
    return df[np.isfinite(df[feats + ["margin"]].to_numpy(float)).all(axis=1)]


def refit_predict(past, train_cur, score_cur, feats):
    """qw.refit_predict with the arm's features: OLS on past + this season's
    earlier train rows before each slate; predict that slate's score rows."""
    past = complete(past, feats)
    train_cur = complete(train_cur, feats)
    score_cur = score_cur[np.isfinite(score_cur[feats].to_numpy(float)).all(axis=1)].copy()
    b0, s0 = dyn.wls(past, feats=feats)
    for d, idx in score_cur.groupby("date").groups.items():
        seen = train_cur[train_cur["date"] < d]
        b, s = dyn.wls(pd.concat([past, seen], ignore_index=True), feats=feats) \
            if len(seen) else (b0, s0)
        score_cur.loc[idx, "m"], score_cur.loc[idx, "p"] = dyn.predict(
            b, s, score_cur.loc[idx], feats)
    return score_cur.set_index(KEY)[["m", "p"]], (b0, s0)


def fold(Y):
    wts = dyn.fold_weights(Y)
    G = {t: add_te(v10c.games(t, wts, qw.CFG)) for t in dyn.GAME_YEARS if t <= Y}
    cur = G[Y][np.isfinite(G[Y][dyn.CORE].to_numpy(float)).all(axis=1)].copy()
    rep = [t for t in qw.REPORT_YEARS if REPORT_FROM <= t < Y]
    coef_rows = []
    for arm, (bf, af) in ARMS.items():
        past = complete(pd.concat([G[t] for t in G if t < Y], ignore_index=True), bf)
        b, s = dyn.wls(past, feats=bf)
        cur[f"m_{arm}"], cur[f"p_{arm}"] = dyn.predict(b, s, cur, bf)
        coef_rows.append(dict(season=Y, arm=arm, route="base", n=len(past), sigma=s,
                              **dict(zip(["intercept"] + bf, b))))
        route = pd.Series("base", index=cur.index)
        if rep:
            cpast = pd.concat([qw.covered(G[t], rns.terms(t, qw.TRAIN_LEAD, "od", Y, False))
                               for t in rep], ignore_index=True)
            train_cur = qw.covered(cur, rns.terms(Y, qw.TRAIN_LEAD, "od", Y, False))
            score_cur = qw.covered(cur, rns.terms(Y, LOCK_LEAD, "q", Y, False))
            pr, (b0, s0) = refit_predict(cpast, train_cur, score_cur, af)
            coef_rows.append(dict(season=Y, arm=arm, route="report", n=len(complete(cpast, af)),
                                  sigma=s0, **dict(zip(["intercept"] + af, b0))))
            idx = cur.set_index(KEY).index
            hit = idx.isin(pr.index)
            o = pr.reindex(idx[hit])
            cur.loc[hit, f"m_{arm}"] = o["m"].to_numpy()
            cur.loc[hit, f"p_{arm}"] = o["p"].to_numpy()
            route[hit] = "report"
        if arm == "prod":
            cur["route"] = route
    mn = np.fmin(cur["gp_home"], cur["gp_away"])
    cur["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
    cur["season"] = Y
    dyn.log(f"fold {Y}: {len(cur)} games, report route {(cur['route'] == 'report').sum()}")
    cols = ["season", "slate_date", "home", "away", "gp_home", "gp_away", "stage", "route",
            "margin", "talent_diff", "talent_early"] + \
        [f"{x}_{a}" for a in ARMS for x in ("m", "p")]
    return cur[cols], pd.DataFrame(coef_rows)


def run(seasons):
    orb_pct.use("pct")
    os.makedirs(OUT, exist_ok=True)
    for Y in seasons:
        assert Y < 2027, "2026-27 is never a test season"
        P, C = fold(Y)
        P.to_csv(os.path.join(OUT, f"predictions_{Y}.csv"), index=False, float_format="%.6g")
        C.to_csv(os.path.join(OUT, f"coefs_{Y}.csv"), index=False, float_format="%.6g")


# -------------------------------------------------------------- report --
def load(kind):
    return pd.concat([pd.read_csv(f) for f in
                      sorted(glob.glob(os.path.join(OUT, f"{kind}_*.csv")))], ignore_index=True)


def paired(g):
    y = (g["margin"] > 0).to_numpy(float)
    return (dyn.ci(dyn.ll(g["p_tE"], y) - dyn.ll(g["p_prod"], y)),
            dyn.ci((g["p_tE"] - y) ** 2 - (g["p_prod"] - y) ** 2))


def fmt(c):
    (m, e, n), (bm, be, _) = c
    return f"{m:+.5f} ± {e:.5f}  Brier {bm:+.5f} ± {be:.5f}  (n {n})"


def report():
    P = load("predictions")
    C = load("coefs")
    y_all = (P["margin"] > 0).to_numpy(float)
    L = ["Early-season talent weight: walk-forward test (research/talent_early.py,"
         " talent_early_v1).",
         f"Registration: {REG}.",
         "Arm tE minus prod, paired by game, ± 95%; negative = tE better. Production routing;",
         "report route at the 5 h lock (q rule, 300 min), trained od at 30 min.", ""]
    L.append(f"Games: {len(P)} over seasons {', '.join(f'{s - 1}-{str(s)[2:]}' for s in sorted(P['season'].unique().tolist()))}; "
             f"report route {(P['route'] == 'report').sum()}; prod log loss games 1+ "
             f"{dyn.ll(P.loc[P['stage'] != 'game 0', 'p_prod'], y_all[P['stage'] != 'game 0']).mean():.4f}")
    L.append("")
    E = P[P["stage"] == "games 1-9"]
    r1 = paired(E)
    L.append("== 1-2. Primary: games 1-9 against outcomes (log loss; Brier)")
    L.append(f"  pooled     {fmt(r1)}")
    better, seas = 0, []
    for s, h in E.groupby("season"):
        c = paired(h)
        seas.append(c[0][0])
        better += c[0][0] < 0
        L.append(f"  {s - 1}-{str(s)[2:]}    {fmt(c)}")
    for r, h in E.groupby("route"):
        L.append(f"  route {r:6s} {fmt(paired(h))}")
    L.append(f"  tE better in {better} of {len(seas)} seasons")
    L.append("")
    T = P[P["stage"] == "games 10+"]
    r3 = paired(T)
    L.append("== 3. Games 10+ against outcomes")
    L.append(f"  pooled     {fmt(r3)}")
    for s, h in T.groupby("season"):
        L.append(f"  {s - 1}-{str(s)[2:]}    {fmt(paired(h))}")
    Z = P[P["stage"] == "game 0"]
    if len(Z):
        L.append(f"  game 0 (not in the rule) {fmt(paired(Z))}")
    L.append("")
    J = P.merge(dyn.market_rows(), on=["season"] + KEY, how="inner")
    J = J[J["basis"] == "reconstructed"]
    L.append("== 4-5. Against the close (reconstructed): arm − close log loss; [tE − prod]")
    gate = {}
    for (book, s, st), h in J.groupby(["close_book", "season", "stage"]):
        if st == "game 0" or len(h) < 50:
            continue
        y = (h["margin"] > 0).to_numpy(float)
        lc = dyn.ll(h["close_q_home"].to_numpy(float), y)
        a, b = dyn.ci(dyn.ll(h["p_prod"], y) - lc), dyn.ci(dyn.ll(h["p_tE"], y) - lc)
        d = paired(h)
        L.append(f"  {book:7s} {s - 1}-{str(s)[2:]} {st:9s} n {len(h):4d}  prod {a[0]:+.4f} ± "
                 f"{a[1]:.4f}  tE {b[0]:+.4f} ± {b[1]:.4f}  [{d[0][0]:+.5f} ± {d[0][1]:.5f},"
                 f" Brier {d[1][0]:+.5f}]")
        if st == "games 10+" and (book, s) in (("dk", 2026), ("espnbet", 2025)):
            gate[(book, s)] = d
    L.append("")
    L.append("== 6. Fitted coefficients per fold (season-start fit; report route refits per slate)")
    for (s, route), h in C.groupby(["season", "route"]):
        te = h[h["arm"] == "tE"].iloc[0]
        pr = h[h["arm"] == "prod"].iloc[0]
        L.append(f"  {s - 1}-{str(s)[2:]} {route:6s} n {int(te['n']):5d}  tE: talent_early "
                 f"{te['talent_early']:+.3f} talent_diff {te['talent_diff']:+.3f} delta "
                 f"{te['delta']:+.3f} d_phase {te['d_phase']:+.3f} | prod talent_diff "
                 f"{pr['talent_diff']:+.3f}")
    L.append(f"  corr(talent_early, talent_diff) games 1-9: "
             f"{E[['talent_early', 'talent_diff']].corr().iloc[0, 1]:+.2f}")
    L.append("")
    L.append("== Pass rule (registered)")

    def not_worse(c):
        m, e, _ = c[0]
        return m <= 0 or (m - e < 0 and abs(m) < 0.0005)
    (m1, e1, _), (b1, _, _) = r1
    a = m1 < 0 and m1 + e1 < 0
    b = np.sign(b1) == np.sign(m1)
    c = better >= len(seas) / 2
    d = not_worse(r3) and len(gate) == 2 and all(not_worse(v) for v in gate.values())
    trip = [x for x in seas if x < -0.008]
    e = not trip
    for lab, ok, txt in (
            ("(a)", a, f"games 1-9 log loss {m1:+.5f} ± {e1:.5f}, upper bound {m1 + e1:+.5f} < 0"),
            ("(b)", b, f"Brier {b1:+.5f}, same sign"),
            ("(c)", c, f"better in {better} of {len(seas)} seasons (need ≥ half)"),
            ("(d)", d, "games 10+ outcomes " + f"{r3[0][0]:+.5f} ± {r3[0][1]:.5f}; " +
             "; ".join(f"{k[0]} {k[1]} {v[0][0]:+.5f} ± {v[0][1]:.5f}" for k, v in gate.items())
             + ("" if len(gate) == 2 else " (a gate book-season is missing)")),
            ("(e)", e, "no season gain over 0.008" if e else
             f"tripwire: {len(trip)} season(s) beyond -0.008; check for leakage")):
        L.append(f"  {lab} {'pass' if ok else 'FAIL'}  {txt}")
    L.append(f"  VERDICT: tE {'PASSES' if all((a, b, c, d, e)) else 'FAILS'} the registered rule")
    L.append("")
    # 7. information only: 2026-27 native locks (selection data)
    L += native_lines()
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def native_lines():
    """Reading 7: tE's coefficient from the latest fold applied on the
    2026-27 locks' rebuilt features (loss_feature_audit cache): margin
    shift talent_early_coef x talent_early on the lock margin. Selection
    data, information only."""
    import loss_feature_audit as A
    L = ["== 7. Information only (selection data, not in the rule): 2026-27 native locks"]
    try:
        m = A.population()
    except Exception as e:  # noqa: BLE001
        return L + [f"  unavailable ({e!r}); run loss_feature_audit.py --build"]
    C = load("coefs")
    last = C[(C["arm"] == "tE") & (C["route"] == "report")].sort_values("season").iloc[-1]
    m = add_te(m.dropna(subset=["talent_diff"]).copy())
    sig = last["sigma"]
    p0 = m["lock_p_home"].astype(float).clip(1e-4, 1 - 1e-4)
    m0 = sig * A.norm.ppf(p0)
    p1 = A.norm.cdf((m0 + last["talent_early"] * m["talent_early"]) / sig)
    y = m["home_won"].astype(float).to_numpy()
    for lab, sel in (("games 1-9", m["gp"] < 10), ("games 1+", m["gp"] >= 1)):
        d = dyn.ci(dyn.ll(p1[sel.to_numpy()], y[sel.to_numpy()])
                   - dyn.ll(p0[sel].to_numpy(), y[sel.to_numpy()]))
        L.append(f"  {lab}: tE − lock log loss {d[0]:+.4f} ± {d[1]:.4f} (n {d[2]}), "
                 f"talent_early coef {last['talent_early']:+.3f} (fold {int(last['season'])})")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seasons", nargs="*", type=int, default=list(SEASONS))
    a = ap.parse_args()
    if a.run:
        run(a.seasons)
    if a.report:
        report()


if __name__ == "__main__":
    main()
