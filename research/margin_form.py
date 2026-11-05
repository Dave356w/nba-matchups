#!/usr/bin/env python3
"""Margin form term: walk-forward test under production routing.

  python research/margin_form.py --run [--seasons 2017 ... 2027]
  python research/margin_form.py --report     # -> diagnostics/margin_form/RESULTS.txt

Owner's request (2026-11-05), from the box-score projector work (PR #147):
the projector's fitted blend put half its weight on the last 10 games, and
the totals model keeps a 5-game form term (research/totals_improve.py:
squared error -1.54 ± 1.27), but the margin model has none; its four-factor
rating uses one 25-game decay, and the half-life sweep only replaced that
single decay (diagnostics/half_life_sweep). This tests a short-window form
term beside it. Arms, routing and rule fixed before the run.

Form, per game: form_h = delta at half-life h - delta at half-life 25, where
delta is the production composite difference (home - away) on the faded
rating window (nba_composite.fade_inputs: last season x rho fading out by
game 10, this season's games strictly before the game, the fold's composite
weights), so form is how far recent play departs from the 25-game rating.

Routing as production (research/v10_candidate.py, config v10h25: ORB%,
drafted rookies by pick, half-life 25): each fold Y refits the composite
weights on the weight seasons before Y; the base route is one OLS margin fit
on the game seasons before Y (static); games the NBA report covers, from
2023-24 on, use the report route (player_availability.FEATURES_V6), refitted
before every slate on the report seasons before Y plus season Y's covered
games so far (the D arm, as v10 ships). P = Phi(margin / sigma). The report
route reads od statuses (v11's Questionable weighting is not reproduced
here); both arms share the routing, so the comparison isolates the term.

Arms:
  B    production terms (base nba_composite.V6_FEATURES, report
       player_availability.FEATURES_V6)
  F5   B + form_5 on both routes
  F10  B + form_10 on both routes

Readings, each arm minus B on identical games (negative = better), ± 95%,
paired: log loss, Brier and margin MSE against outcomes, by stage (game 0,
1-9, 10+), route and fold; against the close, each basis x book x season x
stage alone (reconstructed rows and the 2026-27 rebuilt rows, hindsight).

Decision rule, set before the run (the G7 gate as v10 applied it,
docs/model_guidelines.md): an arm is a candidate for a new model tag only
if, on folds 2017-2026, its log loss is not worse than B's on either large
priced book-season (reconstructed DraftKings 2025-26 and ESPN BET 2024-25,
games 10+), with Brier agreeing, and not worse against outcomes on games
1+; an improvement over 0.008 in any season is a leakage tripwire. Passing
makes it a candidate for the owner's decision and an amendment, not a
change. Research only: nothing here touches the ledgers, pages or
model/*.json.
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
import nba_composite as nc                        # noqa: E402
import dynamic_fit as dyn                         # noqa: E402
import orb_pct                                    # noqa: E402
import v10_candidate as v10c                      # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "margin_form")
CFG = v10c.CONFIGS["v10h25"]
FEATS, AVAIL = list(dyn.FEATS), list(dyn.AVAIL_FEATS)
ARMS = {"B": [], "F5": ["form_5"], "F10": ["form_10"]}
RULE_FOLDS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
Z95 = 1.96


# --------------------------------------------------------------- inputs --
_logs = {}


def logs(t):
    if t not in _logs:
        try:
            _logs[t] = nc.load_logs(t)
        except Exception:                          # noqa: BLE001
            _logs[t] = {}
    return _logs[t]


def with_form(g, t, wts):
    """Season t's base-route games plus form_5 and form_10."""
    L, P = logs(t), logs(t - 1)
    lg3 = nc.league_3p(L)
    pct0 = nc.last_season_3p(P)
    f5, f10 = [], []
    for r in g.itertuples(index=False):
        lh, la = L.get(r.home), L.get(r.away)
        if lh is None or la is None:
            f5.append(np.nan); f10.append(np.nan)
            continue
        pct = lg3.get(pd.Timestamp(r.date), np.nan)
        pct = pct if np.isfinite(pct) else pct0
        out = []
        for hl in (5.0, 10.0):
            x = nc.fade_inputs(lh, int(r.gp_home), P.get(r.home), la, int(r.gp_away),
                               P.get(r.away), pct, wts, hl)
            out.append(x["delta"] - r.delta if x is not None else np.nan)
        f5.append(out[0]); f10.append(out[1])
    g = g.copy()
    g["form_5"], g["form_10"] = f5, f10
    return g


def fold(Y):
    """Season Y's games with every arm's production-routed prediction."""
    wts = dyn.fold_weights(Y)
    G = {t: with_form(v10c.games(t, wts, CFG), t, wts) for t in dyn.GAME_YEARS if t <= Y}
    cur = G[Y][np.isfinite(G[Y][dyn.CORE].to_numpy(float)).all(axis=1)].copy()
    rep = [t for t in v10c.REPORT_YEARS if t < Y]
    key = ["slate_date", "home", "away"]
    for arm, extra in ARMS.items():
        feats, afeats = FEATS + extra, AVAIL + extra
        past = pd.concat([G[t] for t in G if t < Y], ignore_index=True)
        past = past[np.isfinite(past[feats + ["margin"]].to_numpy(float)).all(axis=1)]
        b, s = dyn.wls(past, feats=feats)
        cur[f"m_{arm}"], cur[f"p_{arm}"] = dyn.predict(b, s, cur, feats)
        cur[f"route_{arm}"] = "base"
        if rep:
            cpast = pd.concat([v10c.covered(G[t], t, CFG) for t in rep], ignore_index=True)
            ccur = v10c.covered(cur, Y, CFG)
            if len(ccur):
                out, _, _ = dyn.arms(cpast, ccur, afeats, Y, f"avail {arm}")
                o = out.set_index(key)
                idx = cur.set_index(key).index
                hit = idx.isin(o.index)
                oo = o.reindex(idx[hit])
                cur.loc[hit, f"m_{arm}"] = oo["m_D"].to_numpy()
                cur.loc[hit, f"p_{arm}"] = oo["p_D"].to_numpy()
                cur.loc[hit, f"route_{arm}"] = "avail"
    mn = np.fmin(cur["gp_home"], cur["gp_away"])
    cur["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
    cur["season"] = Y
    dyn.log(f"fold {Y}: {len(cur)} games; report route {int((cur['route_B'] == 'avail').sum())}")
    return cur[["season", "slate_date", "home", "away", "gp_home", "gp_away", "stage", "margin",
                "form_5", "form_10", "route_B"] +
               [c for a in ARMS for c in (f"m_{a}", f"p_{a}")]]


def run(seasons):
    orb_pct.use("pct" if CFG["orb"] else "per100")
    P = pd.concat([fold(Y) for Y in seasons], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    P.to_csv(os.path.join(OUT, f"predictions_{min(seasons)}-{max(seasons)}.csv"), index=False,
             float_format="%.6g")


# -------------------------------------------------------------- report --
def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ci(d):
    d = np.asarray(d, float)
    return d.mean(), Z95 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan


def paired(g, arm):
    y = g["y"].to_numpy()
    return (ll(g[f"p_{arm}"], y) - ll(g["p_B"], y), (g[f"p_{arm}"] - y) ** 2 - (g["p_B"] - y) ** 2,
            (g[f"m_{arm}"] - g["margin"]) ** 2 - (g["m_B"] - g["margin"]) ** 2)


def load():
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "predictions_*.csv")))],
                  ignore_index=True)
    P = P.drop_duplicates(["season", "slate_date", "home", "away"], keep="last")
    P["y"] = (P["margin"] > 0).astype(float)
    return P


def report():
    P = load()
    R = P[P["season"].isin(RULE_FOLDS)]
    L = ["Margin form term, walk-forward under production routing (v10h25, report route D),",
         "folds 2017-2026 for the rule; 2026-27 information only. Arm - B, ± 95%.", ""]
    L.append("== 1. Against outcomes (log loss / Brier / margin MSE)")
    for lab, g in [("games 1+", R[R["stage"] != "game 0"])] + \
            [(s, R[R["stage"] == s]) for s in ("games 1-9", "games 10+")] + \
            [(f"route {r}, games 1+", R[(R["route_B"] == r) & (R["stage"] != "game 0")])
             for r in ("base", "avail")]:
        L.append(f"  {lab:22s} n {len(g):5d}  B log loss {ll(g['p_B'], g['y']).mean():.4f}")
        for arm in list(ARMS)[1:]:
            d_ll, d_br, d_mse = paired(g, arm)
            m, h = ci(d_ll)
            wins = sum(paired(x, arm)[0].mean() < 0 for _, x in g.groupby("season"))
            L.append(f"    {arm:4s} {m:+.4f} ± {h:.4f}  Brier {d_br.mean():+.5f}  MSE "
                     f"{ci(d_mse)[0]:+.2f} ± {ci(d_mse)[1]:.2f}  better in {wins} of {g['season'].nunique()}")
    L.append("  By fold, games 1+ (log loss arm - B):")
    for Y, g in P[P["stage"] != "game 0"].groupby("season"):
        L.append(f"    {Y} n {len(g):4d}  " + "  ".join(
            f"{a} {ci(paired(g, a)[0])[0]:+.4f} ± {ci(paired(g, a)[0])[1]:.4f}" for a in list(ARMS)[1:]))
    rec = pd.read_csv(os.path.join(ROOT, "data", "nba_reconstructed.csv")).assign(basis="reconstructed")
    reb = pd.read_csv(os.path.join(ROOT, "data", "nba_rebuilt.csv")).assign(basis="rebuilt")
    M = pd.concat([rec, reb])[["basis", "season", "slate_date", "home", "away", "close_book",
                               "close_q_home"]].dropna(subset=["close_q_home"])
    W = P.merge(M, on=["season", "slate_date", "home", "away"])
    L += ["", "== 2. Against the close, each basis x book x season x stage (arm - close; [arm - B, Brier])"]
    gate = {}
    for (basis, book, Y, st), g in W.groupby(["basis", "close_book", "season", "stage"]):
        if st == "game 0" or len(g) < 30:
            continue
        y = g["y"].to_numpy()
        lc = ll(g["close_q_home"], y)
        m, h = ci(ll(g["p_B"], y) - lc)
        L.append(f"  {basis:13s} {book:7s} {Y} {st:10s} n {len(g):4d}  B {m:+.4f} ± {h:.4f}")
        for arm in list(ARMS)[1:]:
            d_ll, d_br, _ = paired(g, arm)
            L.append(f"      {arm:4s} {ci(ll(g[f'p_{arm}'], y) - lc)[0]:+.4f}  [{d_ll.mean():+.4f} ± "
                     f"{ci(d_ll)[1]:.4f}, Brier {d_br.mean():+.5f}]")
            gate[(basis, book, Y, st, arm)] = (d_ll.mean(), d_br.mean())
    L += ["", "== 3. Decision rule (set before the run; the G7 gate as v10 applied it)"]
    for arm in list(ARMS)[1:]:
        books = [gate.get(("reconstructed", "dk", 2026, "games 10+", arm)),
                 gate.get(("reconstructed", "espnbet", 2025, "games 10+", arm))]
        r1 = all(b is not None and b[0] <= 0 and b[1] <= 0 for b in books)
        g1 = R[R["stage"] != "game 0"]
        d_ll, _, _ = paired(g1, arm)
        r2 = d_ll.mean() <= 0
        best = min(paired(x, arm)[0].mean() for _, x in g1.groupby("season"))
        r3 = best >= -0.008
        L.append(f"  {arm:4s} priced books DK 2025-26 {books[0][0]:+.4f} / ESPN BET 2024-25 "
                 f"{books[1][0]:+.4f} (Brier {books[0][1]:+.5f} / {books[1][1]:+.5f}): "
                 f"{'pass' if r1 else 'fail'}; games 1+ {d_ll.mean():+.4f}: {'pass' if r2 else 'fail'}; "
                 f"tripwire (best fold {best:+.4f}): {'pass' if r3 else 'fail'} -> "
                 f"{'CANDIDATE' if r1 and r2 and r3 else 'not a candidate'}")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "RESULTS.txt"), "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seasons", nargs="*", type=int, default=[t for t in dyn.GAME_YEARS if t >= 2017])
    a = ap.parse_args()
    if a.run:
        run(a.seasons)
    if a.report:
        report()


if __name__ == "__main__":
    main()
