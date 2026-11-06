#!/usr/bin/env python3
"""Offence/defence weights refitted on game margins: the registered test (diagnostics/od_weights/REGISTRATION.md).

  python research/od_weights.py --run --seasons 2018 2019 2022 ...
  python research/od_weights.py --report

Registered 2026-11-06 (commit 900a57f), amended by the owner before any code
ran (commit dbacc45): the new era, 2016-17 .. today minus 2019-20 and
2020-21. Arms:
  base  production terms (v10h25 config: ORB%, rookies by pick, half-life 25).
  OD2   base + delta_off (O_h - O_a, the composite's offensive half) and
        doff_phase = delta_off x phase: offence and defence each with their
        own slope and season-phase growth, fitted on the final margin.
  GW8   (description only) the eight factor gaps (f_h - f_a) / sd x 100 in
        place of delta, with delta x phase and the other terms.

Walk-forward with production routing (research/interaction_weight.py): each
test season Y is scored by game-level fits on the era seasons before Y only
(base route at season start; report route refitted before each slate on the
report seasons before Y plus Y's covered games so far). Composite weights
keep production's protocol. 2026-27 to date is reported apart. Decision
rule: see the registration. ± is 95%, game-level, paired. Research only:
nothing here touches the ledgers, the pages or model/*.json.
"""
from __future__ import annotations

import argparse
import glob
import json
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

OUT = os.path.join(ROOT, "diagnostics", "od_weights")
CFG = v10c.CONFIGS["v10h25"]
ERA = [2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026, 2027]   # no 2020, 2021
SEASONS = ERA[1:]                                 # test seasons; 2027 to date apart
COMPLETED = [t for t in SEASONS if t <= 2026]
Z = [f"z{k}" for k in range(8)]
ARMS = ("base", "OD2", "GW8")
BASE = "base"
KEY = ["season", "slate_date", "home", "away"]
RULE_ARM = "OD2"


def arm_feats(arm, feats):
    """An arm's terms from a route's production terms."""
    if arm == "OD2":
        return list(feats) + ["delta_off", "doff_phase"]
    if arm == "GW8":
        return Z + [f for f in feats if f != "delta"]
    return list(feats)


# ------------------------------------------------------------- inputs --
_vecs: list = []


def fade_inputs_vecs(lh, ih, ph, la, ia, pa, lg_pct, weights, half_life=nc.HALF_LIFE):
    """nc.fade_inputs, recording each game's two feature vectors in order."""
    sh = nc.fade_side(lh, ih, ph, lg_pct, weights, half_life)
    sa = nc.fade_side(la, ia, pa, lg_pct, weights, half_life)
    if sh is None or sa is None:
        return None
    _vecs.append((np.asarray(sh[0], float), np.asarray(sa[0], float)))
    return dict(delta=nc.composite(sh[0] - sa[0], weights["sd"], weights["w"]),
                luck_def=sh[1] - sa[1], ft_diff=sh[2] - sa[2])


def add_terms(g, vecs, weights):
    """delta_off, doff_phase and the eight factor gaps z0..z7 (per SD x 100)."""
    sd, w = np.asarray(weights["sd"], float), np.asarray(weights["w"], float)
    D = np.array([fh - fa for fh, fa in vecs], float).reshape(-1, 8) / sd * 100
    g = g.copy()
    g["delta_off"] = D[:, :4] @ w[:4]
    g["doff_phase"] = g["delta_off"] * g["phase"]
    for k in range(8):
        g[Z[k]] = D[:, k]
    return g


def build(fn, t, wts):
    _vecs.clear()
    g = fn(t, wts)
    if len(_vecs) != len(g):
        raise RuntimeError(f"season {t}: {len(_vecs)} vectors for {len(g)} games")
    return add_terms(g, list(_vecs), wts)


def era_games(t, wts):
    return v10c.games(t, wts, CFG)


def implied(feats, b):
    c = dict(zip(["intercept"] + feats, b))
    out = {k: float(v) for k, v in c.items()}
    if "delta_off" in c:
        for ph in (0.5, 1.0):
            s = c["delta"] + c.get("d_phase", 0.0) * ph
            k = c["delta_off"] + c.get("doff_phase", 0.0) * ph
            out[f"off_share_{ph}"] = (s + k) / (2 * s + k)
    return out


def stage(cur):
    mn = np.fmin(cur["gp_home"], cur["gp_away"])
    return np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))


# --------------------------------------------------------------- folds --
def fold(Y):
    wts = dyn.fold_weights(Y)
    G = {t: build(era_games, t, wts) for t in ERA if t <= Y}
    rep = [t for t in v10c.REPORT_YEARS if t < Y and t in ERA]
    out, coefs = None, []
    for arm in ARMS:
        bf, af = arm_feats(arm, v10c.FEATS), arm_feats(arm, v10c.AVAIL)
        past = pd.concat([G[t] for t in G if t < Y], ignore_index=True)
        past = past[np.isfinite(past[bf + ["margin"]].to_numpy(float)).all(axis=1)]
        cur = G[Y][np.isfinite(G[Y][v10c.CORE].to_numpy(float)).all(axis=1)].copy()
        b, s = dyn.wls(past, feats=bf)
        cur["m"], cur["p"] = dyn.predict(b, s, cur, bf)
        cur["route"] = "base"
        coefs.append(dict(season=Y, arm=arm, route="base", sigma=s,
                          n_train=len(past), **implied(bf, b)))
        if rep:
            cpast = pd.concat([v10c.covered(G[t], t, CFG) for t in rep], ignore_index=True)
            ccur = v10c.covered(cur, Y, CFG)
            if len(ccur):
                o, _, st = dyn.arms(cpast, ccur, af, Y, f"{arm} avail")
                coefs.append(dict(season=Y, arm=arm, route="avail (static)",
                                  sigma=st["sigma"],
                                  **implied(af, [st[k] for k in ["intercept"] + af])))
                k = ["slate_date", "home", "away"]
                o = o.set_index(k)
                idx = cur.set_index(k).index
                hit = idx.isin(o.index)
                r = o.reindex(idx[hit])
                cur.loc[hit, "m"] = r["m_D"].to_numpy()
                cur.loc[hit, "p"] = r["p_D"].to_numpy()
                cur.loc[hit, "route"] = "avail"
        cur["stage"], cur["season"] = stage(cur), Y
        out = merge(out, cur, arm)
        dyn.log(f"fold {Y} {arm}: {len(cur)} games, base sigma {s:.2f}")
    return out, coefs


def merge(out, cur, arm):
    d = cur[KEY + ["gp_home", "gp_away", "stage", "route", "margin", "m", "p"]].rename(
        columns={"m": f"m_{arm}", "p": f"p_{arm}", "route": f"route_{arm}"})
    if out is None:
        return d
    return out.merge(d[KEY + [f"m_{arm}", f"p_{arm}", f"route_{arm}"]], on=KEY, how="inner")


def run(seasons):
    orb_pct.use("pct")
    nc.fade_inputs = fade_inputs_vecs
    os.makedirs(OUT, exist_ok=True)
    for Y in seasons:
        P, C = fold(Y)
        P.to_csv(os.path.join(OUT, f"predictions_{Y}.csv"), index=False, float_format="%.6g")
        with open(os.path.join(OUT, f"coefs_{Y}.json"), "w") as f:
            json.dump(C, f, indent=1)


# -------------------------------------------------------------- report --
def diff(g, arm):
    y = (g["margin"] > 0).to_numpy(float)
    d = dyn.ll(g[f"p_{arm}"], y) - dyn.ll(g[f"p_{BASE}"], y)
    b = (g[f"p_{arm}"] - y) ** 2 - (g[f"p_{BASE}"] - y) ** 2
    e = (g["margin"] - g[f"m_{arm}"]) ** 2 - (g["margin"] - g[f"m_{BASE}"]) ** 2
    return dyn.ci(d), dyn.ci(b), dyn.ci(e)


def outcome_lines(W, title):
    L = [f"== {title}: arm - base (log loss / Brier / margin MSE, ± 95%)"]
    G = W[W["stage"] != "game 0"]
    groups = [("games 1+", G)] + list(G.groupby("stage")) + \
        ([(f"{r} route", g) for r, g in G.groupby(f"route_{BASE}")]
         if G[f"route_{BASE}"].nunique() > 1 else [])
    for label, g in groups:
        y = (g["margin"] > 0).to_numpy(float)
        L.append(f"  {label:14s} n {len(g):5d}  base log loss {dyn.ll(g['p_base'], y).mean():.4f}")
        for arm in ARMS[1:]:
            (m, e, _), (bm, be, _), (mm, me, _) = diff(g, arm)
            wins = sum(diff(s, arm)[0][0] < 0 for _, s in g.groupby("season"))
            L.append(f"    {arm:4s} {m:+.4f} ± {e:.4f}  Brier {bm:+.4f} ± {be:.4f}  "
                     f"MSE {mm:+.2f} ± {me:.2f}  better in {wins} of {g['season'].nunique()}")
    L += ["", "  By season, games 1+ (log loss arm - base):"]
    for s, g in G.groupby("season"):
        L.append(f"    {int(s)}  n {len(g):5d}  " + "  ".join(
            f"{a} {c[0]:+.4f} ± {c[1]:.4f}" for a, c in
            ((a, diff(g, a)[0]) for a in ARMS[1:])))
    return L


def market_lines(W):
    L = ["", "== Against the close, each basis x book x season alone "
         "(arm - close log loss; [arm - base, Brier])"]
    J = W.merge(dyn.market_rows(), on=KEY, how="inner")
    gate = []
    for (basis, book, s), g in J.groupby(["basis", "close_book", "season"]):
        for st, h in [("games 1+", g[g["stage"] != "game 0"])] + list(g.groupby("stage")):
            if len(h) < 30:
                continue
            y = (h["margin"] > 0).to_numpy(float)
            lc = dyn.ll(h["close_q_home"].to_numpy(float), y)
            v = dyn.ci(dyn.ll(h["p_base"], y) - lc)
            L.append(f"  {basis:13s} {book:7s} {int(s)} {st:10s} n {len(h):4d}  "
                     f"base {v[0]:+.4f} ± {v[1]:.4f}")
            for arm in ARMS[1:]:
                a = dyn.ci(dyn.ll(h[f"p_{arm}"], y) - lc)
                (dm, de, _), (bm, be, _), _ = diff(h, arm)
                L.append(f"      {arm:4s} {a[0]:+.4f} ± {a[1]:.4f}  "
                         f"[{dm:+.4f} ± {de:.4f}, Brier {bm:+.4f} ± {be:.4f}]")
                if basis == "reconstructed" and st == "games 10+" and \
                        (book, int(s)) in (("dk", 2026), ("espnbet", 2025)):
                    gate.append((arm, book, int(s), dm, bm))
    return L, gate


NAMES = ["off eFG", "off TOV", "off ORB%", "off FTr", "def eFG", "def TOV", "def ORB%",
         "def FTr"]


def coef_table(C):
    L = ["", "== OD2 fitted weights by fold (off_share 0.5 = the composite's split)"]
    cols = ["delta", "d_phase", "delta_off", "doff_phase", "off_share_0.5", "off_share_1.0"]
    for route, g in C[C["arm"] == "OD2"].groupby("route", sort=False):
        L.append(f"  {route}")
        for d in g.to_dict("records"):
            L.append(f"    {int(d['season'])}  " + "  ".join(
                f"{c} {d[c]:+.3f}" for c in cols if c in d and pd.notna(d[c])))
    L += ["", "== GW8 (description): fitted slope per factor gap (per SD x 100), base route"]
    for d in C[(C["arm"] == "GW8") & (C["route"] == "base")].to_dict("records"):
        L.append(f"    {int(d['season'])}  " + "  ".join(
            f"{n} {d[z]:+.3f}" for n, z in zip(NAMES, Z)))
    return L


def gate_lines(W, C, gate):
    L = ["", "== Decision rule (registration as amended, commit dbacc45)"]
    G = W[(W["stage"] != "game 0") & W["season"].isin(COMPLETED)]
    (m, e, _), (bm, _, _), _ = diff(G, RULE_ARM)
    per = {int(s): diff(g, RULE_ARM)[0][0] for s, g in G.groupby("season")}
    wins = sum(v < 0 for v in per.values())
    c1 = m + e < 0 and wins >= 5
    sh = C[(C["arm"] == RULE_ARM) & (C["route"] == "base") & C["season"].isin(COMPLETED)]
    below = int((sh["off_share_1.0"] < 0.5).sum())
    c2 = below >= 5
    rows = [r for r in gate if r[0] == RULE_ARM]
    c3 = len(rows) == 2 and all(dm <= 0 and bm_ <= 0 for *_, dm, bm_ in rows)
    gain = max(-v for v in per.values())
    c4 = gain <= 0.008
    ok = c1 and c2 and c3 and c4
    L += [f"  1. games 1+, completed seasons: {m:+.4f} ± {e:.4f} (Brier {bm:+.4f}), better in "
          f"{wins} of {len(per)}  -> {'pass' if c1 else 'FAIL'}",
          f"  2. offence share < 0.5 at phase 1 in {below} of {len(sh)} folds  -> "
          f"{'pass' if c2 else 'FAIL'}",
          "  3. priced games 10+: " + ("; ".join(
              f"{b} {s} {dm:+.4f} (Brier {bm_:+.4f})" for _, b, s, dm, bm_ in rows) or "none")
          + f"  -> {'pass' if c3 else 'FAIL'}",
          f"  4. largest season gain {gain:.4f} (tripwire 0.008)  -> {'pass' if c4 else 'FAIL'}",
          f"  OD2: {'CANDIDATE (the owner decides)' if ok else 'NOT a candidate'}"]
    return L


def report():
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(
        os.path.join(OUT, "predictions_*.csv")))], ignore_index=True)
    P = P.dropna(subset=[f"p_{a}" for a in ARMS]).reset_index(drop=True)
    C = []
    for f in sorted(glob.glob(os.path.join(OUT, "coefs_*.json"))):
        C += json.load(open(f))
    C = pd.DataFrame(C)
    done, live = P[P["season"].isin(COMPLETED)], P[~P["season"].isin(COMPLETED)]
    L = [__doc__.strip().splitlines()[0], "",
         f"Completed test seasons {sorted(int(x) for x in done['season'].unique())}: "
         f"{len(done)} games; report route on {(done['route_base'] == 'avail').sum()}. "
         "Negative = arm better.", ""]
    L += outcome_lines(done, "New era, completed seasons (the rule's evidence)")
    if len(live):
        L += [""] + outcome_lines(live, "2026-27 to date (apart, not counted)")
    m, gate = market_lines(P)
    L += m + gate_lines(P, C, gate) + coef_table(C)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seasons", nargs="+", type=int, default=SEASONS)
    a = ap.parse_args()
    if a.run:
        run(a.seasons)
    if a.report:
        report()


if __name__ == "__main__":
    main()
