#!/usr/bin/env python3
"""Interaction weighting: home/away and offence/defence weights on the production model, walk-forward.

  python research/interaction_weight.py --run --seasons 2017 2018 ...
  python research/interaction_weight.py --report

Owner's request (2026-11-06): test on the production model the weighting of
the P2 box-score projector (Colab notebook; arm P2 of the owner's projector
research). P2 projects home = w ORtg_h + (1 - w) DRtg_a and away = (1 - w)
ORtg_a + w DRtg_h (x pace), fitted w = 0.55 for 2025-26: the home team's
own ratings weigh 0.55 in both scores. With "higher = better" ratings its
margin is w C_h - (1 - w) C_a, C = offence + defence, so P2's w is a weight
on the home team's rating against the away team's. Production's composite
gap (delta) forces w = 0.5 and one weight on offence and defence.

Every team's composite C = O + D splits exactly into its offensive half O
(off eFG%, TOV, ORB%, FTr) and defensive half D (the four opponent factors),
each at the production weights (weights.json is linear in the features).
Lower-case o, d are O, D minus the league's to date (games strictly before
the date this season, with last season's league totals worth 10 games a
team), so the level of the league in each era drops out.

Arms, fixed before the run (each adds terms to both routes; nothing else
changes):
  base  production terms (v10h25 config: ORB%, rookies by pick, half-life
        25; base route nba_composite.V6_FEATURES, report route
        player_availability.FEATURES_V6).
  HA    + c_sum = (o_h + d_h) + (o_a + d_a): home vs away weight (P2's w).
        Implied w = (b + g) / (2 b), b delta's slope at the phase, g c_sum's.
  OD    + delta_off = O_h - O_a: offence vs defence weight. Implied offence
        share = (b + k) / (2 b + k).
  HAOD  + delta_off, c_sum, o_sum = o_h + o_a: free weights on O_h, D_h,
        O_a, D_a (P2's linear structure, generalised).
  MX    + mx = (o_h d_a - o_a d_h) / 10: a multiplicative matchup, a home
        offence against a weak away defence (and the reverse) beyond the
        sum of their ratings.

Production routing as research/v10_candidate.py (config v10h25): a game the
NBA report covers, both teams past game 0, in a season with an earlier
report season (2024 on) is scored by the report route, refitted before each
slate (dynamic_fit's D); every other game by the base route, fitted once per
fold on the game seasons before Y. Composite weights per fold on
WEIGHT_YEARS before Y. The report route reads Out/Doubtful (v10's statuses);
v11's Questionable weighting is not reproduced here, the same for every arm.

Readings, each arm minus base on identical games (negative = arm better),
± 95%, game-level, paired: log loss, Brier, margin MSE against outcomes, by
stage, route and season; against the close, each basis x book x season
alone; the fitted weights by fold. Decision rule (the G7 change gate,
docs/model_guidelines.md), set before the run: an arm is a candidate only if
its log loss is not worse than base on either large priced book-season
(reconstructed DraftKings 2025-26 and ESPN BET 2024-25, games 10+), Brier
agreeing, and it is better on games 1+ beyond the 95% interval and in at
least 7 of the folds; a gain over 0.008 in a season is a leakage tripwire.
Passing makes it a candidate for a new model tag, the owner's decision, not
a change. Research only: nothing here touches the ledgers, the pages or
model/*.json.
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

OUT = os.path.join(ROOT, "diagnostics", "interaction_weight")
CFG = v10c.CONFIGS["v10h25"]
SEASONS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
PRIOR_GAMES = 10                                  # last season's league, games a team
EXTRA = {"base": [], "HA": ["c_sum"], "OD": ["delta_off"],
         "HAOD": ["delta_off", "c_sum", "o_sum"], "MX": ["mx"]}
ARMS = list(EXTRA)
BASE = "base"
KEY = ["season", "slate_date", "home", "away"]


# ------------------------------------------------------------- inputs --
def halves(f, weights):
    """(O, D): the composite's offensive and defensive halves (win%-points);
    O + D == nc.composite(f, sd, w)."""
    z = np.asarray(f, float) / np.asarray(weights["sd"]) * np.asarray(weights["w"]) * 100
    return float(z[:4].sum()), float(z[4:].sum())


_parts: list = []


def fade_inputs_halves(lh, ih, ph, la, ia, pa, lg_pct, weights, half_life=nc.HALF_LIFE):
    """nc.fade_inputs, recording each game's O_h, D_h, O_a, D_a in order."""
    sh = nc.fade_side(lh, ih, ph, lg_pct, weights, half_life)
    sa = nc.fade_side(la, ia, pa, lg_pct, weights, half_life)
    if sh is None or sa is None:
        return None
    _parts.append(halves(sh[0], weights) + halves(sa[0], weights))
    return dict(delta=nc.composite(sh[0] - sa[0], weights["sd"], weights["w"]),
                luck_def=sh[1] - sa[1], ft_diff=sh[2] - sa[2])


_logs: dict = {}


def logs(t):
    if t not in _logs:
        try:
            _logs[t] = nc.load_logs(t)
        except Exception:  # noqa: BLE001 - no season: no prior
            _logs[t] = {}
    return _logs[t]


def league_halves(t, dates, weights):
    """{date: (O_lg, D_lg)}: the league's composite halves over season t's
    games strictly before each date, plus last season's league totals
    scaled to PRIOR_GAMES games a team."""
    rows = [lg[["date"] + nc.COLS] for lg in logs(t).values() if len(lg)]
    cur = pd.concat(rows).groupby("date")[nc.COLS].sum().sort_index()
    prev = [lg[nc.COLS] for lg in logs(t - 1).values() if len(lg)]
    prior = np.zeros(len(nc.COLS))
    if prev:
        P = pd.concat(prev)
        prior = P.sum().to_numpy(float) * (PRIOR_GAMES * 30 / len(P))
    cum = cur.cumsum()
    out = {}
    for d in sorted(set(pd.to_datetime(dates))):
        before = cum[cum.index < d]
        tot = prior + (before.iloc[-1].to_numpy(float) if len(before) else 0.0)
        f = nc.features_from_totals(tot, orb_pct=nc.is_orb_pct(weights))
        out[d] = halves(f, weights)
    return out


def add_terms(g, parts, lg):
    """The arms' terms from the halves and the league's to date."""
    P = np.asarray(parts, float).reshape(-1, 4)
    L = np.array([lg[pd.Timestamp(d)] for d in g["date"]], float).reshape(-1, 2)
    Oh, Dh, Oa, Da = P.T
    oh, dh, oa, da = Oh - L[:, 0], Dh - L[:, 1], Oa - L[:, 0], Da - L[:, 1]
    g = g.copy()
    g["delta_off"] = Oh - Oa
    g["c_sum"] = oh + dh + oa + da
    g["o_sum"] = oh + oa
    g["mx"] = (oh * da - oa * dh) / 10
    return g


def games(t, wts):
    _parts.clear()
    g = v10c.games(t, wts, CFG)
    if len(_parts) != len(g):
        raise RuntimeError(f"season {t}: {len(_parts)} halves for {len(g)} games")
    return add_terms(g, list(_parts), league_halves(t, g["date"], wts))


# --------------------------------------------------------------- folds --
def implied(feats, b):
    """Implied weights from a fit's coefficients at phase 0.5 and 1."""
    c = dict(zip(["intercept"] + feats, b))
    out = {}
    for ph in (0.5, 1.0):
        s = c["delta"] + c.get("d_phase", 0.0) * ph
        if "c_sum" in c:
            out[f"home_w@{ph}"] = (s + c["c_sum"]) / (2 * s)
        if "delta_off" in c:
            out[f"off_share@{ph}"] = (s + c["delta_off"]) / (2 * s + c["delta_off"])
    return {**{k: float(v) for k, v in c.items()}, **out}


def fold(Y):
    wts = dyn.fold_weights(Y)
    G = {t: games(t, wts) for t in dyn.GAME_YEARS if t <= Y}
    rep = [t for t in v10c.REPORT_YEARS if t < Y]
    out, coefs = None, []
    for arm in ARMS:
        bf = v10c.FEATS + EXTRA[arm]
        af = v10c.AVAIL + EXTRA[arm]
        past = pd.concat([G[t] for t in G if t < Y], ignore_index=True)
        past = past[np.isfinite(past[bf + ["margin"]].to_numpy(float)).all(axis=1)]
        cur = G[Y][np.isfinite(G[Y][v10c.CORE].to_numpy(float)).all(axis=1)].copy()
        b, s = dyn.wls(past, feats=bf)
        cur["m"], cur["p"] = dyn.predict(b, s, cur, bf)
        cur["route"] = "base"
        coefs.append(dict(season=Y, arm=arm, route="base", sigma=s, **implied(bf, b)))
        if rep:
            cpast = pd.concat([v10c.covered(G[t], t, CFG) for t in rep], ignore_index=True)
            ccur = v10c.covered(cur, Y, CFG)
            if len(ccur):
                o, _, st = dyn.arms(cpast, ccur, af, Y, f"{arm} avail")
                coefs.append(dict(season=Y, arm=arm, route="avail (static)", sigma=st["sigma"],
                                  **implied(af, [st[k] for k in ["intercept"] + af])))
                k = ["slate_date", "home", "away"]
                o = o.set_index(k)
                idx = cur.set_index(k).index
                hit = idx.isin(o.index)
                r = o.reindex(idx[hit])
                cur.loc[hit, "m"] = r["m_D"].to_numpy()
                cur.loc[hit, "p"] = r["p_D"].to_numpy()
                cur.loc[hit, "route"] = "avail"
        mn = np.fmin(cur["gp_home"], cur["gp_away"])
        cur["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
        cur["season"] = Y
        d = cur[KEY + ["gp_home", "gp_away", "stage", "route", "margin", "m", "p"]].rename(
            columns={"m": f"m_{arm}", "p": f"p_{arm}", "route": f"route_{arm}"})
        out = d if out is None else out.merge(
            d[KEY + [f"m_{arm}", f"p_{arm}", f"route_{arm}"]], on=KEY, how="inner")
        dyn.log(f"fold {Y} {arm}: {len(d)} games, base sigma {s:.2f}")
    return out, coefs


def run(seasons):
    orb_pct.use("pct")
    nc.fade_inputs = fade_inputs_halves
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


def others():
    return [a for a in ARMS if a != BASE]


def outcome_lines(W):
    L = ["== 1. Against outcomes: arm - base (log loss / Brier / margin MSE, ± 95%)"]
    G = W[W["stage"] != "game 0"]
    groups = [("games 1+", G)] + list(W.groupby("stage")) + \
        [(f"{r} route", g) for r, g in G.groupby(f"route_{BASE}")]
    for label, g in groups:
        y = (g["margin"] > 0).to_numpy(float)
        L.append(f"  {label:14s} n {len(g):5d}  base log loss {dyn.ll(g['p_base'], y).mean():.4f}")
        for arm in others():
            (m, e, _), (bm, be, _), (mm, me, _) = diff(g, arm)
            wins = sum(diff(s, arm)[0][0] < 0 for _, s in g.groupby("season"))
            L.append(f"    {arm:5s} {m:+.4f} ± {e:.4f}  Brier {bm:+.4f} ± {be:.4f}  "
                     f"MSE {mm:+.2f} ± {me:.2f}  better in {wins} of {g['season'].nunique()}")
    L += ["", "  By season, games 1+ (log loss arm - base):"]
    L.append("    season     n  " + "  ".join(f"{a:>17s}" for a in others()))
    for s, g in G.groupby("season"):
        L.append(f"    {s}  {len(g):5d}  " + "  ".join(
            f"{c[0]:+.4f} ± {c[1]:.4f}" for c in (diff(g, a)[0] for a in others())))
    return L


def market_lines(W):
    L = ["", "== 2. Against the close, each basis x book x season alone",
         "   arm - close log loss (± 95%); [arm - base log loss, Brier]"]
    J = W.merge(dyn.market_rows(), on=KEY, how="inner")
    gate = []
    for (basis, book, s), g in J.groupby(["basis", "close_book", "season"]):
        for st, h in [("games 1+", g[g["stage"] != "game 0"])] + list(g.groupby("stage")):
            if len(h) < 30:
                continue
            y = (h["margin"] > 0).to_numpy(float)
            lc = dyn.ll(h["close_q_home"].to_numpy(float), y)
            v = dyn.ci(dyn.ll(h["p_base"], y) - lc)
            L.append(f"  {basis:13s} {book:7s} {s} {st:10s} n {len(h):4d}  "
                     f"base {v[0]:+.4f} ± {v[1]:.4f}")
            for arm in others():
                a = dyn.ci(dyn.ll(h[f"p_{arm}"], y) - lc)
                (dm, de, _), (bm, be, _), _ = diff(h, arm)
                L.append(f"      {arm:5s} {a[0]:+.4f} ± {a[1]:.4f}  "
                         f"[{dm:+.4f} ± {de:.4f}, Brier {bm:+.4f} ± {be:.4f}]")
                if basis == "reconstructed" and st == "games 10+" and \
                        (book, s) in (("dk", 2026), ("espnbet", 2025)):
                    gate.append((arm, book, s, dm, bm))
    return L, gate


def gate_lines(W, gate):
    L = ["", "== 3. Decision rule (G7, set before the run)"]
    G = W[W["stage"] != "game 0"]
    n = G["season"].nunique()
    for arm in others():
        rows = [r for r in gate if r[0] == arm]
        books = len(rows) == 2 and all(dm <= 0 and bm <= 0 for *_, dm, bm in rows)
        (m, e, _), _, _ = diff(G, arm)
        per = [diff(g, arm)[0][0] for _, g in G.groupby("season")]
        wins = sum(x < 0 for x in per)
        ok = books and m + e < 0 and wins >= min(7, n) and -min(per) <= 0.008
        cells = "; ".join(f"{b} {s}: {dm:+.4f} (Brier {bm:+.4f})" for _, b, s, dm, bm in rows)
        L.append(f"  {arm:5s} {'PASS' if ok else 'FAIL'}  priced games 10+ {cells or 'none'}; "
                 f"games 1+ {m:+.4f} ± {e:.4f}, better in {wins} of {n}; "
                 f"largest season gain {-min(per):.4f} (tripwire 0.008)")
    return L


def coef_lines():
    C = []
    for f in sorted(glob.glob(os.path.join(OUT, "coefs_*.json"))):
        C += json.load(open(f))
    C = pd.DataFrame(C)
    L = ["", "== 4. Fitted weights by fold (base route, season-start fit; report route static fit)",
         "   home_w: weight on the home team's rating (P2's w; 0.5 = production);",
         "   off_share: weight on offence vs defence (0.5 = the composite's split)."]
    cols = [c for c in C.columns if c.startswith(("home_w", "off_share"))] + ["c_sum", "delta_off",
                                                                              "o_sum", "mx"]
    for (arm, route), g in C[C["arm"] != BASE].groupby(["arm", "route"], sort=False):
        L.append(f"  {arm} {route}")
        for r in g.itertuples(index=False):
            d = r._asdict()
            L.append(f"    {d['season']}  " + "  ".join(
                f"{c} {d[c]:+.3f}" for c in cols if c in d and pd.notna(d[c])))
    return L


def report():
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(
        os.path.join(OUT, "predictions_*.csv")))], ignore_index=True)
    W = P.dropna(subset=[f"p_{a}" for a in ARMS]).reset_index(drop=True)
    L = [__doc__.strip().splitlines()[0], "",
         f"Games scored by every arm: {len(W)} (seasons {sorted(W['season'].unique())}); "
         f"report route on {(W[f'route_{BASE}'] == 'avail').sum()}. "
         "Each row: arm - base (negative = arm better), ± 95%.", ""]
    L += outcome_lines(W)
    m, gate = market_lines(W)
    L += m + gate_lines(W, gate) + coef_lines()
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
