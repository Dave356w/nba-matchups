#!/usr/bin/env python3
"""v10 candidate: four small changes together, production routing, walk-forward.

  python research/v10_candidate.py --config v9|v10|v10h25 [--seasons 2017 ... 2027]
  python research/v10_candidate.py --report

Owner's request (2026-10-28): run the combined v10 walk-forward backtest of
the changes earlier research left as candidates for the next model version:

  ORB%      the composite's rebounding factors as ORB / (ORB + opp DRB)
            (diagnostics/orb_pct: -0.0003 ± 0.0002, 8 of 10 folds).
  rookies   drafted rookies valued by pick, rookie BPM = a + b log(pick) fitted
            on earlier seasons only, in place of replacement level, wherever
            a player's value is read: talent_diff and av_bpm (history
            backtest A9: -0.0002 ± 0.0002, 9 of 9). Game 0's preseason talent
            gives rookies no minutes role, so it is unchanged.
  hl 20     every decayed input at a 20-game half-life instead of 25: the
            four-factor rating, opponents' 3P luck, own FT%, the offseason
            fade, and the availability terms' rating-window shares
            (diagnostics/half_life_sweep: -0.0002 ± 0.0002, 8 of 9).
  dynamic   the injury-report route refitted before every slate on the
            report seasons before Y plus season Y's covered games so far
            (diagnostics/dynamic_fit RESULTS_avail: games 10+ -0.0015 ±
            0.0010). The base route stays a season-start fit (its dynamic
            arm gained nothing).

Configs (feature builds) and arms, fixed before the run:
  v9      production: ORB per 100, frozen talent, half-life 25.     arm v9
  v10     ORB%, rookies by pick, half-life 20.     arms v10s (static), v10 (dynamic)
  v10h25  as v10 at half-life 25.                  arms v10h25s, v10h25
Every arm uses production routing: a game the NBA report covers, with both
teams past game 0, in a season with an earlier report season (2024 on) is
scored by the injury-report route (player_availability.FEATURES_V6), every
other game by the base route (nba_composite.V6_FEATURES). Each fold Y
refits the composite weights on WEIGHT_YEARS before Y, the base OLS on the
game seasons before Y and the report OLS on the covered games 1+ of the
report seasons before Y (research/dynamic_fit.py's S protocol); P = Phi(m /
sigma). Multi-season BPM, aging and in-season av_bpm are not included:
untested on their own.

Readings, each arm minus v9 on identical games (negative = better), ± 95%,
game-level, paired: log loss, Brier and margin MSE against outcomes, by
stage, route and season; against the close, each basis x book x season
alone. Decision rule, set before the run (the G7 change gate,
docs/model_guidelines.md): v10 is a candidate for a new model tag only if
its log loss is not worse than v9's on either large priced book-season
(reconstructed DraftKings 2025-26 and ESPN BET 2024-25, games 10+), with
Brier agreeing, and not worse against outcomes on games 1+; an improvement
over 0.008 in any season is a leakage tripwire. Passing makes it a
candidate, the owner's decision, not a change. Research only: nothing here
touches the ledgers, the pages or model/*.json.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
import nba_composite as nc                        # noqa: E402
import player_availability as pav                 # noqa: E402
import dynamic_fit as dyn                         # noqa: E402
import history_backtest as hb                     # noqa: E402
import orb_pct                                    # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "v10_candidate")
CACHE = os.path.join(pav.DEFAULT_CACHE, "v10")
CONFIGS = {"v9": dict(orb=False, rookies=False, hl=25.0),
           "v10": dict(orb=True, rookies=True, hl=20.0),
           "v10h25": dict(orb=True, rookies=True, hl=25.0)}
ARMS = {"v9": [("v9", "S")], "v10": [("v10s", "S"), ("v10", "D")],
        "v10h25": [("v10h25s", "S"), ("v10h25", "D")]}
BASE_ARM = "v9"
FEATS, AVAIL = dyn.FEATS, dyn.AVAIL_FEATS
REPORT_YEARS = dyn.REPORT_YEARS                   # 2023-2027
CORE = dyn.CORE


# ------------------------------------------------------------- inputs --
def statuses(t):
    """(report statuses, covered game ids) for season t, cached; the same
    archive and 30-minute lead as player_availability.season_terms."""
    path = os.path.join(CACHE, f"status_{t}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    arch = pav.ReportArchive(pav.REPORT_DIR)
    box = pav.fetch_box(t, cache_dir=pav.DEFAULT_CACHE)
    st, s = pav.game_statuses(box, pav.fetch_tips(t, pav.DEFAULT_CACHE), arch,
                              pav.LEAD_MINUTES)
    arch.save()
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump((st, s["covered"]), open(path, "wb"))
    return st, s["covered"]


_curves = {}


def values(t, rookies):
    """player_id -> value above replacement for season t (last season's BPM),
    with drafted rookies by pick when `rookies` (curve on earlier seasons)."""
    box = pav.fetch_box(t, cache_dir=pav.DEFAULT_CACHE)
    bpm = pav.load_bpm(t - 1)
    value, _, _ = pav.player_values(box, bpm)
    if rookies:
        if t not in _curves:
            _curves[t] = hb.rookie_curve(range(2015, t))
        value, n = hb.rookie_values(value, box, bpm, hb.draft_picks(t - 1), _curves[t])
        dyn.log(f"  rookies {t}: curve {_curves[t][0]:+.2f} {_curves[t][1]:+.2f} log(pick) "
                f"on {_curves[t][2]}; {n} valued")
    return box, bpm, value


_talent, _terms = {}, {}


def talents(t, cfg):
    key = (t, cfg["rookies"])
    if key not in _talent:
        box, bpm, value = values(t, cfg["rookies"])
        _talent[key] = (pav.talent_fn(box, value, pav.arrival_roles(box, bpm)),
                        pav.preseason_talent(t))
    return _talent[key]


def terms(t, cfg):
    """Covered games' av_min / av_bpm (pregame od), at the config's values
    and half-life (player_availability.season_terms, generalised)."""
    key = (t, cfg["rookies"], cfg["hl"])
    if key not in _terms:
        box, _, value = values(t, cfg["rookies"])
        st, covered = statuses(t)
        present = pav.present_map(st, "od")
        feats = {}
        for _, tg in box.groupby("team"):
            for gid, v in pav.team_availability(tg, half_life=cfg["hl"], value=value,
                                                present=present).items():
                feats[(gid, tg["team"].iloc[0])] = v
        g = box[box["home"]].drop_duplicates("game_id")[["game_id", "date", "team", "opp"]]
        rows = []
        for r in g.itertuples(index=False):
            h, a = feats.get((r.game_id, r.team)), feats.get((r.game_id, r.opp))
            if h is None or a is None or r.game_id not in covered:
                continue
            rows.append(dict(slate_date=pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                             home=r.team, away=r.opp, av_min=h[0] - a[0], av_bpm=h[2] - a[2]))
        _terms[key] = pd.DataFrame(rows)
    return _terms[key]


def games(t, wts, cfg):
    tal, tal0 = talents(t, cfg)
    g = nc.fade_v8_games(t, wts, half_life=cfg["hl"], talent=tal, talent0=tal0)
    g["date"] = pd.to_datetime(g["date"])
    g["slate_date"] = g["date"].dt.strftime("%Y-%m-%d")
    return g


def covered(g, t, cfg):
    c = g.merge(terms(t, cfg), on=["slate_date", "home", "away"], how="inner")
    return c[np.minimum(c["gp_home"], c["gp_away"]) >= 1]


# --------------------------------------------------------------- folds --
def fold(Y, cfg):
    """Season Y's games with S (and D on the report route) predictions."""
    wts = dyn.fold_weights(Y)
    G = {t: games(t, wts, cfg) for t in dyn.GAME_YEARS if t <= Y}
    past = pd.concat([G[t] for t in G if t < Y], ignore_index=True)
    past = past[np.isfinite(past[FEATS + ["margin"]].to_numpy(float)).all(axis=1)]
    cur = G[Y][np.isfinite(G[Y][CORE].to_numpy(float)).all(axis=1)].copy()
    b, s = dyn.wls(past)
    m, p = dyn.predict(b, s, cur)
    for arm in ("S", "D"):
        cur[f"m_{arm}"], cur[f"p_{arm}"] = m, p
    cur["route"] = "base"
    rep = [t for t in REPORT_YEARS if t < Y]
    if rep:
        cpast = pd.concat([covered(G[t], t, cfg) for t in rep], ignore_index=True)
        ccur = covered(cur, Y, cfg)
        if len(ccur):
            out, _, _ = dyn.arms(cpast, ccur, AVAIL, Y, "avail")
            key = ["slate_date", "home", "away"]
            out = out.set_index(key)
            idx = cur.set_index(key).index
            hit = idx.isin(out.index)
            o = out.reindex(idx[hit])
            for arm in ("S", "D"):
                cur.loc[hit, f"m_{arm}"] = o[f"m_{arm}"].to_numpy()
                cur.loc[hit, f"p_{arm}"] = o[f"p_{arm}"].to_numpy()
            cur.loc[hit, "route"] = "avail"
    mn = np.fmin(cur["gp_home"], cur["gp_away"])
    cur["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
    cur["season"] = Y
    dyn.log(f"fold {Y}: {len(cur)} games ({(cur['route'] == 'avail').sum()} report route), "
            f"base sigma {s:.2f}")
    return cur[["season", "slate_date", "home", "away", "gp_home", "gp_away", "stage",
                "route", "margin", "m_S", "p_S", "m_D", "p_D"]]


def run(config, seasons):
    cfg = CONFIGS[config]
    orb_pct.use("pct" if cfg["orb"] else "per100")
    dyn.log(f"config {config}: {cfg}")
    P = pd.concat([fold(Y, cfg) for Y in seasons], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    part = f"{min(seasons)}-{max(seasons)}"
    P.to_csv(os.path.join(OUT, f"predictions_{config}_{part}.csv"), index=False,
             float_format="%.6g")


# -------------------------------------------------------------- report --
def load():
    key = ["season", "slate_date", "home", "away"]
    W = None
    for config, arms in ARMS.items():
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(
            os.path.join(OUT, f"predictions_{config}_*.csv")))], ignore_index=True)
        d = d.drop_duplicates(key, keep="last")
        cols = {}
        for arm, src in arms:
            cols[f"m_{src}"], cols[f"p_{src}"] = f"m_{arm}", f"p_{arm}"
        cols["route"] = f"route_{config}"
        d = d.rename(columns=cols)[key + ["stage", "margin"] + list(cols.values())]
        W = d if W is None else W.merge(d.drop(columns=["stage", "margin"]), on=key)
    return W.dropna().reset_index(drop=True)


def arm_names():
    return [a for arms in ARMS.values() for a, _ in arms if a != BASE_ARM]


def diff(g, arm):
    y = (g["margin"] > 0).to_numpy(float)
    d = dyn.ll(g[f"p_{arm}"], y) - dyn.ll(g[f"p_{BASE_ARM}"], y)
    b = (g[f"p_{arm}"] - y) ** 2 - (g[f"p_{BASE_ARM}"] - y) ** 2
    e = (g["margin"] - g[f"m_{arm}"]) ** 2 - (g["margin"] - g[f"m_{BASE_ARM}"]) ** 2
    return dyn.ci(d), dyn.ci(b), dyn.ci(e)


def outcome_lines(W):
    L = ["== 1. Against outcomes: arm - v9 (log loss / Brier / margin MSE, ± 95%)"]
    groups = [("games 1+", W[W["stage"] != "game 0"])] + list(W.groupby("stage")) + \
        [(f"{r} route (v9's)", g[g["stage"] != "game 0"])
         for r, g in W.groupby(f"route_{BASE_ARM}")]
    for label, g in groups:
        y = (g["margin"] > 0).to_numpy(float)
        L.append(f"  {label:20s} n {len(g):5d}  v9 log loss {dyn.ll(g['p_v9'], y).mean():.4f}")
        for arm in arm_names():
            (m, e, _), (bm, be, _), (mm, me, _) = diff(g, arm)
            wins = sum(diff(s, arm)[0][0] < 0 for _, s in g.groupby("season"))
            L.append(f"    {arm:8s} {m:+.4f} ± {e:.4f}  Brier {bm:+.4f} ± {be:.4f}  "
                     f"MSE {mm:+.2f} ± {me:.2f}  better in {wins} of {g['season'].nunique()}")
    L += ["", "  By season, games 1+ (log loss arm - v9):"]
    L.append("    season      n  " + "  ".join(f"{a:>17s}" for a in arm_names()))
    for s, g in W[W["stage"] != "game 0"].groupby("season"):
        cells = [diff(g, a)[0] for a in arm_names()]
        L.append(f"    {s}  {len(g):5d}  " + "  ".join(f"{c[0]:+.4f} ± {c[1]:.4f}" for c in cells))
    return L


def market_lines(W):
    L = ["", "== 2. Against the close, each basis x book x season alone",
         "   arm - close log loss (± 95%); [arm - v9 log loss, Brier]"]
    J = W.merge(dyn.market_rows(), on=["season", "slate_date", "home", "away"], how="inner")
    gate = []
    for (basis, book, s), g in J.groupby(["basis", "close_book", "season"]):
        for st, h in [("games 1+", g[g["stage"] != "game 0"])] + list(g.groupby("stage")):
            if len(h) < 30:
                continue
            y = (h["margin"] > 0).to_numpy(float)
            lc = dyn.ll(h["close_q_home"].to_numpy(float), y)
            v = dyn.ci(dyn.ll(h["p_v9"], y) - lc)
            L.append(f"  {basis:13s} {book:7s} {s} {st:10s} n {len(h):4d}  "
                     f"v9 {v[0]:+.4f} ± {v[1]:.4f}")
            for arm in arm_names():
                a = dyn.ci(dyn.ll(h[f"p_{arm}"], y) - lc)
                (dm, de, _), (bm, be, _), _ = diff(h, arm)
                L.append(f"      {arm:8s} {a[0]:+.4f} ± {a[1]:.4f}  "
                         f"[{dm:+.4f} ± {de:.4f}, Brier {bm:+.4f} ± {be:.4f}]")
                if basis == "reconstructed" and st == "games 10+" and \
                        (book, s) in (("dk", 2026), ("espnbet", 2025)):
                    gate.append((arm, book, s, dm, bm))
    return L, gate


def gate_lines(W, gate):
    L = ["", "== 3. Decision rule (G7, set before the run)"]
    G = W[W["stage"] != "game 0"]
    for arm in arm_names():
        rows = [g for g in gate if g[0] == arm]
        ok_books = all(dm <= 0 and bm <= 0 for _, _, _, dm, bm in rows) and len(rows) == 2
        (m, e, _), _, _ = diff(G, arm)
        worst = min(diff(g, arm)[0][0] for _, g in G.groupby("season"))
        ok = ok_books and m <= 0 and worst > -0.008
        cells = "; ".join(f"{b} {s}: {dm:+.4f} (Brier {bm:+.4f})" for _, b, s, dm, bm in rows)
        L.append(f"  {arm:8s} {'PASS' if ok else 'FAIL'}  priced games 10+ {cells}; outcomes "
                 f"games 1+ {m:+.4f} ± {e:.4f}; largest season gain {-worst:.4f} (tripwire 0.008)")
    return L


def report():
    W = load()
    L = [__doc__.strip().splitlines()[0], "",
         f"Games scored by every arm: {len(W)}; v9's report route on "
         f"{(W[f'route_{BASE_ARM}'] == 'avail').sum()}. Each row: arm - v9 "
         "(negative = arm better), ± 95%.", ""]
    L += outcome_lines(W)
    m, gate = market_lines(W)
    L += m + gate_lines(W, gate)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seasons", nargs="+", type=int, default=dyn.SEASONS)
    a = ap.parse_args()
    if a.config:
        run(a.config, a.seasons)
    if a.report:
        report()


if __name__ == "__main__":
    main()
