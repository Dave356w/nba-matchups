#!/usr/bin/env python3
"""Questionable weighting and lock lead: v10's report route, walk-forward.

  python research/questionable_weight.py --run [--seasons 2024 2025 2026]
  python research/questionable_weight.py --report

Owner's request (2026-10-28): find the best middle ground between an
injury-informed pregame snapshot and early prices, and weight Questionable
players instead of counting them as playing.

Production (v10) reads the latest NBA injury report at the snapshot under
the od rule: Out or Doubtful -> 0, anyone else on the previous box score ->
1. Its report route is fitted on the last report >= 30 min before tip, when
almost nobody is still Questionable, but the first snapshot reads the
overnight edition, about 18 h before tip.

Lock leads L, minutes before each game's tip: 1080, 720, 480, 300, 120, 30.
At each L a game is covered (report route) only if the report published by
tip - L covers it (player_availability.covers); every other game takes the
base route, as production does.

Arms, fixed before the run (v10: ORB%, rookies by pick, half-life 25; the
report route refitted before each slate, dynamic_fit's D):
  od    train od at 30 min (past seasons and this season's earlier games),
        score od at L. Production, locked L minutes before tip.
  q     train od at 30 min, score q at L: Out -> 0; Doubtful, Questionable
        and Probable -> the share of players with that status at L who
        logged minutes, in the report seasons before Y; Available -> 1.
  odL   train od at L, score od at L.
  qL    train q at L, score q at L (rates as in q).
The base route is the same in every arm. Report seasons 2022-23 on; test
seasons 2023-24 to 2025-26 (each needs an earlier report season).

Readings: each arm minus od at the same L on identical games (negative =
better), ± 95%, game-level, paired: log loss, Brier and margin MSE against
outcomes, by stage, route, back-to-back and season; od at each L minus od
at 30 min (what an earlier lock costs the forecast); each against the
close, per basis x book x season. Also: the share of eventual absences of
rotation players (>= 20 min a game) flagged Out/Doubtful at L, and
Questionable play rates by L.

Decision rule, set before the run (G7): an arm is a candidate for a new
model tag only if, at L = 1080 and at L = 300, its log loss is not worse
than od's on both large priced book-seasons (reconstructed DraftKings
2025-26 and ESPN BET 2024-25, games 10+), Brier agreeing, and not worse
against outcomes on games 1+; a season gain over 0.008 is a leakage
tripwire. Passing makes it a candidate, the owner's decision. The lock
lead is a forward-test design choice that needs a registered amendment.
Research only: nothing here touches the ledgers, the pages or model/*.json.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
import player_availability as pav                 # noqa: E402
import dynamic_fit as dyn                         # noqa: E402
import orb_pct                                    # noqa: E402
import v10_candidate as v10c                      # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "questionable_weight")
CACHE = os.path.join(pav.DEFAULT_CACHE, "qweight")
CFG = v10c.CONFIGS["v10h25"]
LEADS = (1080, 720, 480, 300, 120, 30)
TRAIN_LEAD = pav.LEAD_MINUTES                     # 30
GATE_LEADS = (1080, 300)
ARMS = ("od", "q", "odL", "qL")
REPORT_YEARS = (2023, 2024, 2025, 2026)
SEASONS = (2024, 2025, 2026)
AVAIL = dyn.AVAIL_FEATS
ROTATION_MPG = 20.0
KEY = ["slate_date", "home", "away"]


# ------------------------------------------------------------- inputs --
def statuses(t, lead):
    """(report statuses at tip - lead, covered game ids) for season t, cached."""
    path = os.path.join(CACHE, f"status_{t}_{lead}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    arch = pav.ReportArchive(pav.REPORT_DIR)
    box = pav.fetch_box(t, cache_dir=pav.DEFAULT_CACHE)
    st, s = pav.game_statuses(box, pav.fetch_tips(t, pav.DEFAULT_CACHE), arch, lead)
    arch.save()
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump((st, s["covered"]), open(path, "wb"))
    return st, s["covered"]


def rates(Y, lead):
    """status -> share who logged minutes, at `lead`, report seasons before Y."""
    st = pd.concat([statuses(t, lead)[0] for t in REPORT_YEARS if t < Y], ignore_index=True)
    return pav.play_rates(st)


def present(t, lead, rule, Y):
    st, _ = statuses(t, lead)
    return pav.present_map(st, rule, rates(Y, lead) if rule == "q" else None)


_terms = {}


def terms(t, lead, rule, Y, cover_lead):
    """Covered games' av_min / av_bpm at `lead` under `rule` (q rates from
    report seasons before Y); coverage read at `cover_lead`."""
    k = (t, lead, rule, Y if rule == "q" else None, cover_lead)
    if k not in _terms:
        box, _, value = v10c.values(t, CFG["rookies"])
        pres = present(t, lead, rule, Y)
        _, cov = statuses(t, cover_lead)
        feats = {}
        for _, tg in box.groupby("team"):
            for gid, v in pav.team_availability(tg, half_life=CFG["hl"], value=value,
                                                present=pres).items():
                feats[(gid, tg["team"].iloc[0])] = v
        g = box[box["home"]].drop_duplicates("game_id")[["game_id", "date", "team", "opp"]]
        rows = []
        for r in g.itertuples(index=False):
            h, a = feats.get((r.game_id, r.team)), feats.get((r.game_id, r.opp))
            if h is None or a is None or r.game_id not in cov:
                continue
            rows.append(dict(slate_date=pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                             home=r.team, away=r.opp, av_min=h[0] - a[0], av_bpm=h[2] - a[2]))
        _terms[k] = pd.DataFrame(rows, columns=KEY + ["av_min", "av_bpm"])
    return _terms[k]


def covered(g, tm):
    c = g.merge(tm, on=KEY, how="inner")
    return c[np.minimum(c["gp_home"], c["gp_away"]) >= 1]


def refit_predict(past, train_cur, score_cur):
    """D: before each slate, OLS on past + this season's earlier train rows;
    predict that slate's score rows (dynamic_fit.arms, train and score split)."""
    past = past[np.isfinite(past[AVAIL + ["margin"]].to_numpy(float)).all(axis=1)]
    ok = np.isfinite(train_cur[AVAIL + ["margin"]].to_numpy(float)).all(axis=1)
    train_cur = train_cur[ok]
    score_cur = score_cur[np.isfinite(score_cur[AVAIL].to_numpy(float)).all(axis=1)].copy()
    b0, s0 = dyn.wls(past, feats=AVAIL)
    for d, idx in score_cur.groupby("date").groups.items():
        seen = train_cur[train_cur["date"] < d]
        b, s = dyn.wls(pd.concat([past, seen], ignore_index=True), feats=AVAIL) \
            if len(seen) else (b0, s0)
        score_cur.loc[idx, "m"], score_cur.loc[idx, "p"] = dyn.predict(b, s, score_cur.loc[idx], AVAIL)
    return score_cur.set_index(KEY)[["m", "p"]]


def arm_inputs(arm, lead):
    """(train lead, train rule, score rule) for an arm at lock lead `lead`."""
    return {"od": (TRAIN_LEAD, "od", "od"), "q": (TRAIN_LEAD, "od", "q"),
            "odL": (lead, "od", "od"), "qL": (lead, "q", "q")}[arm]


# --------------------------------------------------------------- folds --
def fold(Y):
    wts = dyn.fold_weights(Y)
    G = {t: v10c.games(t, wts, CFG) for t in dyn.GAME_YEARS if t <= Y}
    past = pd.concat([G[t] for t in G if t < Y], ignore_index=True)
    past = past[np.isfinite(past[dyn.FEATS + ["margin"]].to_numpy(float)).all(axis=1)]
    cur = G[Y][np.isfinite(G[Y][dyn.CORE].to_numpy(float)).all(axis=1)].copy()
    b, s = dyn.wls(past)
    cur["m_base"], cur["p_base"] = dyn.predict(b, s, cur)
    rep = [t for t in REPORT_YEARS if t < Y]
    out = []
    for lead in LEADS:
        c = cur.copy()
        _, cov = statuses(Y, lead)
        for arm in ARMS:
            tl, trule, srule = arm_inputs(arm, lead)
            cpast = pd.concat([covered(G[t], terms(t, tl, trule, Y, tl)) for t in rep],
                              ignore_index=True)
            train_cur = covered(cur, terms(Y, tl, trule, Y, tl))
            score_cur = covered(cur, terms(Y, lead, srule, Y, lead))
            pr = refit_predict(cpast, train_cur, score_cur)
            idx = c.set_index(KEY).index
            hit = idx.isin(pr.index)
            c[f"m_{arm}"], c[f"p_{arm}"] = c["m_base"], c["p_base"]
            o = pr.reindex(idx[hit])
            c.loc[hit, f"m_{arm}"] = o["m"].to_numpy()
            c.loc[hit, f"p_{arm}"] = o["p"].to_numpy()
            if arm == "od":
                c["route"] = np.where(hit, "avail", "base")
        c["lead"] = lead
        out.append(c)
        dyn.log(f"fold {Y} lead {lead}: {len(c)} games, report route {(c['route'] == 'avail').sum()}")
    P = pd.concat(out, ignore_index=True)
    mn = np.fmin(P["gp_home"], P["gp_away"])
    P["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
    P["season"] = Y
    P["b2b"] = (P["h_b2b"] + P["a_b2b"]) > 0
    cols = ["season", "lead", "slate_date", "home", "away", "gp_home", "gp_away", "stage",
            "route", "b2b", "margin"] + [f"{x}_{a}" for a in ARMS for x in ("m", "p")]
    return P[cols]


def run(seasons):
    orb_pct.use("pct")
    os.makedirs(OUT, exist_ok=True)
    for Y in seasons:
        fold(Y).to_csv(os.path.join(OUT, f"predictions_{Y}.csv"), index=False,
                       float_format="%.6g")


# -------------------------------------------------------------- report --
def load():
    import glob
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "predictions_*.csv")))],
                     ignore_index=True)


def paired(g, a, b):
    y = (g["margin"] > 0).to_numpy(float)
    return (dyn.ci(dyn.ll(g[f"p_{a}"], y) - dyn.ll(g[f"p_{b}"], y)),
            dyn.ci((g[f"p_{a}"] - y) ** 2 - (g[f"p_{b}"] - y) ** 2),
            dyn.ci((g["margin"] - g[f"m_{a}"]) ** 2 - (g["margin"] - g[f"m_{b}"]) ** 2))


def fmt(c):
    (m, e, _), (bm, be, _), (mm, me, _) = c
    return f"{m:+.4f} ± {e:.4f}  Brier {bm:+.4f} ± {be:.4f}  MSE {mm:+.2f} ± {me:.2f}"


def status_lines():
    L = ["== 1. What the report knows at each lead (rotation players, >= 20 min a game)",
         "   flagged = sat and listed Out/Doubtful at the lead, of those who sat and were listed"
         " at 30 min; Q = listed Questionable at the lead, share who played"]
    for t in REPORT_YEARS:
        box = pav.fetch_box(t, cache_dir=pav.DEFAULT_CACHE)
        mpg = box[box["minutes"] > 0].groupby("player_id")["minutes"].mean()
        fin, _ = statuses(t, TRAIN_LEAD)
        fin = fin[fin["player_id"].map(mpg).fillna(0) >= ROTATION_MPG]
        sat = fin[~fin["played"]][["game_id", "player_id"]]
        L.append(f"  {t - 1}-{str(t)[2:]}  sat and listed at 30 min: {len(sat)}")
        for lead in LEADS:
            st, cov = statuses(t, lead)
            m = sat.merge(st[["game_id", "player_id", "status"]], on=["game_id", "player_id"],
                          how="left")
            flag = m["status"].isin(("Out", "Doubtful")).mean()
            r = st[st["player_id"].map(mpg).fillna(0) >= ROTATION_MPG]
            q = r[r["status"] == "Questionable"]
            L.append(f"    {lead / 60:5.1f} h  covered games {len(cov):5d}  flagged {flag:5.1%}  "
                     f"Q n {len(q):5d} played {q['played'].mean():5.1%}  "
                     f"rates (all listed) " + ", ".join(f"{k} {v:.0%}" for k, v in
                                                        sorted(pav.play_rates(st).items())))
    return L


def outcome_lines(P):
    L = ["", "== 2. Against outcomes, games 1+: each arm minus od at the same lead (± 95%)"]
    G = P[P["stage"] != "game 0"]
    for lead, g in G.groupby("lead", sort=False):
        y = (g["margin"] > 0).to_numpy(float)
        L.append(f"  lead {lead / 60:4.1f} h  n {len(g)}  report route {(g['route'] == 'avail').sum()}"
                 f"  od log loss {dyn.ll(g['p_od'], y).mean():.4f}")
        for arm in ARMS[1:]:
            wins = sum(paired(s, arm, "od")[0][0] < 0 for _, s in g.groupby("season"))
            L.append(f"    {arm:4s} {fmt(paired(g, arm, 'od'))}  better in {wins} of {g['season'].nunique()}")
        for lab, h in (("report route", g[g["route"] == "avail"]), ("back-to-back", g[g["b2b"]]),
                       ("not back-to-back", g[~g["b2b"]])):
            if len(h) > 30:
                L.append(f"      {lab:17s} n {len(h):5d}  " + "  ".join(
                    f"{a} {paired(h, a, 'od')[0][0]:+.4f} ± {paired(h, a, 'od')[0][1]:.4f}" for a in ARMS[1:]))
    L += ["", "== 3. Earlier lock: od and q at each lead minus od at 30 min, identical games, games 1+"]
    key = ["season"] + KEY
    base = G[G["lead"] == TRAIN_LEAD].set_index(key)
    for lead, g in G.groupby("lead", sort=False):
        j = g.set_index(key).join(base[["p_od", "m_od"]], rsuffix="_30", how="inner").reset_index()
        j["p_ref"], j["m_ref"] = j["p_od_30"], j["m_od_30"]
        for arm in ("od", "q"):
            L.append(f"  lead {lead / 60:4.1f} h  {arm:3s} - od@30m  {fmt(paired(j, arm, 'ref'))}"
                     f"  (back-to-back {paired(j[j['b2b']], arm, 'ref')[0][0]:+.4f},"
                     f" other {paired(j[~j['b2b']], arm, 'ref')[0][0]:+.4f})")
    return L


def market_lines(P):
    L = ["", "== 4. Against the close: arm - close log loss (± 95%); [arm - od, Brier]"]
    J = P.merge(dyn.market_rows(), on=["season"] + KEY, how="inner")
    gate = []
    for (basis, book, s), g in J.groupby(["basis", "close_book", "season"]):
        for lead, h in g[g["stage"] == "games 10+"].groupby("lead", sort=False):
            if len(h) < 100:
                continue
            y = (h["margin"] > 0).to_numpy(float)
            lc = dyn.ll(h["close_q_home"].to_numpy(float), y)
            v = dyn.ci(dyn.ll(h["p_od"], y) - lc)
            L.append(f"  {basis:13s} {book:7s} {s} games 10+ lead {lead / 60:4.1f} h n {len(h):4d}"
                     f"  od {v[0]:+.4f} ± {v[1]:.4f}")
            for arm in ARMS[1:]:
                a = dyn.ci(dyn.ll(h[f"p_{arm}"], y) - lc)
                (dm, de, _), (bm, be, _), _ = paired(h, arm, "od")
                L.append(f"      {arm:4s} {a[0]:+.4f} ± {a[1]:.4f}  [{dm:+.4f} ± {de:.4f}, "
                         f"Brier {bm:+.4f} ± {be:.4f}]")
                if basis == "reconstructed" and (book, s) in (("dk", 2026), ("espnbet", 2025)) \
                        and lead in GATE_LEADS:
                    gate.append((arm, lead, book, s, dm, bm))
    return L, gate


def gate_lines(P, gate):
    L = ["", "== 5. Decision rule (G7, set before the run), at leads " +
         ", ".join(f"{x / 60:.0f} h" for x in GATE_LEADS)]
    G = P[P["stage"] != "game 0"]
    for arm in ARMS[1:]:
        ok, cells = True, []
        for lead in GATE_LEADS:
            rows = [x for x in gate if x[0] == arm and x[1] == lead]
            g = G[G["lead"] == lead]
            (m, e, _), _, _ = paired(g, arm, "od")
            worst = min(paired(s, arm, "od")[0][0] for _, s in g.groupby("season"))
            ok &= len(rows) == 2 and all(dm <= 0 and bm <= 0 for *_, dm, bm in rows) \
                and m <= 0 and worst > -0.008
            cells.append(f"{lead / 60:.0f} h: " + "; ".join(f"{b} {s} {dm:+.4f} (Brier {bm:+.4f})"
                                                           for _, _, b, s, dm, bm in rows)
                         + f"; games 1+ {m:+.4f} ± {e:.4f}")
        L.append(f"  {arm:4s} {'PASS' if ok else 'FAIL'}  " + " | ".join(cells))
    return L


def report():
    P = load()
    L = [__doc__.strip().splitlines()[0], ""]
    L += status_lines() + outcome_lines(P)
    m, gate = market_lines(P)
    L += m + gate_lines(P, gate)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seasons", nargs="+", type=int, default=list(SEASONS))
    a = ap.parse_args()
    if a.run:
        run(a.seasons)
    if a.report or a.run:
        report()


if __name__ == "__main__":
    main()
