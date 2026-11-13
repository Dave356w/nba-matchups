#!/usr/bin/env python3
"""Report-name suffix fix: v11's report route, walk-forward (G7 gate).

  python research/report_name_suffix.py --run [--seasons 2024 2025 2026]
  python research/report_name_suffix.py --report

CLAUDE.md, report-name suffix fix (registered 2026-11-13). Since 2023-24 the
NBA injury report prints suffixes joined to the last name ('MurphyIII,Trey'),
which the report matcher missed: such players, listed Out, counted as playing,
in v11's training terms, its Questionable play rates and its live scoring.

Arms, fixed before the run, on research/questionable_weight.py's machinery
(v10 base: ORB%, rookies by pick, half-life 25; the report route refitted
before each slate on earlier report seasons plus this season's earlier games):
  q     production v11: train od at 30 min, score q at the lead, play rates
        from report seasons before Y; the existing matcher.
  qS    the same with the suffix fix in every report read (training terms,
        play rates, scoring) -- the registered fix.
The base route is the same in both arms; coverage is identical (it reads
matchups, not names). Report seasons 2022-23 on; test seasons 2023-24 ..
2025-26; leads 18 h and 5 h (v11's gate leads; 5 h is the lock).

Readings: qS minus q on identical games (negative = better), ± 95%, paired,
log loss, Brier and margin MSE against outcomes, games 1+, by season and
route; against the close per basis x book x season, games 10+; how many
listed players each matcher finds.

Gate (G7, as for v11): qS passes if at both leads its log loss is not worse
than q's on both large priced book-seasons (reconstructed DraftKings 2025-26
and ESPN BET 2024-25, games 10+), Brier agreeing, and not worse against
outcomes on games 1+; a season gain over 0.008 is a leakage tripwire.

ou3 (the fix's ou3 amendment, same date): the totals model's report stage,
walk-forward on 2023-24 .. 2025-26, each fold fitted on the earlier report
seasons (from 2022-23) on the deployed T*'s residual at the 300-minute lead;
arms ou3 (existing matcher, q_rates.json's table) and ou3S (the fix and its
table). Squared error on the total, games 1+, ou3S minus ou3 (± 95%), by
season and stage; information for the owner, no pass rule. T* is the
deployed fit (all seasons) and both tables span the report seasons: the same
in both arms, so the difference isolates the matcher.

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
import questionable_weight as qw                  # noqa: E402
import v10_candidate as v10c                      # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "report_name_suffix")
CACHE = os.path.join(pav.DEFAULT_CACHE, "report_name_suffix")
LEADS = (1080, 300)
ARMS = ("q", "qS")
SEASONS = (2024, 2025, 2026)
KEY = qw.KEY
_terms, _stats = {}, {}


def statuses(t, lead, split):
    """(report statuses at tip - lead, covered game ids, matching stats)."""
    k = (t, lead, split)
    if k in _stats:
        return _stats[k]
    path = os.path.join(CACHE, f"status_{t}_{lead}_{'sfx' if split else 'old'}.pkl")
    if os.path.exists(path):
        _stats[k] = pickle.load(open(path, "rb"))
        return _stats[k]
    arch = pav.ReportArchive(pav.REPORT_DIR)
    box = pav.fetch_box(t, cache_dir=pav.DEFAULT_CACHE)
    st, s = pav.game_statuses(box, pav.fetch_tips(t, pav.DEFAULT_CACHE), arch, lead,
                              split=split)
    arch.save()
    out = (st, s["covered"], dict(listed=s["listed"], matched=s["matched"]))
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    _stats[k] = out
    return out


def rates(Y, lead, split):
    st = pd.concat([statuses(t, lead, split)[0] for t in qw.REPORT_YEARS if t < Y],
                   ignore_index=True)
    return pav.play_rates(st)


def terms(t, lead, rule, Y, split):
    """Covered games' av_min / av_bpm at `lead` under `rule` (q rates from
    report seasons before Y), with or without the suffix fix."""
    k = (t, lead, rule, Y if rule == "q" else None, split)
    if k not in _terms:
        box, _, value = v10c.values(t, qw.CFG["rookies"])
        st, cov, _ = statuses(t, lead, split)
        pres = pav.present_map(st, rule, rates(Y, lead, split) if rule == "q" else None)
        feats = {}
        for _, tg in box.groupby("team"):
            for gid, v in pav.team_availability(tg, half_life=qw.CFG["hl"], value=value,
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


def fold(Y):
    wts = dyn.fold_weights(Y)
    G = {t: v10c.games(t, wts, qw.CFG) for t in dyn.GAME_YEARS if t <= Y}
    past = pd.concat([G[t] for t in G if t < Y], ignore_index=True)
    past = past[np.isfinite(past[dyn.FEATS + ["margin"]].to_numpy(float)).all(axis=1)]
    cur = G[Y][np.isfinite(G[Y][dyn.CORE].to_numpy(float)).all(axis=1)].copy()
    b, s = dyn.wls(past)
    cur["m_base"], cur["p_base"] = dyn.predict(b, s, cur)
    rep = [t for t in qw.REPORT_YEARS if t < Y]
    out = []
    for lead in LEADS:
        c = cur.copy()
        for arm in ARMS:
            split = arm == "qS"
            cpast = pd.concat([qw.covered(G[t], terms(t, qw.TRAIN_LEAD, "od", Y, split))
                               for t in rep], ignore_index=True)
            train_cur = qw.covered(cur, terms(Y, qw.TRAIN_LEAD, "od", Y, split))
            score_cur = qw.covered(cur, terms(Y, lead, "q", Y, split))
            pr = qw.refit_predict(cpast, train_cur, score_cur)
            idx = c.set_index(KEY).index
            hit = idx.isin(pr.index)
            c[f"m_{arm}"], c[f"p_{arm}"] = c["m_base"], c["p_base"]
            o = pr.reindex(idx[hit])
            c.loc[hit, f"m_{arm}"] = o["m"].to_numpy()
            c.loc[hit, f"p_{arm}"] = o["p"].to_numpy()
            if arm == "q":
                c["route"] = np.where(hit, "avail", "base")
        c["lead"] = lead
        out.append(c)
        dyn.log(f"fold {Y} lead {lead}: {len(c)} games, report route {(c['route'] == 'avail').sum()}")
    P = pd.concat(out, ignore_index=True)
    mn = np.fmin(P["gp_home"], P["gp_away"])
    P["stage"] = np.where(mn == 0, "game 0", np.where(mn < 10, "games 1-9", "games 10+"))
    P["season"] = Y
    cols = ["season", "lead", "slate_date", "home", "away", "gp_home", "gp_away", "stage",
            "route", "margin"] + [f"{x}_{a}" for a in ARMS for x in ("m", "p")]
    return P[cols]


def ou3_run(seasons):
    """Walk-forward report stage for ou3 (old vs fixed matcher)."""
    import json
    import nba_composite as nc
    import totals
    model = totals.load()
    base = {k: v for k, v in model.items() if k != "report"}
    rows = totals.training_rows({y: nc.load_logs(y) for y in totals._years(totals.FIT_YEARS)})
    rows = rows[rows["season"].isin(qw.REPORT_YEARS)].copy()
    rows["T0"] = [totals.predict(base, f)[0] for f in rows.to_dict("records")]
    tables = {False: json.load(open(pav.Q_RATES_FILE))["rates"],
              True: json.load(open(pav.Q_RATES_SFX_FILE))["rates"]}
    out = []
    for split in (False, True):
        arm = "ou3S" if split else "ou3"
        av = {}
        for t in qw.REPORT_YEARS:
            av.update(pav.season_totals_terms(t, totals.REPORT_LEAD, tables[split], split=split))
        R = rows.copy()
        for k in totals.REPORT_TERMS:
            R[k] = [av.get((d, h, w), {}).get(k, 0.0)
                    for d, h, w in zip(R["date"], R["home"], R["away"])]
        for Y in seasons:
            fm = totals.fit_report(base, R[R["season"] < Y])
            c = fm["report"]["coef"]
            te = R[(R["season"] == Y) & (np.fmin(R["gp_home"], R["gp_away"]) >= 1)].copy()
            te[f"T_{arm}"] = te["T0"] + c["intercept"] + sum(c[k] * te[k] for k in totals.REPORT_TERMS)
            out.append(te[["season", "date", "home", "away", "gp_home", "gp_away", "total",
                           f"T_{arm}"]])
    a = pd.concat([o for o in out if "T_ou3" in o], ignore_index=True)
    b = pd.concat([o for o in out if "T_ou3S" in o], ignore_index=True)
    P = a.merge(b[["season", "date", "home", "away", "T_ou3S"]],
                on=["season", "date", "home", "away"])
    P.to_csv(os.path.join(OUT, "predictions_ou3.csv"), index=False, float_format="%.6g")


def ou3_lines():
    path = os.path.join(OUT, "predictions_ou3.csv")
    if not os.path.exists(path):
        return []
    P = pd.read_csv(path)
    L = ["", "== 5. ou3 report stage, walk-forward: squared error on the total, ou3S minus ou3 "
             "(points², ± 95%; negative = better)"]

    def d(g):
        return dyn.ci((g["total"] - g["T_ou3S"]) ** 2 - (g["total"] - g["T_ou3"]) ** 2)
    mn = np.fmin(P["gp_home"], P["gp_away"])
    P["stage"] = np.where(mn < 10, "games 1-9", "games 10+")
    m = d(P)
    moved = (P["T_ou3S"] - P["T_ou3"]).abs()
    L.append(f"  games 1+  n {len(P)}  {m[0]:+.3f} ± {m[1]:.3f}  "
             f"|dT| > 0.5 pt: {(moved > 0.5).sum()}, max {moved.max():.2f}")
    for s, g in P.groupby("season"):
        x = d(g)
        L.append(f"    {s - 1}-{str(s)[2:]}  n {len(g)}  {x[0]:+.3f} ± {x[1]:.3f}")
    for st, g in P.groupby("stage"):
        x = d(g)
        L.append(f"    {st:9s} n {len(g)}  {x[0]:+.3f} ± {x[1]:.3f}")
    return L


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


def paired(g):
    y = (g["margin"] > 0).to_numpy(float)
    return (dyn.ci(dyn.ll(g["p_qS"], y) - dyn.ll(g["p_q"], y)),
            dyn.ci((g["p_qS"] - y) ** 2 - (g["p_q"] - y) ** 2),
            dyn.ci((g["margin"] - g["m_qS"]) ** 2 - (g["margin"] - g["m_q"]) ** 2))


def match_lines():
    L = ["== 1. Listed players matched to a box score (report rows for the game's teams)"]
    for t in qw.REPORT_YEARS:
        for lead in (pav.LEAD_MINUTES,) + LEADS:
            a, b = statuses(t, lead, False)[2], statuses(t, lead, True)[2]
            L.append(f"  {t - 1}-{str(t)[2:]} lead {lead / 60:4.1f} h  listed {a['listed']:5d}  "
                     f"matched old {a['matched']:5d}  fixed {b['matched']:5d}  "
                     f"(+{b['matched'] - a['matched']})")
    L.append("  play rates (report seasons 2022-23 .. 2025-26):")
    for lead in (pav.LEAD_MINUTES,) + LEADS:
        old, new = rates(9999, lead, False), rates(9999, lead, True)
        L.append(f"    {lead / 60:4.1f} h  " + "  ".join(
            f"{k} {old.get(k, np.nan):.3f}->{new.get(k, np.nan):.3f}"
            for k in ("Out", "Doubtful", "Questionable", "Probable", "Available")))
    return L


def fmt(c):
    (m, e, _), (bm, be, _), (mm, me, _) = c
    return f"{m:+.4f} ± {e:.4f}  Brier {bm:+.4f} ± {be:.4f}  MSE {mm:+.2f} ± {me:.2f}"


def outcome_lines(P):
    L = ["", "== 2. Against outcomes, games 1+: qS minus q (± 95%; negative = better)"]
    G = P[P["stage"] != "game 0"]
    for lead, g in G.groupby("lead", sort=False):
        moved = (g["p_qS"] - g["p_q"]).abs()
        L.append(f"  lead {lead / 60:4.1f} h  n {len(g)}  report route {(g['route'] == 'avail').sum()}"
                 f"  q log loss {dyn.ll(g['p_q'], (g['margin'] > 0).to_numpy(float)).mean():.4f}"
                 f"  |dP| > 0.01: {(moved > 0.01).sum()}, max {moved.max():.3f}")
        L.append(f"    all        {fmt(paired(g))}")
        for s, h in g.groupby("season"):
            L.append(f"    {s - 1}-{str(s)[2:]}    {fmt(paired(h))}")
        for st, h in g.groupby("stage"):
            L.append(f"    {st:10s} {fmt(paired(h))}")
    return L


def market_lines(P):
    L = ["", "== 3. Against the close, games 10+: arm - close log loss (± 95%); [qS - q, Brier]"]
    J = P.merge(dyn.market_rows(), on=["season"] + KEY, how="inner")
    gate = []
    for (basis, book, s), g in J.groupby(["basis", "close_book", "season"]):
        for lead, h in g[g["stage"] == "games 10+"].groupby("lead", sort=False):
            if len(h) < 100:
                continue
            y = (h["margin"] > 0).to_numpy(float)
            lc = dyn.ll(h["close_q_home"].to_numpy(float), y)
            a, b = dyn.ci(dyn.ll(h["p_q"], y) - lc), dyn.ci(dyn.ll(h["p_qS"], y) - lc)
            (dm, de, _), (bm, be, _), _ = paired(h)
            L.append(f"  {basis:13s} {book:7s} {s} lead {lead / 60:4.1f} h n {len(h):4d}  "
                     f"q {a[0]:+.4f} ± {a[1]:.4f}  qS {b[0]:+.4f} ± {b[1]:.4f}  "
                     f"[{dm:+.4f} ± {de:.4f}, Brier {bm:+.4f} ± {be:.4f}]")
            if basis == "reconstructed" and (book, s) in (("dk", 2026), ("espnbet", 2025)):
                gate.append((lead, book, s, dm, bm))
    return L, gate


def gate_lines(P, gate):
    L = ["", "== 4. Gate (G7, set before the run), leads " +
         ", ".join(f"{x / 60:.0f} h" for x in LEADS)]
    G = P[P["stage"] != "game 0"]
    ok, cells = True, []
    for lead in LEADS:
        rows = [x for x in gate if x[0] == lead]
        g = G[G["lead"] == lead]
        (m, e, _), _, _ = paired(g)
        best = min(paired(s)[0][0] for _, s in g.groupby("season"))
        ok &= len(rows) == 2 and all(dm <= 0 and bm <= 0 for *_, dm, bm in rows) \
            and m <= 0 and best > -0.008
        cells.append(f"{lead / 60:.0f} h: " + "; ".join(f"{b} {s} {dm:+.4f} (Brier {bm:+.4f})"
                                                       for _, b, s, dm, bm in rows)
                     + f"; games 1+ {m:+.4f} ± {e:.4f}")
    L.append(f"  qS {'PASS' if ok else 'FAIL'}  " + " | ".join(cells))
    return L


def report():
    P = load()
    L = [__doc__.strip().splitlines()[0], ""]
    L += match_lines() + outcome_lines(P)
    m, gate = market_lines(P)
    L += m + gate_lines(P, gate) + ou3_lines()
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--ou3", action="store_true", help="also run the ou3 report-stage walk-forward")
    ap.add_argument("--seasons", nargs="+", type=int, default=list(SEASONS))
    a = ap.parse_args()
    if a.run:
        run(a.seasons)
    if a.ou3:
        os.makedirs(OUT, exist_ok=True)
        ou3_run(a.seasons)
    if a.report or a.run or a.ou3:
        report()


if __name__ == "__main__":
    main()
