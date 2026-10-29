#!/usr/bin/env python3
"""v11 review (owner's questions, 2026-10-29): pace, totals, and the picks.

  python research/v11_review.py [--out diagnostics/v11_review/RESULTS.txt]

1. Pace. v11 reads per-possession rates (the four factors) and fits the
   final margin, with no pace term: a given rating gap is worth the same
   margin in a fast and a slow matchup. Here, on v11's walk-forward
   predictions (diagnostics/questionable_weight, arm q = v11, every fit on
   earlier seasons; 2017-23 from diagnostics/v10_candidate arm v10h25, which
   v11 equals before the report seasons), a second stage scales the
   predicted margin by the matchup's expected pace, lp = log(mean of both
   teams' decayed possessions / league possessions to date), games strictly
   before the slate (research/walk_forward.py):
     m' = m (1 + k lp)              pace scales the margin
     sigma' = sigma exp(s lp)       and the spread of outcomes
   k and s are fitted on earlier seasons only, then scored on 2023-24,
   2024-25 and 2025-26 at the 5 h lock lead (and 18 h), against outcomes
   and, on the priced games, against the close. G7's yardstick: log loss
   must not worsen on either large priced book-season, Brier agreeing.

2. Totals. v11 predicts only the home margin, so it has no total and no
   O/U probability. The same pregame box-score inputs can give one: a
   walk-forward total from both teams' decayed points per possession
   (offence and defence) and expected possessions. Against the books'
   opening and closing totals (research/totals_fetch.py, ESPN feed; main
   lines only, both prices within -135..+115):
   the outcome's weight on (model total - line), and flat 1u O/U on the
   model's side at the open and the close, beside the no-vig null.
   Hindsight on two book-seasons; information only.

3. The picks against v11's strengths. The reconstructed rows read the
   30-minute injury report, which did not exist at the open, so their
   opening-price hindsight mixes timing. Here the three tests are re-run
   with the walk-forward v11 P at 18 h (about when lines open) at the
   open, and at 5 h (the lock) and 30 min at the close, per book-season
   and stage, beside the disagreement weight w (outcome on logit P -
   logit q beside logit q) and where it sits: value favourites vs value
   underdogs, and by the size of the gap.

Research intervals are 95% unless labelled. Nothing here changes the model,
a rule or a page.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import analysis  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
from model_review import fit, logit, logloss, mean_ci, weight  # noqa: E402

Z95 = 1.96
LEADS = {1080: "18 h", 300: "5 h", 30: "30 min"}
QW = ROOT / "diagnostics" / "questionable_weight"
V10 = ROOT / "diagnostics" / "v10_candidate"
OUT = ROOT / "diagnostics" / "v11_review" / "RESULTS.txt"


def season_name(y):
    return f"{int(y) - 1}-{str(int(y))[-2:]}"


def pm(x, d=4):
    m, h = x
    return f"{m:+.{d}f} ± {h:.{d}f}" if np.isfinite(m) else "--"


# ------------------------------------------------------------ predictions --
def v11_predictions():
    """Walk-forward v11 (arm q) by lead, 2023-24 .. 2025-26: one row per game
    and lead with the predicted margin m, P and the final margin."""
    q = pd.concat([pd.read_csv(QW / f"predictions_{s}.csv") for s in (2024, 2025, 2026)])
    return q.rename(columns={"m_q": "m", "p_q": "p"})[
        ["season", "lead", "slate_date", "home", "away", "gp_home", "gp_away",
         "stage", "route", "b2b", "margin", "m", "p"]]


def early_predictions():
    """Walk-forward v10h25 2016-17 .. 2022-23 (= v11 before the report
    seasons; production routing, dynamic refit): training seasons only."""
    d = pd.read_csv(V10 / "predictions_v10h25_2017-2023.csv")
    return d.rename(columns={"m_D": "m", "p_D": "p"})[
        ["season", "slate_date", "home", "away", "gp_home", "gp_away", "stage",
         "route", "margin", "m", "p"]]


def priced(lead_p):
    """Reconstructed priced rows with p_home replaced by the walk-forward v11
    P at each lead: {lead: frame}. The rows' own p_home (leave-one-season-
    out, 30-minute report) is kept as p_recon."""
    r = ledger.load(ledger.RECON_PATH)
    r["slate_date"] = r["slate_date"].astype(str)
    k = ["season", "slate_date", "home", "away"]
    p = lead_p.pivot_table(index=k, columns="lead", values="p").reset_index()
    m = r.merge(p, on=k, how="inner")
    out = {}
    for lead in LEADS:
        g = m.drop(columns=list(LEADS)).copy()
        g["p_recon"] = g["p_home"]
        g["p_home"] = m[lead].to_numpy()
        out[lead] = g
    return out


def cells(g):
    """(label, rows) per book-season and stage; never pooled."""
    opener, early = analysis.stage(g)
    late = ~opener & ~early
    book = g["close_book"].astype(str)
    season = pd.to_numeric(g["season"], errors="coerce")
    out = []
    for bk, y, st, mask in (("dk", 2026, "games 10+", late),
                            ("espnbet", 2025, "games 10+", late),
                            ("espnbet", 2026, "games 10+", late),
                            ("espnbet", 2025, "games 1–9", early),
                            ("espnbet", 2026, "games 1–9", early)):
        sel = (book == bk) & (season == y) & mask
        out.append((f"{market.BOOK_NAMES[bk]} {season_name(y)} {st}", bk, g[sel]))
    return out


# -------------------------------------------------------------- 3. picks --
def _qp(g, price):
    q = analysis.price_q_home(g, price).to_numpy(float)
    return q


def scores_block(frames):
    lines = ["== 3a. v11 by lead against the open and the close (log loss model − price, ± 95%; "
             "w = outcome weight on logit P − logit q)"]
    for lead, g in frames.items():
        lines.append(f"  P at {LEADS[lead]}")
        for label, _bk, x in cells(g):
            y = pd.to_numeric(x["home_won"], errors="coerce").to_numpy(float)
            p = x["p_home"].to_numpy(float)
            parts = []
            for price in ("open", "close"):
                q = _qp(x, price)
                ok = np.isfinite(q) & np.isfinite(p) & np.isfinite(y) & (q > 0) & (q < 1)
                if ok.sum() < 30:
                    parts.append(f"{price}: n {ok.sum()}")
                    continue
                d = logloss(p[ok], y[ok]) - logloss(q[ok], y[ok])
                parts.append(f"{price} n {ok.sum():>4}  LL {pm(mean_ci(d))}  "
                             f"w {pm(weight(p[ok], q[ok], y[ok]), 2)}")
            lines.append(f"    {label:<34} " + " | ".join(parts))
    return lines


def _value_open(g):
    """Value-side bets at the open: side, edge, favourite flag, units,
    null, CLV (no-vig pp toward the bet, open -> close)."""
    qo, qc = _qp(g, "open"), _qp(g, "close")
    p = g["p_home"].to_numpy(float)
    y = pd.to_numeric(g["home_won"], errors="coerce").to_numpy(float)
    hml, aml = (pd.to_numeric(g[f"open_{s}_ml"], errors="coerce").to_numpy(float)
                for s in ("home", "away"))
    ok = np.isfinite(qo) & np.isfinite(p) & np.isin(y, [0, 1]) & np.isfinite(hml) \
        & np.isfinite(aml) & (p != qo)
    home = p > qo
    side_q = np.where(home, qo, 1 - qo)
    ml = np.where(home, hml, aml)
    won = np.where(home, y, 1 - y)
    d = pd.DataFrame(dict(edge=np.where(home, p - qo, qo - p), q=side_q, ml=ml, won=won,
                          clv=100 * (np.where(home, qc, 1 - qc) - side_q)))[ok]
    d["fav"] = d["q"] > 0.5
    d["units"] = [market.unit_profit(m, w) for m, w in zip(d["ml"], d["won"])]
    d["null"] = d["q"] * np.array([market.decimal_payout(m) for m in d["ml"]]) - 1
    return d


def _roi(d):
    if len(d) < 2:
        return f"n {len(d):>4}"
    u = d["units"].to_numpy(float)
    roi, h = mean_ci(u)
    c = d["clv"].to_numpy(float)
    c = c[np.isfinite(c)]
    cm, ch = mean_ci(c) if len(c) > 1 else (np.nan, np.nan)
    return (f"n {len(d):>4}  ROI {100 * roi:+6.1f}% ± {100 * h:4.1f}  null {100 * d['null'].mean():+.1f}%  "
            f"CLV {cm:+.2f} ± {ch:.2f} pp")


def where_block(frames):
    lines = ["== 3b. Where v11's opening disagreement pays: value side at the open, P at 18 h "
             "(timing-honest), by the side's market role and the gap (± 95%; CLV = no-vig move "
             "to the close toward the bet)"]
    g = frames[1080]
    for label, _bk, x in cells(g):
        d = _value_open(x)
        if not len(d):
            continue
        lines.append(f"  {label}")
        for name, sel in (("all value sides", np.ones(len(d), bool)),
                          ("value favourite (q > 0.5) = H2·F", d["fav"]),
                          ("value underdog", ~d["fav"]),
                          ("gap < 0.04", d["edge"] < 0.04),
                          ("gap 0.04-0.08", (d["edge"] >= 0.04) & (d["edge"] < 0.08)),
                          ("gap 0.08-0.12", (d["edge"] >= 0.08) & (d["edge"] < 0.12)),
                          ("gap >= 0.12 = H4", d["edge"] >= 0.12),
                          ("gap >= 0.12, underdog", (d["edge"] >= 0.12) & ~d["fav"])):
            lines.append(f"    {name:<34} " + _roi(d[np.asarray(sel)]))
    return lines


def _rule_row(r):
    if not r:
        return "no bets"
    return (f"n {r['n']:>4}  ROI {100 * r['roi']:+6.1f}% ± {100 * Z95 * r['roi_se']:4.1f}  "
            f"null {100 * r['roi_null']:+.1f}%")


def rules_block(frames):
    lines = ["== 3c. The three tests by the P they read (hindsight; these games chose the rules). "
             "@open uses the open's price and line; @close the close's. ± 95%",
             "   recon P = the reconstructed rows (leave-one-season-out, 30-minute report: news "
             "the open did not have)"]
    variants = [("recon P (30 min, LOSO)", frames[30].assign(p_home=frames[30]["p_recon"])),
                *[(f"P at {LEADS[k]}", frames[k]) for k in LEADS]]
    for label, bk, _ in cells(frames[30]):
        lines.append(f"  {label}")
        for vname, g in variants:
            x = [c for c in cells(g) if c[0] == label][0][2]
            if not len(x):
                continue
            for b in analysis.hypothesis_backtest(x, bk):
                if not (b["open"] or b["close"]):
                    continue
                mv = b["move"]
                move = (f"  move {mv['mean']:+.2f} ± {Z95 * mv['se']:.2f} {mv['unit']}"
                        if mv and np.isfinite(mv["se"]) else "")
                lines.append(f"    {vname:<24} {b['label']:<14} @open {_rule_row(b['open'])}"
                             f"{move} | @close {_rule_row(b['close'])}")
    return lines


def alignment(frames):
    return scores_block(frames) + [""] + where_block(frames) + [""] + rules_block(frames)


# ------------------------------------------------------- team-game inputs --
SHRINK_GAMES = 5.0       # pregame rates lean on the league until a team has ~5 games


def season_games(y):
    """One row per game of season y (home view), from the Basketball-Reference
    logs: final points, possessions, and each team's decayed (half-life 25)
    pregame pace and points per 100 possessions (offence, defence) over its
    games strictly before this one, beside the league's to date."""
    import nba_composite as nc
    from walk_forward import game_possessions
    logs = nc.load_logs(y)
    rows = []
    for tm, lg in logs.items():
        poss = game_possessions(lg)
        pts, opp = lg["pts"].to_numpy(float), lg["opp_pts"].to_numpy(float)
        for i in range(len(lg)):
            w = 0.5 ** (np.arange(i)[::-1] / nc.HALF_LIFE)
            wp = (w * poss[:i]).sum()
            rows.append(dict(date=pd.Timestamp(lg["date"].iloc[i]), team=tm,
                             opp=lg["opp"].iloc[i], home=bool(lg["home"].iloc[i]),
                             gp=i, poss=poss[i], pts=pts[i], opp_pts=opp[i],
                             pace=wp / w.sum() if i else np.nan,
                             ortg=100 * (w * pts[:i]).sum() / wp if i else np.nan,
                             drtg=100 * (w * opp[:i]).sum() / wp if i else np.nan))
    t = pd.DataFrame(rows)
    # league to date: strictly before each date
    day = t.groupby("date").agg(poss=("poss", "sum"), pts=("pts", "sum"), n=("poss", "size"))
    cum = day.cumsum().shift(1)
    lg = pd.DataFrame({"lg_pace": cum["poss"] / cum["n"], "lg_rtg": 100 * cum["pts"] / cum["poss"]})
    t = t.join(lg, on="date")
    k = t["gp"] / (t["gp"] + SHRINK_GAMES)
    for c, ref in (("pace", "lg_pace"), ("ortg", "lg_rtg"), ("drtg", "lg_rtg")):
        t[c] = np.where(t["gp"] > 0, k * t[c] + (1 - k) * t[ref], t[ref])
    h = t[t["home"]].set_index(["date", "opp"])                  # opp = the away team
    a = t[~t["home"]].set_index(["date", "team"])[["gp", "pace", "ortg", "drtg", "pts"]]
    a.index.names = ["date", "opp"]
    g = h.join(a, rsuffix="_a", how="inner").drop(columns="home")
    g = g.reset_index().rename(columns={"team": "home_tm", "opp": "away_tm"})
    g["season"] = y
    g["slate_date"] = g["date"].dt.strftime("%Y-%m-%d")
    g["total"] = g["pts"] + g["pts_a"]
    g["lp"] = np.log(0.5 * (g["pace"] + g["pace_a"]) / g["lg_pace"])
    exp_poss = g["pace"] * g["pace_a"] / g["lg_pace"]
    g["naive_total"] = exp_poss * (g["ortg"] * g["drtg_a"] + g["ortg_a"] * g["drtg"]) \
        / g["lg_rtg"] / 100
    return g.rename(columns={"home_tm": "home", "away_tm": "away"})[
        ["season", "slate_date", "home", "away", "gp", "gp_a", "lp", "naive_total",
         "lg_pace", "lg_rtg", "total"]].dropna(subset=["lg_pace"])


def all_games(seasons):
    return pd.concat([season_games(y) for y in seasons], ignore_index=True)


# --------------------------------------------------------------- 1. pace --
PACE_SEASONS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
TEST = (2024, 2025, 2026)


def pace_frame(games, lead):
    """v11 walk-forward rows (lead `lead` for 2023-24 on, v10h25 before)
    with lp, the outcome, and each row's predictive sd: m / Phi^-1(P) (its
    fold's route sigma; rows with |m| < 0.5 take their season x route
    median)."""
    from statistics import NormalDist
    inv = NormalDist().inv_cdf
    q = v11_predictions()
    q = q[q["lead"] == lead].drop(columns="lead")
    e = early_predictions()
    d = pd.concat([e, q], ignore_index=True)
    d = d.merge(games[["season", "slate_date", "home", "away", "lp"]],
                on=["season", "slate_date", "home", "away"], how="left")
    d["lp"] = d["lp"].fillna(0.0)
    z = np.array([inv(min(max(p, 1e-9), 1 - 1e-9)) for p in d["p"]])
    d["sigma"] = np.where(np.abs(d["m"]) >= 0.5, d["m"] / np.where(z == 0, np.nan, z), np.nan)
    med = d.groupby(["season", "route"])["sigma"].transform("median")
    d["sigma"] = d["sigma"].fillna(med)
    d["y"] = (d["margin"] > 0).astype(float)
    d["late"] = np.fmin(d["gp_home"], d["gp_away"]) >= 10
    d["early"] = (np.fmin(d["gp_home"], d["gp_away"]) >= 1) & ~d["late"]
    return d


def _phi(x):
    from math import erf, sqrt
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(x, float) / sqrt(2)))


def _fit_pace(tr):
    """k (least squares of margin - m on m*lp) and s (ML of sigma exp(s lp)
    given m(1 + k lp)), on training rows."""
    x = tr["m"] * tr["lp"]
    k = float((x * (tr["margin"] - tr["m"])).sum() / (x * x).sum())
    mk = tr["m"] * (1 + k * tr["lp"])
    best = (np.inf, 0.0)
    for s in np.linspace(-6, 6, 241):
        p = _phi(mk / (tr["sigma"] * np.exp(s * tr["lp"])))
        ll = logloss(p, tr["y"]).mean()
        best = min(best, (ll, s))
    return k, best[1]


def pace_arms(d, k, s):
    return {
        "v11": d["p"].to_numpy(float),
        "pace scales margin (k)": _phi(d["m"] * (1 + k * d["lp"]) / d["sigma"]),
        "margin and spread (k, s)": _phi(d["m"] * (1 + k * d["lp"]) / (d["sigma"] * np.exp(s * d["lp"]))),
        "physics (k = 1, s = 0.5)": _phi(d["m"] * (1 + d["lp"]) / (d["sigma"] * np.exp(0.5 * d["lp"]))),
    }, {
        "v11": d["m"].to_numpy(float),
        "pace scales margin (k)": (d["m"] * (1 + k * d["lp"])).to_numpy(float),
        "margin and spread (k, s)": (d["m"] * (1 + k * d["lp"])).to_numpy(float),
        "physics (k = 1, s = 0.5)": (d["m"] * (1 + d["lp"])).to_numpy(float),
    }


def pace_block(games):
    lines = ["== 1. Pace: v11's walk-forward margin scaled by the matchup's expected pace "
             "(lp = log matchup / league possessions, pregame). Arm − v11, ± 95%; fits on earlier seasons only"]
    sd = games.loc[np.fmin(games["gp"], games["gp_a"]) >= 10, "lp"]
    lines.append(f"  Matchup pace spread, games 10+: sd(lp) {sd.std():.3f} (a 1-sd matchup is "
                 f"{100 * (np.exp(sd.std()) - 1):.1f}% more possessions than the league)")
    recon = ledger.load(ledger.RECON_PATH)
    recon["slate_date"] = recon["slate_date"].astype(str)
    for lead in (300, 1080):
        d = pace_frame(games, lead)
        lines.append(f"  v11 at {LEADS[lead]}")
        for y in TEST:
            tr = d[(d["season"] < y) & (d["early"] | d["late"])]
            k, s = _fit_pace(tr)
            te = d[d["season"] == y]
            arms, margins = pace_arms(te, k, s)
            lines.append(f"    {season_name(y)}  fitted on {len(tr)} games: k {k:+.2f}, s {s:+.2f}")
            for st, mask in (("games 1+", te["early"] | te["late"]), ("games 10+", te["late"])):
                mask = mask.to_numpy()
                yv = te["y"].to_numpy(float)[mask]
                base = logloss(arms["v11"][mask], yv)
                bb = (arms["v11"][mask] - yv) ** 2
                mb = (margins["v11"][mask] - te["margin"].to_numpy(float)[mask]) ** 2
                for name in list(arms)[1:]:
                    dl = logloss(arms[name][mask], yv) - base
                    db = (arms[name][mask] - yv) ** 2 - bb
                    dm = (margins[name][mask] - te["margin"].to_numpy(float)[mask]) ** 2 - mb
                    lines.append(f"      {st:<9} n {mask.sum():>4}  {name:<26} LL {pm(mean_ci(dl))}  "
                                 f"Brier {pm(mean_ci(db))}  MSE {pm(mean_ci(dm), 2)}")
            # the priced games: same differences on the book-seasons G7 reads
            pr = te.merge(recon[["season", "slate_date", "home", "away", "close_book"]],
                          on=["season", "slate_date", "home", "away"], how="inner")
            for bk in ("dk", "espnbet"):
                x = pr[(pr["close_book"] == bk) & pr["late"]]
                if len(x) < 200:
                    continue
                a2, _ = pace_arms(x, k, s)
                yv = x["y"].to_numpy(float)
                for name in list(a2)[1:3]:
                    dl = logloss(a2[name], yv) - logloss(a2["v11"], yv)
                    lines.append(f"      priced {market.BOOK_NAMES[bk]} games 10+ n {len(x)}  {name:<26} "
                                 f"LL {pm(mean_ci(dl))}")
    return lines


# -------------------------------------------------------------- 2. totals --
TOTALS = ROOT / "diagnostics" / "totals_signal" / "totals_odds.csv"


def totals_model(games):
    """Walk-forward total: total ~ a + b naive_total, fitted on earlier
    seasons' games 1+ (an OLS calibration of the possessions x efficiency
    product); returns games with `model_total`."""
    out = []
    for y in sorted(games["season"].unique()):
        tr = games[(games["season"] < y) & (np.fmin(games["gp"], games["gp_a"]) >= 1)]
        te = games[games["season"] == y].copy()
        if len(tr) < 500:
            continue
        X = np.column_stack([np.ones(len(tr)), tr["naive_total"]])
        b = np.linalg.lstsq(X, tr["total"].to_numpy(float), rcond=None)[0]
        te["model_total"] = b[0] + b[1] * te["naive_total"]
        out.append(te)
    return pd.concat(out, ignore_index=True)


def _ou_units(side_over, total, line, over_ml, under_ml):
    res = np.where(total == line, 0.5, np.where(side_over, total > line, total < line).astype(float))
    ml = np.where(side_over, over_ml, under_ml)
    u = np.array([0.0 if r == 0.5 else market.unit_profit(m, bool(r)) for m, r in zip(ml, res)])
    qo = np.array([market.devig(o, un) for o, un in zip(over_ml, under_ml)], float)
    q = np.where(side_over, qo, 1 - qo)
    null = q * np.array([market.decimal_payout(m) for m in ml]) - 1
    return u, null


def totals_block(games):
    lines = ["== 2. Totals (O/U). v11 has no total; a walk-forward total from the same box-score "
             "inputs against the books' totals (± 95%)"]
    if not TOTALS.exists():
        return lines + ["  no cached totals (run research/totals_fetch.py)"]
    t = pd.read_csv(TOTALS, dtype={"game_id": str})
    t = t[t["basis"] == "reconstructed"].dropna(subset=["book"])
    recon = ledger.load(ledger.RECON_PATH)
    recon["slate_date"] = recon["slate_date"].astype(str)
    recon["game_id"] = recon["game_id"].astype(str)
    m = totals_model(games)
    allg = m[np.fmin(m["gp"], m["gp_a"]) >= 10]
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))  # noqa: E731
    lines.append(f"  All games 10+ ({season_name(allg['season'].min())} .. "
                 f"{season_name(allg['season'].max())}, walk-forward): RMSE model {rm(allg['model_total'] - allg['total']):.2f} "
                 f"pts vs league-to-date mean {rm(allg['lg_rtg'] / 100 * allg['lg_pace'] * 2 - allg['total']):.2f}")
    j = recon[["game_id", "season", "slate_date", "home", "away", "close_book"]].merge(
        m, on=["season", "slate_date", "home", "away"], how="inner")
    j = j.merge(t, on="game_id", how="inner")
    j = j[j["book"] == j["close_book"]]            # the book that priced the row
    for bk, y, st in (("dk", 2026, "games 10+"), ("espnbet", 2025, "games 10+"),
                      ("espnbet", 2025, "games 1–9"), ("espnbet", 2026, "games 1–9")):
        gp = np.fmin(j["gp"], j["gp_a"])
        sel = (j["book"] == bk) & (j["season"] == y) & ((gp >= 10) if st == "games 10+" else (gp >= 1) & (gp < 10))
        x = j[sel]
        if len(x) < 50:
            continue
        lines.append(f"  {market.BOOK_NAMES[bk]} {season_name(y)} {st}  n {len(x)}")
        for when in ("open", "close"):
            line = x[f"{when}_total"].to_numpy(float)
            lo, hi = analysis.ATS_MAIN_LINE     # main lines only (see totals_model.main_line)
            o, u = x[f"{when}_over"].to_numpy(float), x[f"{when}_under"].to_numpy(float)
            ok = np.isfinite(line) & (o >= lo) & (o <= hi) & (u >= lo) & (u <= hi)
            xx, line = x[ok], line[ok]
            tot, mt = xx["total"].to_numpy(float), xx["model_total"].to_numpy(float)
            diff = mt - line
            # outcome weight on (model - line): OLS of (total - line) on (model - line)
            X = np.column_stack([np.ones(len(diff)), diff])
            b, res, *_ = np.linalg.lstsq(X, tot - line, rcond=None)
            r = (tot - line) - X @ b
            cov = (r @ r / (len(r) - 2)) * np.linalg.inv(X.T @ X)
            lines.append(f"    {when:<5} n {len(xx):>4}  RMSE line {rm(tot - line):.2f}  model {rm(tot - mt):.2f}  "
                         f"model − line sd {diff.std():.2f}  weight on (model − line) "
                         f"{b[1]:+.2f} ± {Z95 * np.sqrt(cov[1, 1]):.2f}")
            for thr in (0.0, 3.0, 6.0):
                s = np.abs(diff) >= thr if thr else diff != 0
                if s.sum() < 10:
                    continue
                u, null = _ou_units(diff[s] > 0, tot[s], line[s], xx[f"{when}_over"].to_numpy(float)[s],
                                    xx[f"{when}_under"].to_numpy(float)[s])
                roi, h = mean_ci(u)
                extra = ""
                if when == "open":
                    cl = xx["close_total"].to_numpy(float)[s]
                    mv = np.where(diff[s] > 0, cl - line[s], line[s] - cl)
                    mv = mv[np.isfinite(mv)]
                    extra = f"  line moved toward it {pm(mean_ci(mv), 2)} pts"
                lines.append(f"      model side, |model − line| ≥ {thr:.0f}: n {s.sum():>4}  ROI "
                             f"{100 * roi:+6.1f}% ± {100 * h:4.1f}  null {100 * null.mean():+.1f}%{extra}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--parts", default="pace,totals,picks")
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    lines = ["v11 review: pace, totals and the picks (research/v11_review.py). ± 95% unless labelled.", ""]
    if parts & {"pace", "totals"}:
        games = all_games(PACE_SEASONS)
        if "pace" in parts:
            lines += pace_block(games) + [""]
        if "totals" in parts:
            lines += totals_block(games) + [""]
    if "picks" in parts:
        frames = priced(v11_predictions())
        lines += alignment(frames) + [""]
    text = "\n".join(lines)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
