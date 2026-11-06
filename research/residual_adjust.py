#!/usr/bin/env python3
"""Per-team rolling 10-day model residual, halved, as a sides and totals adjustment.

  python research/residual_adjust.py --totals-run   # walk-forward T* and ou2 totals
  python research/residual_adjust.py --report       # -> diagnostics/residual_adjust/RESULTS.txt

Owner's request (2026-11-06): test a per-team rolling 10-day model
residual / 2 adjustment for sides and totals. Signal, arms, baselines,
readings and decision rules are fixed here before the run.

Signal. For team t and slate date D, R_t(D) is the mean, over t's games of
the same season dated D-10 .. D-1 (ten calendar days, strictly before the
slate), of the baseline model's residual on that game from t's side:
sides, final margin minus predicted margin from t's view (home +, away -);
totals, final total minus predicted total. Each residual uses the
baseline's own pregame (walk-forward) prediction for that game, as a live
record would hold it, never an adjusted one. A team with no game in the
window has R = 0. Games the baseline abstained on contribute nothing.

Arm A (the request; weight 1/2 fixed, not fitted; sigma unchanged):
  sides   m' = m + (R_home - R_away) / 2,  P' = Phi(m' / sigma)
  totals  T' = T + (R_home + R_away) / 2,  P(over) = 1 - Phi((line - T') / sigma)
Halving each team's mean residual and crediting half of each game's
residual to each team give the same formula.

Baselines.
  Sides: production routing, walk-forward: diagnostics/margin_form arm B
  (research/margin_form.py: v10h25; the base route fitted on earlier
  seasons; the NBA-report route, from 2023-24, refitted before each slate;
  od statuses, not v11's Questionable weighting). sigma is recovered as
  m / Phi^-1(P): one value per fold on the base route, per slate on the
  report route. Folds 2017 .. 2026 for the rule; 2026-27 information only.
  Totals: T* (totals.FEATURES, the ou1 features) walk-forward, each season
  fitted on earlier ones (2016-17 trains only); and production ou2 = T* plus
  the report terms at the 300-minute lead with v11's Questionable weighting
  (research/totals_report.py arm rep5h), fitted on T*'s residual over the
  report seasons before the scored one (2022-23 trains only). ou2 seasons
  2023-24 .. 2025-26 for the rule; 2026-27 information only. Each baseline's
  arm uses that baseline's own residuals.

Readings, arm minus baseline on identical games, paired, ± 95%:
  sides   log loss, Brier and margin MSE against outcomes by stage (games
          1-9, 10+), route and fold; against the close, each basis x book x
          season x stage alone (reconstructed and 2026-27 rebuilt rows;
          hindsight).
  totals  squared error against the final total by stage and season;
          P(over) log loss at each book's closing main line (both prices
          -135..+115, pushes dropped), each book x season x stage; the
          books' open / close readings of research/totals_model.evaluate.
  both    information only: the fitted weight, the OLS slope (with an
          intercept) of the baseline's residual on R_home -/+ R_away over
          games 1+; 0.5 is the requested weight, 0 no signal.

Decision rules, set before the run:
  sides   the G7 gate as research/margin_form.py applied it: on folds 2017 ..
          2026, log loss not worse than B on reconstructed DraftKings 2025-26
          and ESPN BET 2024-25 games 10+ with Brier agreeing, not worse on
          games 1+ against outcomes, and no fold better by more than 0.008
          (leakage tripwire).
  totals  against ou2 on 2023-24 .. 2025-26: squared error on games 1+
          better with the 95% interval below zero (research/totals_report.py's
          selection), and P(over) log loss and Brier at the close not worse
          on DraftKings 2025-26 and ESPN BET 2024-25 games 10+.
Passing makes the arm a candidate for the owner's decision and an amendment
with a new model tag, not a change. Research only: nothing here touches the
ledgers, records, pages or model/*.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

OUT = os.path.join(ROOT, "diagnostics", "residual_adjust")
SIDES = os.path.join(ROOT, "diagnostics", "margin_form", "predictions_2017-2027.csv")
TOTALS = os.path.join(OUT, "totals_predictions.csv")
WINDOW = 10            # calendar days before the slate: D-10 .. D-1
WEIGHT = 0.5           # the requested residual / 2
SIDE_FOLDS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
T_SEASONS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026, 2027)
REPORT = (2023, 2024, 2025, 2026, 2027)      # 2023 trains ou2 only
OU2_RULE = (2024, 2025, 2026)
LEAD = 300
Z95 = 1.96
_N = NormalDist()
PHI = np.vectorize(_N.cdf, otypes=[float])
PHI_INV = np.vectorize(_N.inv_cdf, otypes=[float])
STAGES = (("games 1+", 1, 10 ** 6), ("games 1-9", 1, 9), ("games 10+", 10, 10 ** 6))


# --------------------------------------------------------------- signal --
def rolling_residual(df, e, sides, window=WINDOW):
    """Per row of df (season, slate_date, home, away): each team's mean
    residual over its same-season games dated D-window .. D-1 and their
    count. e is each row's residual (home view for sides, the total's for
    totals); NaN rows (no prediction or no result) are skipped."""
    d = pd.to_datetime(df["slate_date"]).to_numpy("datetime64[D]")
    e = np.asarray(e, float)
    season = df["season"].to_numpy()
    long = pd.DataFrame({"season": np.concatenate([season, season]),
                         "team": np.concatenate([df["home"].to_numpy(), df["away"].to_numpy()]),
                         "date": np.concatenate([d, d]),
                         "r": np.concatenate([e, -e if sides else e])})
    long = long[np.isfinite(long["r"])].sort_values("date", kind="stable")
    look = {}
    for k, g in long.groupby(["season", "team"]):
        look[k] = (g["date"].to_numpy("datetime64[D]"),
                   np.concatenate([[0.0], np.cumsum(g["r"].to_numpy(float))]))
    out = {}
    for side in ("home", "away"):
        R, N = np.zeros(len(df)), np.zeros(len(df), int)
        for i, (s, t, day) in enumerate(zip(season, df[side].to_numpy(), d)):
            hit = look.get((s, t))
            if hit is None:
                continue
            dates, cs = hit
            lo = np.searchsorted(dates, day - np.timedelta64(window, "D"), "left")
            hi = np.searchsorted(dates, day, "left")            # same-day games excluded
            if hi > lo:
                R[i], N[i] = (cs[hi] - cs[lo]) / (hi - lo), hi - lo
        out[side] = (R, N)
    return out["home"][0], out["away"][0], out["home"][1], out["away"][1]


def adjust_sides(m, sigma, r_home, r_away, weight=WEIGHT):
    m2 = np.asarray(m, float) + weight * (np.asarray(r_home) - np.asarray(r_away))
    return m2, PHI(m2 / np.asarray(sigma, float))


def adjust_totals(T, r_home, r_away, weight=WEIGHT):
    return np.asarray(T, float) + weight * (np.asarray(r_home) + np.asarray(r_away))


# ------------------------------------------------------------- helpers --
def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return (float(d.mean()) if len(d) else np.nan), np.nan
    return float(d.mean()), Z95 * float(d.std(ddof=1)) / np.sqrt(len(d))


def pm(t, k=4):
    return f"{t[0]:+.{k}f} ± {t[1]:.{k}f}"


def slope(e, s):
    """OLS slope of e on s with an intercept, ± 95%."""
    e, s = np.asarray(e, float), np.asarray(s, float)
    ok = np.isfinite(e) & np.isfinite(s)
    X = np.column_stack([np.ones(ok.sum()), s[ok]])
    b, *_ = np.linalg.lstsq(X, e[ok], rcond=None)
    r = e[ok] - X @ b
    se = np.sqrt(r @ r / (len(r) - 2) * np.linalg.inv(X.T @ X)[1, 1])
    return float(b[1]), Z95 * float(se)


def ceiling(e, s):
    """In-sample MSE reduction from the signal at its OLS slope, b^2 var(s):
    an upper bound on what any fixed weight on it buys."""
    e, s = np.asarray(e, float), np.asarray(s, float)
    ok = np.isfinite(e) & np.isfinite(s)
    return slope(e, s)[0] ** 2 * float(s[ok].var()), float(np.mean((e[ok] - e[ok].mean()) ** 2))


def stage_mask(gp, lo, hi):
    gp = np.asarray(gp)
    return (gp >= lo) & (gp <= hi)


def window_lines(n_home, n_away, sig, gp, label):
    m = stage_mask(gp, 1, 10 ** 6)
    n = np.concatenate([n_home[m], n_away[m]])
    adj = WEIGHT * np.asarray(sig)[m]
    return [f"  window, games 1+: {n.mean():.2f} team games on average; none for "
            f"{(n == 0).mean():.1%} of team-games; adjustment {label} sd {adj.std():.2f}, "
            f"mean |adj| {np.abs(adj).mean():.2f}, 95th pct |adj| {np.percentile(np.abs(adj), 95):.2f}"]


# --------------------------------------------------------------- sides --
def sides_frame():
    P = pd.read_csv(SIDES)
    P = P.drop_duplicates(["season", "slate_date", "home", "away"], keep="last").reset_index(drop=True)
    z = PHI_INV(np.clip(P["p_B"].to_numpy(float), 1e-12, 1 - 1e-12))
    raw = np.where(np.abs(z) > 0.02, P["m_B"].to_numpy(float) / np.where(z == 0, np.nan, z), np.nan)
    P["sig_raw"] = raw
    by_slate = P.groupby(["season", "route_B", "slate_date"])["sig_raw"].transform("median")
    by_route = P.groupby(["season", "route_B"])["sig_raw"].transform("median")
    # the base route has one sigma per fold; the report route one per slate
    P["sigma"] = np.where(P["route_B"] == "avail", by_slate.fillna(by_route), by_route)
    P["y"] = (P["margin"] > 0).astype(float)
    P["gp_min"] = np.fmin(P["gp_home"], P["gp_away"])
    e = P["margin"] - P["m_B"]
    rh, ra, nh, na = rolling_residual(P, e, sides=True)
    P["R_home"], P["R_away"], P["n_home"], P["n_away"] = rh, ra, nh, na
    P["S"] = rh - ra
    P["m_A"], P["p_A"] = adjust_sides(P["m_B"], P["sigma"], rh, ra)
    return P


def side_diffs(g):
    y = g["y"].to_numpy()
    return (ll(g["p_A"], y) - ll(g["p_B"], y), (g["p_A"] - y) ** 2 - (g["p_B"] - y) ** 2,
            (g["m_A"] - g["margin"]) ** 2 - (g["m_B"] - g["margin"]) ** 2)


def sides_report():
    P = sides_frame()
    check = np.abs(PHI(P["m_B"] / P["sigma"]) - P["p_B"]).max()
    R = P[P["season"].isin(SIDE_FOLDS)]
    L = ["== SIDES: arm A = B + (R_home - R_away) / 2 on the predicted margin",
         f"  baseline: diagnostics/margin_form arm B (production routing, walk-forward), {len(P)} games "
         f"{P['season'].min()}..{P['season'].max()}; sigma recovered, max |Phi(m/sigma) - P| {check:.1e}"]
    L += window_lines(P["n_home"].to_numpy(), P["n_away"].to_numpy(), P["S"].to_numpy(),
                      P["gp_min"].to_numpy(), "(points)")
    L += ["", "-- 1. Against outcomes, folds 2017-2026 (A - B; log loss ± 95%, Brier, margin MSE ± 95%)"]
    groups = [(lab, R[stage_mask(R["gp_min"], lo, hi)]) for lab, lo, hi in STAGES] + \
             [(f"route {r}, games 1+", R[(R["route_B"] == r) & (R["gp_min"] >= 1)]) for r in ("base", "avail")]
    for lab, g in groups:
        d_ll, d_br, d_mse = side_diffs(g)
        wins = sum(side_diffs(x)[0].mean() < 0 for _, x in g.groupby("season"))
        L.append(f"  {lab:20s} n {len(g):5d}  B log loss {ll(g['p_B'], g['y']).mean():.4f}  "
                 f"A - B {pm(ci(d_ll))}  Brier {d_br.mean():+.5f}  MSE {pm(ci(d_mse), 2)}  "
                 f"better in {wins} of {g['season'].nunique()}")
    L.append("  By fold, games 1+ (A - B log loss; margin MSE):")
    for Y, g in P[P["gp_min"] >= 1].groupby("season"):
        d_ll, _, d_mse = side_diffs(g)
        tag = "" if Y in SIDE_FOLDS else "   (information only)"
        L.append(f"    {Y} n {len(g):4d}  {pm(ci(d_ll))}   MSE {pm(ci(d_mse), 2)}{tag}")

    rec = pd.read_csv(os.path.join(ROOT, "data", "nba_reconstructed.csv")).assign(basis="reconstructed")
    reb = pd.read_csv(os.path.join(ROOT, "data", "nba_rebuilt.csv")).assign(basis="rebuilt")
    M = pd.concat([rec, reb])[["basis", "season", "slate_date", "home", "away", "close_book",
                               "close_q_home"]].dropna(subset=["close_q_home"])
    W = P.merge(M, on=["season", "slate_date", "home", "away"])
    L += ["", "-- 2. Against the close, each basis x book x season x stage "
              "(model - close log loss; [A - B ± 95%, Brier])"]
    gate = {}
    for (basis, book, Y, st), g in W.groupby(["basis", "close_book", "season", "stage"]):
        if st == "game 0" or len(g) < 30:
            continue
        y = g["y"].to_numpy()
        lc = ll(g["close_q_home"], y)
        d_ll, d_br, _ = side_diffs(g)
        gate[(basis, book, Y, st)] = (d_ll.mean(), d_br.mean())
        L.append(f"  {basis:13s} {book:7s} {Y} {st:10s} n {len(g):4d}  B {pm(ci(ll(g['p_B'], y) - lc))}  "
                 f"A {ci(ll(g['p_A'], y) - lc)[0]:+.4f}  [{pm(ci(d_ll))}, Brier {d_br.mean():+.5f}]")

    L += ["", "-- 3. Fitted weight (information): slope of (margin - m_B) on (R_home - R_away), "
              "folds 2017-2026; requested 0.5"]
    for lab, lo, hi in STAGES:
        g = R[stage_mask(R["gp_min"], lo, hi)]
        cut, mse = ceiling(g["margin"] - g["m_B"], g["S"])
        L.append(f"  {lab:10s} n {len(g):5d}  slope {pm(slope(g['margin'] - g['m_B'], g['S']), 3)}  "
                 f"in-sample MSE reduction at that slope {cut:.2f} of {mse:.1f} (upper bound)")
    g = R[R["gp_min"] >= 1]
    L.append("  by fold, games 1+: " + "  ".join(
        f"{Y} {slope(x['margin'] - x['m_B'], x['S'])[0]:+.2f}" for Y, x in g.groupby("season")))

    L += ["", "-- 4. Decision rule (set before the run; the G7 gate as research/margin_form.py applied it)"]
    books = [gate.get(("reconstructed", "dk", 2026, "games 10+")),
             gate.get(("reconstructed", "espnbet", 2025, "games 10+"))]
    r1 = all(b is not None and b[0] <= 0 and b[1] <= 0 for b in books)
    g1 = R[R["gp_min"] >= 1]
    d_all = side_diffs(g1)[0].mean()
    r2 = d_all <= 0
    best = min(side_diffs(x)[0].mean() for _, x in g1.groupby("season"))
    r3 = best >= -0.008
    L.append(f"  A  priced books DK 2025-26 {books[0][0]:+.4f} / ESPN BET 2024-25 {books[1][0]:+.4f} "
             f"(Brier {books[0][1]:+.5f} / {books[1][1]:+.5f}): {'pass' if r1 else 'fail'}; "
             f"games 1+ {d_all:+.4f}: {'pass' if r2 else 'fail'}; tripwire (best fold {best:+.4f}): "
             f"{'pass' if r3 else 'fail'} -> {'CANDIDATE' if r1 and r2 and r3 else 'not a candidate'}")
    return L


# -------------------------------------------------------------- totals --
def totals_run():
    """Walk-forward T* (all seasons) and ou2 (report seasons) per game."""
    import nba_composite as nc
    import player_availability as pav
    import totals
    import totals_model as tm
    import totals_report as trep
    rows = totals.training_rows({y: nc.load_logs(y) for y in T_SEASONS})
    rows["slate_date"] = rows["date"]
    rows["stage_gp"] = np.fmin(rows["gp_home"], rows["gp_away"])
    base = tm.walk_forward(rows, list(totals.FEATURES))
    base = base.rename(columns={"T": "T_star", "sigma": "sigma_star"})
    os.makedirs(pav.DEFAULT_CACHE, exist_ok=True)
    rates = pav.lead_rates(json.load(open(os.path.join(ROOT, "model", "logit_avail.json")))["q_rates"], LEAD)
    terms = {t: trep.report_terms(t, LEAD, rates)[0] for t in REPORT}
    d = base[base["season"].isin(REPORT)].copy()
    cols = {}
    for k in trep.TERMS + ["covered"]:
        h = [terms[s].get((dt, t), {}).get(k, np.nan) for s, dt, t in zip(d.season, d.slate_date, d.home)]
        a = [terms[s].get((dt, t), {}).get(k, np.nan) for s, dt, t in zip(d.season, d.slate_date, d.away)]
        cols[k] = (np.array(h, float) + np.array(a, float)) if k != "covered" else \
            np.array([bool(x == 1) and bool(z == 1) for x, z in zip(h, a)])
    X = np.column_stack([np.nan_to_num(cols[k]) for k in trep.TERMS])
    res = (d["total"] - d["T_star"]).to_numpy(float)
    early = (d["stage_gp"] < 10).to_numpy()
    T2, s2 = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for s in REPORT[1:]:
        tr = ((d["season"] < s) & (d["stage_gp"] >= 1)).to_numpy() & np.isfinite(res)
        te = (d["season"] == s).to_numpy()
        X1 = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(X1, res[tr], rcond=None)[0]
        r2 = res[tr] - X1 @ b
        T2[te] = d["T_star"].to_numpy()[te] + np.column_stack([np.ones(te.sum()), X[te]]) @ b
        s2[te] = np.where(early[te], r2[early[tr]].std(), r2[~early[tr]].std())
        print(f"ou2 fit for {s}: " + "  ".join(f"{n} {v:+.3f}" for n, v in zip(["a"] + trep.TERMS, b)),
              file=sys.stderr)
    d["T_ou2"], d["sigma_ou2"], d["rep_covered"] = T2, s2, cols["covered"]
    keep = ["season", "slate_date", "home", "away"]
    out = base.merge(d[keep + ["T_ou2", "sigma_ou2", "rep_covered"]], on=keep, how="left")
    os.makedirs(OUT, exist_ok=True)
    out[keep + ["gp_home", "gp_away", "stage_gp", "total", "T_star", "sigma_star", "T_ou2",
                "sigma_ou2", "rep_covered"]].to_csv(TOTALS, index=False, float_format="%.6g")
    print(f"wrote {TOTALS}: {len(out)} games", file=sys.stderr)


def totals_arm(G, base):
    """Rows with baseline `base` (T_star / T_ou2): T, sigma, the signal and arm A."""
    g = G[np.isfinite(G[f"T_{base}"])].copy().reset_index(drop=True)
    g["T"], g["sigma"] = g[f"T_{base}"], g[f"sigma_{base}"]
    rh, ra, nh, na = rolling_residual(g, g["total"] - g["T"], sides=False)
    g["R_home"], g["R_away"], g["n_home"], g["n_away"] = rh, ra, nh, na
    g["S"] = rh + ra
    g["T_A"] = adjust_totals(g["T"], rh, ra)
    return g


def over_cells(j, base_lab):
    """P(over) log loss at each book's closing main line, A - baseline."""
    import totals_model as tm
    L, gate = [], {}
    for (bk, Y), x in j.groupby(["book", "season"]):
        for lab, lo, hi in STAGES[1:]:
            xs = x[stage_mask(x["stage_gp"], lo, hi)]
            line = xs["close_total"].to_numpy(float)
            o, u = xs["close_over"].to_numpy(float), xs["close_under"].to_numpy(float)
            tot = xs["total"].to_numpy(float)
            ok = tm.main_line(line, o, u) & (tot != line)
            if ok.sum() < 30:
                continue
            y = (tot[ok] > line[ok]).astype(float)
            q = tm._q_over(o[ok], u[ok])
            s = xs["sigma"].to_numpy(float)[ok]
            pb = 1 - PHI((line[ok] - xs["T"].to_numpy(float)[ok]) / s)
            pa = 1 - PHI((line[ok] - xs["T_A"].to_numpy(float)[ok]) / s)
            d_ll = ll(pa, y) - ll(pb, y)
            d_br = (pa - y) ** 2 - (pb - y) ** 2
            gate[(bk, Y, lab)] = (d_ll.mean(), d_br.mean())
            L.append(f"  {bk:7s} {Y} {lab:10s} n {ok.sum():4d}  {base_lab} - market "
                     f"{pm(ci(ll(pb, y) - ll(q, y)))}  A - market {ci(ll(pa, y) - ll(q, y))[0]:+.4f}  "
                     f"[A - {base_lab} {pm(ci(d_ll))}, Brier {d_br.mean():+.5f}]")
    return L, gate


def totals_report():
    import totals_model as tm
    G = pd.read_csv(TOTALS)
    L = ["== TOTALS: arm A = baseline + (R_home + R_away) / 2 on the predicted total"]
    gates = {}
    for base, lab, rule in (("ou2", "ou2", OU2_RULE), ("star", "T*", None)):
        g = totals_arm(G, base)
        rule = rule or tuple(s for s in sorted(g["season"].unique()) if s < 2027)
        R = g[g["season"].isin(rule)]
        L += ["", f"-- baseline {lab}: {len(g)} games, seasons {g['season'].min()}..{g['season'].max()} "
                  f"(rule seasons {rule[0]}..{rule[-1]}; 2026-27 information only)"]
        L += window_lines(g["n_home"].to_numpy(), g["n_away"].to_numpy(), g["S"].to_numpy(),
                          g["stage_gp"].to_numpy(), "(points)")
        L.append(f"   1. Squared error against the final total, A - {lab} (± 95%; negative = better)")
        sq = lambda x: (x["total"] - x["T_A"]) ** 2 - (x["total"] - x["T"]) ** 2  # noqa: E731
        for slab, lo, hi in STAGES:
            x = R[stage_mask(R["stage_gp"], lo, hi)]
            rm_b = np.sqrt(((x["total"] - x["T"]) ** 2).mean())
            rm_a = np.sqrt(((x["total"] - x["T_A"]) ** 2).mean())
            wins = sum(sq(z).mean() < 0 for _, z in x.groupby("season"))
            L.append(f"     {slab:10s} n {len(x):5d}  RMSE {rm_b:.3f} -> {rm_a:.3f}  "
                     f"Δ {pm(ci(sq(x)), 2)}  better in {wins} of {x['season'].nunique()}")
            if slab == "games 1+":
                gates[(base, "sq")] = ci(sq(x))
        for Y, x in g[g["stage_gp"] >= 1].groupby("season"):
            tag = "" if Y in rule else "   (information only)"
            L.append(f"       {Y} games 1+ n {len(x):4d}  Δ {pm(ci(sq(x)), 2)}{tag}")
        L.append(f"   2. Fitted weight (information): slope of (total - {lab}) on (R_home + R_away); "
                 "requested 0.5")
        for slab, lo, hi in STAGES:
            x = R[stage_mask(R["stage_gp"], lo, hi)]
            cut, mse = ceiling(x["total"] - x["T"], x["S"])
            L.append(f"     {slab:10s} n {len(x):5d}  slope {pm(slope(x['total'] - x['T'], x['S']), 3)}  "
                     f"in-sample MSE reduction at that slope {cut:.2f} of {mse:.1f} (upper bound)")
        x = R[R["stage_gp"] >= 1]
        L.append("     by season, games 1+: " + "  ".join(
            f"{Y} {slope(z['total'] - z['T'], z['S'])[0]:+.2f}" for Y, z in x.groupby("season")))
        j = tm.priced(g[["season", "slate_date", "home", "away", "stage_gp", "total", "T", "sigma",
                         "T_A"]])
        L.append(f"   3. P(over) log loss at the closing main line, each book x season x stage "
                 f"(model - no-vig market; [A - {lab} ± 95%, Brier])")
        cells, gate = over_cells(j, lab)
        L += ["   " + c for c in cells]
        gates[(base, "over")] = gate
        if base == "ou2":
            L.append(f"   4. Books, games 10+ (research/totals_model.evaluate; {lab} then A; "
                     "information only)")
            for (bk, Y), x in j.groupby(["book", "season"]):
                xs = x[x["stage_gp"] >= 10]
                if len(xs) < 60:
                    continue
                L.append(f"     {bk} {Y} games 10+")
                for when in ("open", "close"):
                    L += ["   " + s for s in tm.evaluate(xs, when, lab)]
                    L += ["   " + s for s in tm.evaluate(xs.assign(T=xs["T_A"]), when, "A")]

    L += ["", "-- Decision rule (set before the run), against ou2 on 2023-24 .. 2025-26"]
    m, h = gates[("ou2", "sq")]
    r1 = m + h < 0
    gate = gates[("ou2", "over")]
    books = [gate.get(("dk", 2026, "games 10+")), gate.get(("espnbet", 2025, "games 10+"))]
    r2 = all(b is not None and b[0] <= 0 and b[1] <= 0 for b in books)
    fb = lambda b: "n/a" if b is None else f"{b[0]:+.4f} (Brier {b[1]:+.5f})"  # noqa: E731
    L.append(f"  A  squared error games 1+ {m:+.2f} ± {h:.2f}: {'pass' if r1 else 'fail'}; "
             f"P(over) log loss at the close DK 2025-26 {fb(books[0])} / ESPN BET 2024-25 {fb(books[1])}: "
             f"{'pass' if r2 else 'fail'} -> {'CANDIDATE' if r1 and r2 else 'not a candidate'}")
    return L


def report():
    head = ["Per-team rolling 10-day model residual / 2: sides and totals adjustment",
            "(research/residual_adjust.py; walk-forward hindsight, ± 95%; research only, no model change).",
            f"R_team = mean of the team's model residuals over its games dated D-{WINDOW} .. D-1 "
            "(same season; 0 when none);",
            "sides m' = m + (R_home - R_away)/2, totals T' = T + (R_home + R_away)/2; sigma unchanged.", ""]
    L = head + sides_report() + ["", ""] + totals_report()
    text = "\n".join(L) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "RESULTS.txt"), "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--totals-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.totals_run:
        totals_run()
    if a.report:
        report()


if __name__ == "__main__":
    main()
