#!/usr/bin/env python3
"""In-season BPM: last season's value updated with this season's box lines.

  python research/bpm_live.py check [--fit 2016-2019 2021-2024] [--test 2025 2026]

Owner's question, 2026-10-22: production values every player by last
season's BBR BPM, frozen for the season (player_availability.LiveAvailability;
minutes update, values do not). This module supplies the dated alternative
research/team_quality.py --bpm tests, walk-forward:

  BPM to date   BBR publishes no daily history, so BPM is rebuilt from ESPN
                box lines (research/win_shares.py box_full_<y>.csv): per-100-
                possession rates (points, FGA, FTA, 3PM, AST, TOV, ORB, DRB,
                STL, BLK, PF; possessions = the team's estimate x minutes /
                (team minutes / 5)), each less the league's minute-weighted
                rate over the same games (BPM is league-relative; raw rates
                drift with the era), and the team's net rating per 100, all
                over games strictly before the date, mapped to BPM by a
                minutes-weighted least-squares fit on BBR's full-season BPM
                of FIT seasons only (default 2016-2024, before both priced
                test seasons). Coefficients are fixed per run, like BPM's own;
                `check` scores the mapping on held-out seasons' BBR tables.
  prior         production's value: last-season BPM shrunk toward
                replacement (-2.0) by MP / (MP + 500); unmatched (rookies) at
                replacement.
  bpmb          Bayes: (M0 x prior + MP x obs) / (M0 + MP), M0 = 750 (the Win
                Shares test's constant), MP and obs to date.
  bpmo          this season only: MP / (MP + M0) x (obs - replacement).
Values are BPM above replacement per 48, the scale of talent_diff, read by
win_shares.dyn_talent_fn (previous box score's roster x minutes role).
Research only; the model reads none of it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402
import win_shares as ws  # noqa: E402

RATES = ["pts", "fga", "fta", "tpm", "ast", "tov", "orb", "drb", "stl", "blk", "pf"]
FEATURES = RATES + ["net"]
BPM_PRIOR_MP = 750.0        # in-season minutes for half weight vs the prior
FIT_MIN_MP = 250.0          # players in the coefficient fit (weighted by MP)
FIT_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
R = pav.REPLACEMENT_BPM


def player_rates(full, through=None):
    """Per player over games before `through` (all if None): minutes,
    possessions on court, per-100 rates and the minute-weighted team net
    rating per 100, summed over team stints."""
    f = full if through is None else full[full["date"] < pd.Timestamp(through)]
    f = f[f["minutes"] > 0]
    cols = ["player_id", "name", "minutes", "poss", "net"] + RATES
    if not len(f):
        return pd.DataFrame(columns=cols)
    tg = ws.team_games(f)
    tot = tg.groupby("team")[ws.STATS + ["minutes"]
                             + ["o_" + c for c in ws.STATS + ["minutes"]]].sum()
    T = {c: tot[c].to_numpy(float) for c in ws.STATS + ["minutes"]}
    O = {c: tot["o_" + c].to_numpy(float) for c in ws.STATS + ["minutes"]}
    team_poss = pd.Series(ws.possessions(T, O), index=tot.index)
    team_net = pd.Series(100 * ws._div(T["pts"] - O["pts"], team_poss.to_numpy()),
                         index=tot.index)
    p = f.groupby(["player_id", "team"], as_index=False).agg(
        name=("name", "last"), **{c: (c, "sum") for c in ws.STATS + ["minutes"]})
    share = ws._div(p["minutes"].to_numpy(float),
                    tot.loc[p["team"], "minutes"].to_numpy(float) / 5)
    p["poss"] = team_poss.loc[p["team"]].to_numpy() * share
    p["netm"] = team_net.loc[p["team"]].to_numpy() * p["minutes"]
    g = p.groupby("player_id").agg(name=("name", "last"), minutes=("minutes", "sum"),
                                   poss=("poss", "sum"), netm=("netm", "sum"),
                                   **{c: (c, "sum") for c in RATES})
    w = g["minutes"].to_numpy(float)
    for c in RATES:
        r = 100 * ws._div(g[c].to_numpy(float), g["poss"].to_numpy(float))
        g[c] = r - (w * r).sum() / w.sum()      # BPM is relative to the league
    g["net"] = ws._div(g["netm"].to_numpy(float), w)
    return g.reset_index()[cols]


def design(rates):
    return np.column_stack([np.ones(len(rates)), rates[FEATURES].to_numpy(float)])


def fit_beta(seasons, cache_dir=pav.DEFAULT_CACHE):
    """Minutes-weighted least squares of BBR's full-season BPM on the box
    rates, over `seasons` with cached box lines; players with FIT_MIN_MP+
    minutes whose box minutes are within 10% of BBR's. -> (beta, n)."""
    X, y, w = [], [], []
    for s in seasons:
        if not os.path.exists(os.path.join(cache_dir, f"box_full_{s}.csv")):
            continue
        m = _matched(player_rates(ws.load_full(s, cache_dir)), pav.load_bpm(s))
        m = m[m["minutes"] >= FIT_MIN_MP]
        X.append(design(m))
        y.append(m["bbr_bpm"].to_numpy(float))
        w.append(m["minutes"].to_numpy(float))
    if not X:
        raise SystemExit("no cached box_full seasons to fit BPM on "
                         "(run research/win_shares.py fetch)")
    X, y, sw = np.vstack(X), np.concatenate(y), np.sqrt(np.concatenate(w))
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta, len(y)


def _matched(rates, bpm):
    m = rates.assign(key=rates["name"].map(pav.norm_name))
    m = m[m["key"].isin(bpm)].copy()
    m["bbr_bpm"] = m["key"].map(lambda k: bpm[k][0])
    m["bbr_mp"] = m["key"].map(lambda k: bpm[k][1])
    return m[(m["minutes"] - m["bbr_mp"]).abs() < 0.1 * m["bbr_mp"] + 20]


def bpm_history(full, beta):
    """player_id -> (dates, bpm, mp): BPM rebuilt from games strictly before
    each game date of the season (look up as win_shares.ws_history)."""
    hist = {}
    for d in sorted(full["date"].unique()):
        r = player_rates(full, through=d)
        if not len(r):
            continue
        est = design(r) @ beta
        for pid, b, mp in zip(r["player_id"], est, r["minutes"].to_numpy(float)):
            hist.setdefault(pid, ([], [], []))
            hist[pid][0].append(np.datetime64(pd.Timestamp(d)))
            hist[pid][1].append(b)
            hist[pid][2].append(mp)
    return {pid: (np.array(a), np.array(b, float), np.array(c, float))
            for pid, (a, b, c) in hist.items()}


def prior_bpm(box, bpm):
    """player_id -> production's shrunk last-season BPM (replacement if
    unmatched): R + (BPM - R) x MP / (MP + BPM_SHRINK_MP)."""
    names = box.drop_duplicates("player_id").set_index("player_id")["name"]
    out = {}
    for pid, nm in names.items():
        rec = bpm.get(pav.norm_name(nm))
        out[pid] = R if rec is None else \
            R + (rec[0] - R) * rec[1] / (rec[1] + pav.BPM_SHRINK_MP)
    return out


def dated_value(hist, prior, m0=BPM_PRIOR_MP, use_prior=True):
    """(player_id, date) -> value above replacement per 48. use_prior: Bayes
    (m0 x prior + MP x obs) / (m0 + MP) - R; else this season only, shrunk
    to replacement: MP / (MP + m0) x (obs - R)."""
    memo = {}

    def v(pid, date):
        key = (pid, date)
        if key in memo:
            return memo[key]
        p0 = prior.get(pid, R) if use_prior else R
        out = p0 - R
        h = hist.get(pid)
        if h is not None:
            k = int(np.searchsorted(h[0], np.datetime64(pd.Timestamp(date)), "right")) - 1
            if k >= 0 and h[2][k] > 0:
                obs, mp = h[1][k], h[2][k]
                out = (m0 * p0 + mp * obs) / (m0 + mp) - R
        memo[key] = out
        return out
    return v


def prior_records(box, bpm):
    """player_id -> (last-season BPM, MP); unmatched (rookies) (R, 0)."""
    names = box.drop_duplicates("player_id").set_index("player_id")["name"]
    out = {}
    for pid, nm in names.items():
        rec = bpm.get(pav.norm_name(nm))
        out[pid] = (R, 0.0) if rec is None else (float(rec[0]), float(rec[1]))
    return out


def _to_date(h, date):
    """(BPM, MP) over games strictly before date from a bpm_history entry."""
    if h is None:
        return 0.0, 0.0
    k = int(np.searchsorted(h[0], np.datetime64(pd.Timestamp(date)), "right")) - 1
    return (float(h[1][k]), float(h[2][k])) if k >= 0 else (0.0, 0.0)


def pooled_bpm(b0, mp0, obs, mp, k=None):
    """Production's shrinkage extended by this season's minutes: (k R + mp0
    b0 + mp obs) / (k + mp0 + mp), k = BPM_SHRINK_MP. With mp = 0 it is
    production's value; a rookie (mp0 = 0) moves with his own minutes."""
    k = pav.BPM_SHRINK_MP if k is None else k
    return (k * R + mp0 * b0 + mp * obs) / (k + mp0 + mp)


def pooled_value(hist, records, frozen_from=None):
    """(player_id, date) -> value above replacement from pooled_bpm.
    frozen_from: players with at least this many last-season minutes keep
    production's frozen value; only those below it are updated."""
    memo = {}

    def v(pid, date):
        key = (pid, date)
        if key not in memo:
            b0, mp0 = records.get(pid, (R, 0.0))
            if frozen_from is not None and mp0 >= frozen_from:
                memo[key] = pooled_bpm(b0, mp0, 0.0, 0.0) - R
            else:
                obs, mp = _to_date(hist.get(pid), date)
                memo[key] = pooled_bpm(b0, mp0, obs, mp) - R
        return memo[key]
    return v


def season_terms(t, box, role, beta, cache_dir=pav.DEFAULT_CACHE):
    """{column: (team, date) -> talent} for season t: talent_bpmb (Bayes),
    talent_bpmo (this season only), talent_bpmp (minutes pooled, every
    player) and talent_bpmn (pooled for players under BPM_SHRINK_MP
    last-season minutes, frozen for the rest); plus a summary line."""
    if not os.path.exists(os.path.join(cache_dir, f"box_full_{t}.csv")):
        return {}, "no box lines"
    hist = bpm_history(ws.load_full(t, cache_dir), beta)
    table = pav.load_bpm(t - 1)
    prior, records = prior_bpm(box, table), prior_records(box, table)
    return ({"talent_bpmb": ws.dyn_talent_fn(box, dated_value(hist, prior), role),
             "talent_bpmo": ws.dyn_talent_fn(box, dated_value(hist, prior, use_prior=False),
                                             role),
             "talent_bpmp": ws.dyn_talent_fn(box, pooled_value(hist, records), role),
             "talent_bpmn": ws.dyn_talent_fn(
                 box, pooled_value(hist, records, frozen_from=pav.BPM_SHRINK_MP), role)},
            f"BPM to date for {len(hist)} players")


def small_sample(seasons, beta, cache_dir=pav.DEFAULT_CACHE, min_second=300.0):
    """Player level: does first-half BPM predict second-half BPM for players
    with little last-season data? Each season splits at its median game
    date; players with min_second+ second-half minutes; target = rebuilt BPM
    over the second half. Estimates from the first half: production's prior
    (frozen), pooled_bpm, and the first half alone. Grouped by last-season
    minutes. -> DataFrame of player rows."""
    rows = []
    for s in seasons:
        full = ws.load_full(s, cache_dir)
        mid = pd.Timestamp(full["date"].median()).normalize()
        first = player_rates(full[full["date"] < mid])
        second = player_rates(full[full["date"] >= mid])
        first["obs"] = design(first) @ beta
        second["target"] = design(second) @ beta
        table = pav.load_bpm(s - 1)
        m = first.merge(second[["player_id", "target", "minutes"]].rename(
            columns={"minutes": "mp2"}), on="player_id")
        m = m[m["mp2"] >= min_second]
        for r in m.itertuples(index=False):
            rec = table.get(pav.norm_name(r.name))
            b0, mp0 = (R, 0.0) if rec is None else (float(rec[0]), float(rec[1]))
            rows.append({"season": s, "player_id": r.player_id, "mp0": mp0,
                         "mp1": r.minutes, "mp2": r.mp2, "target": r.target,
                         "frozen": pooled_bpm(b0, mp0, 0.0, 0.0),
                         "pooled": pooled_bpm(b0, mp0, r.obs, r.minutes),
                         "season_only": r.obs})
    return pd.DataFrame(rows)


def small_report(d):
    """MAE (second-half minutes weighted) and corr of each estimate with the
    second-half BPM, by last-season minutes group."""
    lines = []
    groups = (("no last-season BPM (rookies, new)", d["mp0"] == 0),
              ("1-499 last-season MP", (d["mp0"] > 0) & (d["mp0"] < 500)),
              ("500-1,499", (d["mp0"] >= 500) & (d["mp0"] < 1500)),
              ("1,500+", d["mp0"] >= 1500))
    for label, ok in groups:
        x = d[ok]
        if len(x) < 20:
            continue
        w = x["mp2"].to_numpy(float)
        parts = []
        for c in ("frozen", "pooled", "season_only"):
            err = np.abs(x[c] - x["target"]).to_numpy(float)
            r = np.corrcoef(x[c], x["target"])[0, 1] if x[c].std() > 0 else np.nan
            parts.append(f"{c} MAE {(w * err).sum() / w.sum():.2f} corr "
                         + (f"{r:+.2f}" if np.isfinite(r) else "  n/a"))
        lines.append(f"  {label:<34} n {len(x):>4}  " + "; ".join(parts))
    return "\n".join(lines)


def check(fit, test, cache_dir=pav.DEFAULT_CACHE):
    """Fit on `fit`, score full-season rebuilt BPM against BBR on `test`."""
    beta, n = fit_beta(fit, cache_dir)
    print(f"fit on {fit}: {n} player-seasons; coefficients")
    print("  " + ", ".join(f"{c} {b:+.3f}" for c, b in zip(["const"] + FEATURES, beta)))
    for s in test:
        m = _matched(player_rates(ws.load_full(s, cache_dir)), pav.load_bpm(s))
        m["est"] = design(m) @ beta
        for lo in (250, 1000):
            x = m[m["minutes"] >= lo]
            print(f"  {s} (MP {lo}+, n = {len(x)}): corr {x['est'].corr(x['bbr_bpm']):.3f}, "
                  f"MAE {(x['est'] - x['bbr_bpm']).abs().mean():.2f}, mean diff "
                  f"{(x['est'] - x['bbr_bpm']).mean():+.2f}")
    return beta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--fit", nargs="+", default=["2016-2019", "2021-2024"])
    c.add_argument("--test", nargs="+", type=int, default=[2025, 2026])
    c.add_argument("--cache", default=pav.DEFAULT_CACHE)
    sm = sub.add_parser("small", help="first half vs second half by last-season minutes")
    sm.add_argument("--fit", nargs="+", default=["2016-2019", "2021-2024"])
    sm.add_argument("--seasons", nargs="+", default=["2017-2019", "2021-2026"])
    sm.add_argument("--cache", default=pav.DEFAULT_CACHE)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        check(nc.parse_years(a.fit), a.test, a.cache)
    else:
        beta, _ = fit_beta(nc.parse_years(a.fit), a.cache)
        d = small_sample(nc.parse_years(a.seasons), beta, a.cache)
        print("First half -> second half rebuilt BPM, players with 300+ second-half "
              "minutes (MAE weighted by second-half minutes)")
        print(small_report(d))
        held = d[d["season"] > max(nc.parse_years(a.fit))]
        if len(held):
            print("Seasons after the mapping's fit seasons only:")
            print(small_report(held))
    return 0


if __name__ == "__main__":
    sys.exit(main())
