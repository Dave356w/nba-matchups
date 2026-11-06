#!/usr/bin/env python3
"""Totals offence/defence weights (ODT) on ou2: the registered test (diagnostics/totals_od/REGISTRATION.md).

  python research/totals_od.py --frame            # features of every game -> frame.csv
  python research/totals_od.py --terms 2023 ...    # report terms per season -> terms_<t>.csv
  python research/totals_od.py --walk              # predictions and coefficients
  python research/totals_od.py --report            # RESULTS.txt from them
  (--run does all four stages in turn.)

Registered 2026-11-06 (commit 8301d50) before this code. Arms:
  base  ou2: T* (rel, form, b2b, phase on total - league total), then the
        report stage (av_o, av_d, av_m, last NBA report >= 300 min before
        tip, v11's Questionable weighting) on T*'s residual.
  ODT   base + off_t + off_t x phase in T*: off_t = expected possessions x
        both offences' points per 100 above the league, against
        league-average defence (the exact offensive part of rel).
New era (2016-17 .. today minus 2019-20 and 2020-21), walk-forward on era
seasons before each test season. ± is 95% unless labelled. Research only:
nothing here touches the ledgers, the pages or model/*.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402
import totals  # noqa: E402

OUT = ROOT / "diagnostics" / "totals_od"
LOAD = (2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026, 2027)  # inputs and priors
ERA = (2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026, 2027)               # fits and tests
TESTS = ERA[1:]
COMPLETED = tuple(t for t in TESTS if t <= 2026)
REPORT = (2023, 2024, 2025, 2026, 2027)
TERMS = ["av_o", "av_d", "av_m"]
BASE = list(totals.FEATURES)
ARMS = {"base": BASE, "ODT": BASE + ["off_t", "offt_phase"]}
TRIPWIRE = 10.0
Z90 = 1.645


class SeasonOD(totals.Season):
    """totals.Season whose features also carry off_t and def_t (half-life 25)."""

    def features(self, home, away, date):
        f = super().features(home, away, date)
        if f is None:
            return None
        lg = self.league(pd.Timestamp(date))
        (ph, oh, dh), _, _ = self.team(home, pd.Timestamp(date), totals.HALF_LIFE, lg)
        (pa, oa, da), _, _ = self.team(away, pd.Timestamp(date), totals.HALF_LIFE, lg)
        P, r = ph * pa / lg[0], lg[1]
        f["off_t"] = P * ((oh - r) + (oa - r)) / 100
        f["def_t"] = P * ((dh - r) + (da - r)) / 100
        f["cross_t"] = P * ((oh - r) * (da - r) + (oa - r) * (dh - r)) / r / 100
        return f


def frame(logs=None):
    """Every game of LOAD with ou2's features, off_t / def_t and the total."""
    real = totals.Season
    totals.Season = SeasonOD
    try:
        rows = totals.training_rows(logs or {y: nc.load_logs(y) for y in LOAD})
    finally:
        totals.Season = real
    rows["slate_date"] = rows["date"]
    rows["stage_gp"] = np.fmin(rows["gp_home"], rows["gp_away"])
    rows["offt_phase"] = rows["off_t"] * rows["phase"]
    return rows


def fit_t(tr, feats):
    """OLS of total - league total on feats (games 1+): coefficients."""
    tr = tr[tr["stage_gp"] >= 1]
    X = np.column_stack([np.ones(len(tr)), tr[feats].to_numpy(float)])
    return np.linalg.lstsq(X, (tr["total"] - tr["lg_total"]).to_numpy(float), rcond=None)[0]


def predict_t(g, feats, b):
    return g["lg_total"].to_numpy(float) + \
        np.column_stack([np.ones(len(g)), g[feats].to_numpy(float)]) @ b


def sigmas(res, early):
    late = res[~early].std()
    return (res[early].std() if early.sum() > 1 else late), late


def implied(feats, b):
    c = dict(zip(["intercept"] + feats, map(float, b)))
    out = dict(c)
    if "off_t" in c:
        for ph in (0.5, 1.0):
            k = c["off_t"] + c.get("offt_phase", 0.0) * ph
            out[f"off_share_{ph}"] = (c["rel"] + k) / (2 * c["rel"] + k)
    return out


# ------------------------------------------------------------ report terms --
def _parse(path):
    try:
        pav.parse_report(path)                     # writes its cached CSV
        return True
    except Exception:  # noqa: BLE001 - game_statuses reports and handles it
        return False


def preparse(t, workers=None):
    """Fetch season t's reports at the lead and parse the uncached ones on
    every core (parse_report caches each as CSV next to its PDF; about 2 s a
    report, one at a time, made a season about 20 minutes). The terms then
    read the same reports from the cache: the result is unchanged."""
    from concurrent.futures import ProcessPoolExecutor
    from datetime import timedelta
    import ledger
    box, tips = pav.fetch_box(t), pav.fetch_tips(t)
    cut = []
    for gid in box["game_id"].astype(str).unique():
        tip = ledger.parse_utc(tips.get(gid))
        if tip is not None:
            cut.append(tip.astimezone(pav.ET).replace(tzinfo=None)
                       - timedelta(minutes=totals.REPORT_LEAD))
    arch = pav.ReportArchive(pav.REPORT_DIR)
    arch.prefetch(cut)
    arch.save()
    paths = {arch.latest_before(c)[1] for c in set(cut)} - {None}
    todo = sorted(p for p in paths
                  if not os.path.exists(p[:-4] + f".v{pav.PARSER_VERSION}.csv"))
    log(f"terms {t}: {len(paths)} reports, {len(todo)} to parse on {workers or os.cpu_count()} cores")
    with ProcessPoolExecutor(workers) as ex:
        ok = sum(ex.map(_parse, todo, chunksize=4))
    log(f"terms {t}: parsed {ok} of {len(todo)}")


def season_terms(t):
    """{(date, team): {av_o, av_d, av_m}} for report season t (0 where the
    report does not cover the game), as research/totals_report.py."""
    import totals_report as trp
    preparse(t)
    model = json.load(open(ROOT / "model" / "logit_avail.json"))
    rates = pav.lead_rates(model["q_rates"], totals.REPORT_LEAD)
    terms, _ = trp.report_terms(t, totals.REPORT_LEAD, rates)
    return pd.DataFrame([dict(slate_date=d, team=tm, **{k: v[k] for k in TERMS})
                         for (d, tm), v in terms.items()])


def attach_terms(rows, terms):
    """rows with av_o / av_d / av_m, both teams' sums (0 when missing)."""
    rows = rows.copy()
    for k in TERMS:
        rows[k] = 0.0
    if not len(terms):
        return rows
    T = terms.set_index(["slate_date", "team"])
    for k in TERMS:
        h = T[k].reindex(pd.MultiIndex.from_arrays([rows["slate_date"], rows["home"]])).to_numpy(float)
        a = T[k].reindex(pd.MultiIndex.from_arrays([rows["slate_date"], rows["away"]])).to_numpy(float)
        rows[k] = np.nan_to_num(h + a)
    return rows


# -------------------------------------------------------------------- run --
def walk(rows, arms=None):
    """Per arm and test season: T* walk-forward on era seasons before Y, then
    the report stage on the report seasons before Y (on each arm's own
    out-of-sample T* residual). Returns predictions and coefficients."""
    rows = rows[rows["season"].isin(ERA)].copy()
    early_all = (rows["stage_gp"] < 10).to_numpy()
    coefs = []
    arms = arms or ARMS
    for arm, feats in arms.items():
        rows[f"Ts_{arm}"] = np.nan
        rows[f"sig_{arm}"] = np.nan
        for Y in TESTS:
            tr = rows[rows["season"] < Y]
            b = fit_t(tr, feats)
            trg = tr[tr["stage_gp"] >= 1]
            res = (trg["total"] - predict_t(trg, feats, b)).to_numpy(float)
            se, sl = sigmas(res, (trg["stage_gp"] < 10).to_numpy())
            m = (rows["season"] == Y).to_numpy()
            rows.loc[m, f"Ts_{arm}"] = predict_t(rows[m], feats, b)
            rows.loc[m, f"sig_{arm}"] = np.where(early_all[m], se, sl)
            coefs.append(dict(arm=arm, season=Y, stage="T*", sigma_late=float(sl),
                              **implied(feats, b)))
        rows[f"T_{arm}"] = rows[f"Ts_{arm}"]
        res_all = (rows["total"] - rows[f"Ts_{arm}"]).to_numpy(float)
        for Y in TESTS:
            rep = [t for t in REPORT if t < Y]
            if not rep:
                continue
            tr = (rows["season"].isin(rep) & (rows["stage_gp"] >= 1)).to_numpy() & np.isfinite(res_all)
            X = np.column_stack([np.ones(tr.sum()), rows.loc[tr, TERMS].to_numpy(float)])
            b = np.linalg.lstsq(X, res_all[tr], rcond=None)[0]
            r2 = res_all[tr] - X @ b
            se, sl = sigmas(r2, early_all[tr])
            m = (rows["season"] == Y).to_numpy()
            rows.loc[m, f"T_{arm}"] = rows.loc[m, f"Ts_{arm}"].to_numpy() + \
                np.column_stack([np.ones(m.sum()), rows.loc[m, TERMS].to_numpy(float)]) @ b
            rows.loc[m, f"sig_{arm}"] = np.where(early_all[m], se, sl)
            coefs.append(dict(arm=arm, season=Y, stage="report", sigma_late=float(sl),
                              **dict(zip(["intercept"] + TERMS, map(float, b)))))
    keep = ["season", "slate_date", "home", "away", "stage_gp", "total", "lg_total",
            "off_t", "def_t"] + [f"{p}_{a}" for a in arms for p in ("Ts", "T", "sig")]
    return rows[rows["season"].isin(TESTS)][keep], coefs


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def stage_frame():
    import time
    OUT.mkdir(parents=True, exist_ok=True)
    logs, t0 = {}, time.time()
    for y in LOAD:
        logs[y] = nc.load_logs(y)
        log(f"logs {y}: {len(logs[y])} teams ({time.time() - t0:.0f} s)")
    rows = frame(logs)
    log(f"features built ({time.time() - t0:.0f} s)")
    rows.to_csv(OUT / "frame.csv", index=False)
    log(f"frame: {len(rows)} games, seasons {sorted(rows['season'].unique())}")


def stage_terms(seasons):
    OUT.mkdir(parents=True, exist_ok=True)
    import time
    for t in seasons:
        t0 = time.time()
        log(f"terms {t}: fetching and reading reports ...")
        d = season_terms(t)
        log(f"terms {t}: done in {time.time() - t0:.0f} s")
        d.to_csv(OUT / f"terms_{t}.csv", index=False)
        log(f"terms {t}: {len(d)} team-games, covered (any term non-zero) "
            f"{(d[TERMS].abs().sum(axis=1) > 0).mean() if len(d) else 0:.1%}")


def stage_walk():
    rows = pd.read_csv(OUT / "frame.csv")
    parts = [pd.read_csv(OUT / f"terms_{t}.csv") for t in REPORT if (OUT / f"terms_{t}.csv").exists()]
    missing = [t for t in REPORT if not (OUT / f"terms_{t}.csv").exists()]
    if missing:
        raise SystemExit(f"report terms missing for {missing}: run --terms first")
    rows = attach_terms(rows, pd.concat(parts, ignore_index=True))
    P, C = walk(rows)
    P.to_csv(OUT / "predictions.csv", index=False, float_format="%.5g")
    (OUT / "coefs.json").write_text(json.dumps(C, indent=1))
    log(f"walk: {len(P)} predictions")


# ----------------------------------------------------------------- report --
def phi(x):
    return 0.5 * (1 + erf(float(x) / sqrt(2)))


def ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(se), len(d)


def sq_lines(P, title):
    L = [f"== {title}: squared error on the total, ODT - base (points², negative = ODT better)"]
    groups = [("games 1+", P[P["stage_gp"] >= 1]),
              ("games 1-9", P[(P["stage_gp"] >= 1) & (P["stage_gp"] <= 9)]),
              ("games 10+", P[P["stage_gp"] >= 10])]
    for label, g in groups:
        if not len(g):
            continue
        e = lambda a: (g["total"] - g[f"T_{a}"]) ** 2  # noqa: E731
        m, se, n = ci(e("ODT") - e("base"))
        wins = sum(((s["total"] - s["T_ODT"]) ** 2 - (s["total"] - s["T_base"]) ** 2).mean() < 0
                   for _, s in g.groupby("season"))
        L.append(f"  {label:10s} n {n:5d}  RMSE base {np.sqrt(e('base').mean()):.3f}  ODT "
                 f"{np.sqrt(e('ODT').mean()):.3f}  Δ {m:+.3f} ± {1.96 * se:.3f} (95%)  one-sided p "
                 f"{phi(m / se):.3f}  better in {wins} of {g['season'].nunique()}")
    L.append("  By season, games 1+:")
    for s, g in P[P["stage_gp"] >= 1].groupby("season"):
        m, se, n = ci((g["total"] - g["T_ODT"]) ** 2 - (g["total"] - g["T_base"]) ** 2)
        L.append(f"    {int(s)}  n {n:5d}  Δ {m:+.3f} ± {1.96 * se:.3f}   base bias "
                 f"{(g['T_base'] - g['total']).mean():+.2f}  ODT bias {(g['T_ODT'] - g['total']).mean():+.2f}")
    return L


def market(P):
    """Book rows (reconstructed, main lines) with each arm's P(over) at open and close."""
    import totals_model as tm
    x = P.rename(columns={"T_base": "T", "sig_base": "sigma"})
    j = tm.priced(x)
    out = []
    for when in ("open", "close"):
        line = j[f"{when}_total"].to_numpy(float)
        o, u = j[f"{when}_over"].to_numpy(float), j[f"{when}_under"].to_numpy(float)
        ok = tm.main_line(line, o, u) & (j["total"].to_numpy(float) != line)
        q = np.where(ok, tm._q_over(np.where(ok, o, -110), np.where(ok, u, -110)), np.nan)
        d = j.assign(when=when, line=line, q=q, ok=ok)
        for a in ARMS:
            T = d["T" if a == "base" else "T_ODT"].to_numpy(float)
            s = d["sigma" if a == "base" else "sig_ODT"].to_numpy(float)
            d[f"p_{a}"] = 1 - np.array([phi(v) for v in (line - T) / s])
        out.append(d[d["ok"]])
    return pd.concat(out, ignore_index=True)


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def market_lines(P):
    L = ["", "== Against the book, P(over) log loss (reconstructed rows, main lines; ± 95%)",
         "   arm - no-vig market; [ODT - base]"]
    M = market(P)
    gate = {}
    for (when, bk, s), g in M.groupby(["when", "book", "season"]):
        for st, h in (("games 10+", g[g["stage_gp"] >= 10]),
                      ("games 1-9", g[(g["stage_gp"] >= 1) & (g["stage_gp"] <= 9)])):
            if len(h) < 60:
                continue
            y = (h["total"] > h["line"]).to_numpy(float)
            lq = ll(h["q"], y)
            cells = []
            for a in ARMS:
                m, se, _ = ci(ll(h[f"p_{a}"], y) - lq)
                cells.append(f"{a} {m:+.4f} ± {1.96 * se:.4f}")
            m, se, _ = ci(ll(h["p_ODT"], y) - ll(h["p_base"], y))
            L.append(f"  {when:5s} {bk:7s} {int(s)} {st:9s} n {len(h):4d}  " + "  ".join(cells)
                     + f"  [{m:+.4f} ± {1.96 * se:.4f}]")
            if when == "close" and st == "games 10+":
                gate[(bk, int(s))] = m
    return L, gate


def coef_lines(C):
    L = ["", "== ODT fitted weights by fold (T* stage; off_share 0.5 = production's equal split)"]
    for d in C[(C["arm"] == "ODT") & (C["stage"] == "T*")].to_dict("records"):
        L.append(f"    {int(d['season'])}  rel {d['rel']:+.3f}  off_t {d['off_t']:+.3f}  offt_phase "
                 f"{d['offt_phase']:+.3f}  off_share 0.5 {d['off_share_0.5']:.3f}  1.0 "
                 f"{d['off_share_1.0']:.3f}")
    L.append("  base T* rel slope by fold: " + "  ".join(
        f"{int(d['season'])} {d['rel']:+.3f}" for d in
        C[(C["arm"] == "base") & (C["stage"] == "T*")].to_dict("records")))
    return L


def gate_lines(P, gate):
    G = P[(P["stage_gp"] >= 1) & P["season"].isin(COMPLETED)]
    d = (G["total"] - G["T_ODT"]) ** 2 - (G["total"] - G["T_base"]) ** 2
    m, se, _ = ci(d)
    per = {int(s): float(((g["total"] - g["T_ODT"]) ** 2 - (g["total"] - g["T_base"]) ** 2).mean())
           for s, g in G.groupby("season")}
    wins = sum(v < 0 for v in per.values())
    c1 = m + Z90 * se < 0 and wins >= 5
    need = [("dk", 2026), ("espnbet", 2025)]
    c2 = all(k in gate and gate[k] <= 0 for k in need)
    gain = max(-v for v in per.values())
    c3 = gain <= TRIPWIRE
    ok = c1 and c2 and c3
    return ["", "== Decision rule (registered 2026-11-06, commit 8301d50)",
            f"  1. squared error, games 1+, completed seasons: {m:+.3f} (90% upper bound "
            f"{m + Z90 * se:+.3f}; one-sided p {phi(m / se):.3f}), better in {wins} of {len(per)}  -> "
            f"{'pass' if c1 else 'FAIL'}",
            "  2. P(over) log loss at the close, games 10+, ODT - base: " + "; ".join(
                f"{b} {s} {gate[(b, s)]:+.4f}" if (b, s) in gate else f"{b} {s} missing"
                for b, s in need) + f"  -> {'pass' if c2 else 'FAIL'}",
            f"  3. largest season gain {gain:.2f} points² (tripwire {TRIPWIRE:.0f})  -> "
            f"{'pass' if c3 else 'FAIL'}",
            f"  Power: SE {se:.3f}; minimum effect detectable with 80% power (one-sided 5%) "
            f"{2.49 * se:.2f} points²",
            f"  ODT: {'CANDIDATE (the owner decides)' if ok else 'NOT a candidate'}"]


def report():
    P = pd.read_csv(OUT / "predictions.csv")
    C = pd.DataFrame(json.loads((OUT / "coefs.json").read_text()))
    done = P[P["season"].isin(COMPLETED)]
    live = P[~P["season"].isin(COMPLETED)]
    L = [__doc__.strip().splitlines()[0], "",
         f"Completed test seasons {sorted(int(x) for x in done['season'].unique())}: "
         f"{int((done['stage_gp'] >= 1).sum())} games 1+.", ""]
    L += sq_lines(done, "New era, completed seasons (the rule's evidence)")
    if len(live):
        L += [""] + sq_lines(live, "2026-27 to date (apart, not counted)")
    m, gate = market_lines(P)
    L += m + gate_lines(P, gate) + coef_lines(C)
    (OUT / "RESULTS.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", action="store_true")
    ap.add_argument("--terms", nargs="+", type=int)
    ap.add_argument("--walk", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--run", action="store_true", help="all four stages")
    a = ap.parse_args()
    if a.frame or a.run:
        stage_frame()
    if a.terms or a.run:
        stage_terms(a.terms or REPORT)
    if a.walk or a.run:
        stage_walk()
    if a.report or a.run:
        report()


if __name__ == "__main__":
    main()
