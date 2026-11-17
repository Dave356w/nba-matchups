#!/usr/bin/env python3
"""2026-27 losses: which feature weights does the season's evidence disagree with?

  python research/loss_feature_audit.py --build    # features (fetches, caches)
  python research/loss_feature_audit.py --report

Owner request (2026-11-17): analyze 2026-27 model losses for trends in
feature weighting errors. Research only: nothing here touches the ledgers,
the pages or model/*.json, and no reading selects a rule or a refit.

Features: every 2026-27 game rebuilt with production's v11 inputs
(nc.fade_v8_games: faded rating delta, b2b_net, d_phase, luck_def,
talent_diff, ft_diff; games strictly before each date) and the report
route's av_min / av_bpm (pav.season_rows: last NBA report >= 30 min before
tip, od rule, production matcher). These are hindsight reconstructions of
the inputs, not the frozen lock values; the lock P is read from the native
ledger and is never rescored.

Population: analysis.model_predictions (native v11 five-hour locks, voids
out) with a result, joined to the rebuilt features by date x home x away.

Readings (± 95%):
  1. per-feature contribution (coef x value, in home-margin points, from
     the fitted file of the route the lock used) in losses vs wins of the
     model's pick, signed toward the pick.
  2. weight-error regression: actual margin minus the lock's model margin
     (sigma x Phi^-1(lock P)) on the features; a coefficient is the change
     to that feature's weight this season's results would ask for.
  3. the same against the market: actual minus the market's margin
     (from the lock no-vig q at the route's sigma), and model minus market
     on the features (where the model's disagreement came from).
  4. benchmark: the same regression (2) on 2024-25 and 2025-26 games of
     the same stage window with the production base fit, so a 2026-27
     "error" can be compared with the usual early-season noise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from statistics import NormalDist

_N = NormalDist()


class norm:  # noqa: N801 - scipy-like shim (scipy is not a requirement)
    ppf = staticmethod(np.vectorize(_N.inv_cdf))
    cdf = staticmethod(np.vectorize(_N.cdf))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import analysis                    # noqa: E402
import ledger                      # noqa: E402
import nba_composite as nc         # noqa: E402
import player_availability as pav  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "loss_feature_audit")
CACHE = os.path.join(pav.DEFAULT_CACHE, "loss_feature_audit")
SEASON, TODAY = 2027, "2026-11-17"
BASE = ["delta", "b2b_net", "d_phase", "luck_def", "talent_diff", "ft_diff"]
AV = ["av_min", "av_bpm"]


def load_model(name):
    return json.load(open(os.path.join(ROOT, "model", name)))


def build():
    os.makedirs(CACHE, exist_ok=True)
    weights = nc.load_json("weights.json")
    avail = load_model("logit_avail.json")
    # a scratch copy of data/ so incremental box updates never touch data/
    scratch = os.path.join(CACHE, "data")
    os.makedirs(scratch, exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "data")):
        if f.startswith(("nba_box_2027", "nba_box_pre_2027")):
            shutil.copy(os.path.join(ROOT, "data", f), scratch)
    box = pav.update_box(SEASON, TODAY, scratch)
    bpm = pav.load_bpm(SEASON - 1)
    rookies = pav.rookies_for(SEASON, avail.get("rookies"))
    value, _, _ = pav.player_values(box, bpm, rookies)
    talent = pav.talent_fn(box, value, pav.arrival_roles(box, bpm))
    talent0 = pav.roster_talent(pav.update_preseason_box(SEASON, TODAY, scratch), bpm)
    prior = nc.load_logs(SEASON - 1)
    g = nc.fade_v8_games(SEASON, weights, talent=talent, talent0=talent0,
                         prior_logs=prior)
    g.to_csv(os.path.join(CACHE, "games_2027.csv"), index=False)
    led = ledger.load(ledger.NATIVE_PATH)
    tips = dict(zip(led["game_id"].astype(str), led["tip_utc"].astype(str)))
    archive = pav.ReportArchive(os.path.join(CACHE, "reports"))
    try:
        rows = pav.season_rows(SEASON, TODAY, weights, box, value, talent, talent0,
                               tips, archive, prior, split=pav.splits_suffix(avail))
    finally:
        archive.save()
    rows.to_csv(os.path.join(CACHE, "avail_2027.csv"), index=False)
    # benchmark seasons: base features only (no report needed)
    for y in (2025, 2026):
        b = nc.fade_v8_games(y, weights, talent=None, talent0=None)
        b.to_csv(os.path.join(CACHE, f"games_{y}.csv"), index=False)
    print("built", len(g), "games;", len(rows), "report-covered rows")


# ------------------------------------------------------------------ report --
def ols(X, y):
    """OLS with intercept: coef, 95% half-widths (HC1 robust)."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    V = XtXi @ (X.T * e ** 2) @ X @ XtXi * n / max(n - k, 1)
    return b, 1.96 * np.sqrt(np.diag(V))


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def population():
    led = ledger.load(ledger.NATIVE_PATH)
    p = analysis.model_predictions(led)
    p = p[pd.to_numeric(p["prediction_won"], errors="coerce").isin([0, 1])].copy()
    p = p[p["prediction_stage"].isin(["games 1-9", "games 10+"])
          | pd.to_numeric(np.minimum(p["gp_home"].astype(float),
                                     p["gp_away"].astype(float))).ge(1)]
    g = pd.read_csv(os.path.join(CACHE, "games_2027.csv"))
    a = pd.read_csv(os.path.join(CACHE, "avail_2027.csv"))
    for d in (g, a):
        d["slate_date"] = pd.to_datetime(d["date"] if "date" in d else d["slate_date"]
                                         ).dt.strftime("%Y-%m-%d")
    p["slate_date"] = p["slate_date"].astype(str)
    p = p.rename(columns={"delta": "ledger_delta"})
    m = p.merge(g[["slate_date", "home", "away"] + BASE], how="left",
                on=["slate_date", "home", "away"])
    m = m.merge(a[["slate_date", "home", "away"] + AV], how="left",
                on=["slate_date", "home", "away"])
    m["margin"] = pd.to_numeric(m["home_pts"]) - pd.to_numeric(m["away_pts"])
    m["route_avail"] = m["lock_model_tag"].astype(str).str.contains("avail")
    m["gp"] = np.minimum(m["gp_home"].astype(float), m["gp_away"].astype(float))
    m = m[m["gp"] >= 1]
    return m


def contributions(m):
    base, avail = load_model("logit.json"), load_model("logit_avail.json")
    rows = []
    for _, r in m.iterrows():
        mod = avail if r["route_avail"] else base
        c = {f: w * (r.get(f, np.nan) if f not in AV or np.isfinite(r.get(f, np.nan))
                     else 0.0)
             for f, w in zip(mod["features"], mod["coef"])}
        c["intercept"] = mod["intercept"]
        c["sigma"] = mod["sigma"]
        rows.append(c)
    return pd.DataFrame(rows, index=m.index)


def report():
    m = population()
    C = contributions(m)
    feats = BASE + AV
    sign = np.where(m["lock_p_home"].astype(float) >= .5, 1.0, -1.0)
    sigma = C["sigma"].to_numpy(float)
    P = m["lock_p_home"].astype(float).clip(1e-4, 1 - 1e-4)
    q = m["lock_q_home"].astype(float)
    m["model_margin"] = sigma * norm.ppf(P)
    m["mkt_margin"] = sigma * norm.ppf(q.clip(1e-4, 1 - 1e-4))
    have = m[BASE].notna().all(axis=1)
    L = []
    w = L.append
    w("2026-27 losses: feature weighting audit (research/loss_feature_audit.py).")
    w("Research only; ± is 95%. Features are hindsight rebuilds of v11's inputs;")
    w("lock P, prices and results are the native ledger's (voids out). Contributions")
    w("use each route's fitted file (the report route is refitted per slate in")
    w("production, so its contributions here are approximate).")
    w("")
    n_all = len(m)
    w(f"Population: {n_all} graded native v11 locks, games 1+; features rebuilt "
      f"for {int(have.sum())}; report-route locks {int(m['route_avail'].sum())}, "
      f"with av terms rebuilt {int(m[AV].notna().all(axis=1).sum())}.")
    rec = m.loc[have].copy()
    # reproduction check of the base route: rebuilt P vs lock P
    base = load_model("logit.json")
    bb = rec[~rec["route_avail"]]
    if len(bb):
        mb = base["intercept"] + bb[base["features"]].to_numpy(float) @ np.array(base["coef"])
        pb = norm.cdf(mb / base["sigma"])
        d = np.abs(pb - bb["lock_p_home"].astype(float))
        w(f"Base-route reproduction: {int((d < .005).sum())} of {len(bb)} within 0.005 "
          f"of the lock P (median |diff| {np.median(d):.4f}).")
    w("")
    for stage, sub in (("games 1-9", rec[rec["gp"] < 10]),
                       ("games 10+", rec[rec["gp"] >= 10]),
                       ("games 1+", rec)):
        s = sign[sub.index.map(m.index.get_loc)]
        won = sub["prediction_won"].astype(int).to_numpy()
        w(f"== {stage}: n={len(sub)}, model picks {won.sum()}-{len(won) - won.sum()}")
        w("  1. contribution toward the pick (points of home margin x pick side)")
        w(f"     {'feature':12s} {'wins':>15s} {'losses':>15s} {'loss-win':>16s}")
        for f in feats + ["intercept"]:
            if f not in C:
                continue
            x = C.loc[sub.index, f].to_numpy(float) * s
            mw, cw = mean_ci(x[won == 1])
            ml, cl = mean_ci(x[won == 0])
            dd = ml - mw
            se = np.sqrt((cw / 1.96) ** 2 + (cl / 1.96) ** 2) * 1.96
            w(f"     {f:12s} {mw:+6.2f} ± {cw:4.2f}  {ml:+6.2f} ± {cl:4.2f}  "
              f"{dd:+6.2f} ± {se:4.2f}")
        y = sub["margin"].to_numpy(float)
        for label, tgt in (("2. actual − model margin", y - sub["model_margin"]),
                           ("3a. actual − market margin", y - sub["mkt_margin"]),
                           ("3b. model − market margin",
                            sub["model_margin"] - sub["mkt_margin"])):
            ok = np.isfinite(tgt.to_numpy(float))
            if ok.sum() < 30:
                w(f"  {label}: too few games ({ok.sum()})")
                continue
            b, ci = ols(sub.loc[ok, BASE].to_numpy(float), tgt.to_numpy(float)[ok])
            w(f"  {label} on base features (n={ok.sum()}; weight change per unit, "
              f"and as a share of the base coefficient)")
            w(f"     intercept {b[0]:+6.2f} ± {ci[0]:5.2f}")
            for f, bi, ci_ in zip(BASE, b[1:], ci[1:]):
                k = base["coef"][base["features"].index(f)]
                sd = sub[f].std()
                w(f"     {f:12s} {bi:+7.3f} ± {ci_:6.3f}  ({bi / k:+5.0%} of {k:.3f}; "
                  f"1 SD of feature = {bi * sd:+5.2f} pts)")
        w("")
    # one feature at a time (collinearity: delta, d_phase, talent_diff and
    # luck_def correlate |r| 0.5-0.96, so the joint fits above are unstable)
    av = load_model("logit_avail.json")
    w("== one feature at a time, games 1+: residual on that feature's contribution")
    w("   (coef x value, report-route file); slope = extra multiple of its weight")
    y = rec["margin"].to_numpy(float)
    for f, c in zip(av["features"], av["coef"]):
        x = rec[f].fillna(0).to_numpy(float) * c
        parts = []
        for nm, tgt in (("vs results", y - rec["model_margin"]),
                        ("vs market", rec["mkt_margin"] - rec["model_margin"])):
            b, ci = ols(x[:, None], tgt.to_numpy(float))
            parts.append(f"{nm} x{b[1]:+5.2f} ± {ci[1]:4.2f}")
        w(f"  {f:12s} " + "   ".join(parts))
    b, ci = ols(rec[["model_margin"]].to_numpy(float), y - rec["model_margin"])
    w(f"  whole model margin: results ask x{b[1]:+.2f} ± {ci[1]:.2f} (dispersion)")
    hw = rec["home_won"].astype(float).to_numpy()

    def ll(pp):
        pp = np.clip(pp, 1e-6, 1 - 1e-6)
        return -np.mean(hw * np.log(pp) + (1 - hw) * np.log(1 - pp))
    t = rec["talent_diff"].to_numpy(float) * av["coef"][av["features"].index("talent_diff")]
    sg = av["sigma"]
    w("  in-sample log loss (talent multiple chosen after seeing results; not evidence):")
    w("    " + "  ".join(f"talent x{k}: {ll(norm.cdf((rec['model_margin'] + (k - 1) * t) / sg)):.4f}"
                         for k in (1, 1.5, 2))
      + f"  | lock market: {ll(rec['lock_q_home'].astype(float).to_numpy()):.4f}")
    w("")
    # model's net feature lean in its misses: residual by route
    w("== residual by route (actual − model margin, mean ± 95%)")
    for name, sub in rec.groupby("route_avail"):
        r = (sub["margin"] - sub["model_margin"]).to_numpy(float)
        mu, ci = mean_ci(r)
        w(f"  {'report' if name else 'base':7s} n={len(sub):4d}  {mu:+5.2f} ± {ci:4.2f}  "
          f"(|resid| {np.abs(r).mean():.1f})")
    w("")
    # benchmark seasons
    w("== benchmark: same regression (2) with the production base fit, prior seasons")
    w("   (in-sample for the fit: those seasons are in its training data, so these")
    w("   show the usual spread of a season's weight 'errors' at this sample size,")
    w("   not an out-of-sample test)")
    for y in (2025, 2026):
        b0 = pd.read_csv(os.path.join(CACHE, f"games_{y}.csv"))
        b0["gp"] = np.minimum(b0["gp_home"], b0["gp_away"])
        b0 = b0[(b0["gp"] >= 1)].dropna(subset=[f for f in BASE if f != "talent_diff"])
        # benchmark without talent (needs per-season box values): drop it
        fs = [f for f in BASE if f != "talent_diff"]
        win = b0.sort_values("date").groupby(["home"]).head(10**6)
        early = win[win["gp"] <= 15]
        mm = base["intercept"] + early[fs].to_numpy(float) @ np.array(
            [base["coef"][base["features"].index(f)] for f in fs])
        bb_, ci_ = ols(early[fs].to_numpy(float), early["margin"].to_numpy(float) - mm)
        w(f"  {y - 1}-{str(y)[2:]} games 1-15 (n={len(early)}), talent omitted:")
        for f, bi, c in zip(fs, bb_[1:], ci_[1:]):
            k = base["coef"][base["features"].index(f)]
            w(f"     {f:12s} {bi:+7.3f} ± {c:6.3f}  ({bi / k:+5.0%})")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "RESULTS.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.report:
        report()


if __name__ == "__main__":
    main()
