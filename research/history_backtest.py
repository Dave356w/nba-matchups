#!/usr/bin/env python3
"""History backtest: the current model against outcomes only (no market).

  python research/history_backtest.py [--seasons 2017 ... 2026] [--era 2005-2014]

Owner's question (2026-10-21): market prices exist for two seasons only, but
the model's logic and calibration can be checked against results on every
season the game logs reach. Parts A and B are games 10+; C is games 1-9:

  A. Walk-forward, the production v6 routing (diagnostics/v5_audit folds:
     every fit on seasons strictly before the test season; availability on
     covered games, base otherwise, the v4 fallback when a term is missing),
     2016-17 to 2025-26. Per season and pooled: log loss and Brier against
     simpler arms fitted the same way (home edge only; delta + b2b, the v1
     core; delta + b2b + delta*phase, v4); calibration slope and intercept,
     overall, by season phase and by route; reliability of the favourite's
     side; predicted vs actual home win %; back-to-back games; and each v6
     term dropped in turn (refit without it), the out-of-sample worth of
     every piece of the model's logic.
  B. Out of era, 2004-05 to 2013-14 (--era): seasons no fit or design choice
     has seen. The frozen core as committed (model/weights.json for delta,
     model/logit_v4.json for the logit: delta, b2b, delta*phase; the
     production fallback path), beside the same logit refitted walk-forward
     on the era's earlier seasons (committed weights) and the full refit
     (composite weights and logit both from the era's earlier seasons); and
     B2, the committed four-factor weights beside weights fitted on the era.
     Box scores and injury reports are not cached that far back, so talent,
     luck, FT% and availability are not tested there.
  A8 / B. In-season home edge (2026-10-22, docs/model_guidelines.md W8): the
     intercept is the training seasons' home edge. Each game's logit is
     shifted by this season's own residual on games 10+ strictly before its
     date, b = sum(y - p) / (sum p(1-p) + 1/tau^2) (a Gaussian prior of sd
     tau on the season's offset). tau is set walk-forward, empirical Bayes
     from the earlier test seasons' season-level offsets (none before two
     earlier seasons); B applies the modern seasons' tau to the frozen core
     out of era. Priced seasons are also scored against the close.
  C. Games 1-9 (2026-10-22, W5): the carryover model (cold_start.py),
     walk-forward with each fold's weights and an early logit fitted on the
     earlier seasons (production fits 2023-26 only), 2016-17 to 2025-26; and
     out of era the frozen carryover logit (logit_early_v6.json since v7;
     no box scores there for talent) with the committed weights.
     Calibration and the early home lean against outcomes.
  A9. Rookies by draft slot (2026-10-22): talent_diff values players
     missing from last season's BPM table (rookies) at replacement. A9 gives
     each drafted rookie the rookie-season BPM his pick predicts, a + b
     log(pick), minutes-weighted over earlier seasons' rookies only (BBR
     draft pages), and refits v6 with that talent in place of talent_diff;
     also against the close on the priced seasons.

Research only: nothing here touches the ledgers, the pages or the model.
Writes diagnostics/history_backtest/RESULTS.txt. ± is 95%.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "v5_audit"))
import cold_start                                 # noqa: E402
import common as c                                # noqa: E402
import nba_composite as nc                        # noqa: E402
import player_availability as pav                 # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "history_backtest")
Z95 = 1.96
SEASONS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
V4 = c.V4                                         # delta, b2b_net, d_phase
V1 = ["delta", "b2b_net"]
V6 = c.V5 + ["ftp_gap"]                           # production base logit terms
AV6 = c.AV5 + ["ftp_gap"]                         # production availability terms
TERMS = (("b2b_net", "back-to-back"), ("d_phase", "delta x season phase"),
         ("luck_def", "opponent 3P% luck"), ("talent_diff", "roster talent"),
         ("ftp_gap", "own FT%"))
FAV_BINS = np.round(np.arange(0.50, 1.0001, 0.05), 2)
PHASES = ((0.0, 1 / 3, "opening third"), (1 / 3, 2 / 3, "middle third"),
          (2 / 3, 1.01, "last third"))
TAU_GRID = (0.05, 0.10, 0.20)          # A8 sensitivity, logit sd of a season's offset
EARLY_TRAIN = [2016, 2017, 2018, 2019] + list(range(2021, 2027))


def season_name(y):
    return f"{int(y) - 1}-{str(int(y))[-2:]}"


def mean_ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    return float(d.mean()), float(Z95 * d.std(ddof=1) / np.sqrt(len(d)))


def ll(p, y):
    return c.ll(np.asarray(p, float), np.asarray(y, float))


def brier(p, y):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def calib(p, y):
    """(slope, ±95, intercept, ±95) of y on logit(p); 1 / 0 = calibrated."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    k = np.isfinite(p)
    if k.sum() < 50:
        return (np.nan,) * 4
    m = c.fit(c.logit(p[k])[:, None], y[k], l2=0.0)
    se = np.sqrt(np.diag(m["cov"]))
    return m["coef"][0], Z95 * se[1], m["intercept"], Z95 * se[0]


def fitpred(train, feats, test):
    """Fit a logit on the finite rows of `train`; predict `test` (NaN where a
    feature is missing)."""
    tr = train[c.finite(train, feats)]
    m = c.fit(tr[feats].to_numpy(float), tr["win"].to_numpy(float))
    return c.predict(m, test[feats].to_numpy(float))


def routed(f, base_feats, avail_feats, use_avail=True):
    """P for every test game of fold `f` under the production routing with
    the given term sets; a game whose terms are missing falls back to the
    v4 core, as production does."""
    t = f.test
    r = t["route"].to_numpy()
    p = np.full(len(t), np.nan)
    avail = (r == "avail") & bool(use_avail and f.avail_train is not None)
    base = (r == "base") | ((r == "avail") & ~avail)
    if avail.any():
        p[avail] = fitpred(f.avail_train, avail_feats, t[avail])
    if base.any():
        p[base] = fitpred(f.base_train, base_feats, t[base])
    miss = ~np.isfinite(p)
    if miss.any():
        p[miss] = fitpred(f.v4_train, V4, t[miss])
    return p


def arms(f):
    """{arm: P} for one walk-forward fold."""
    t = f.test
    home = float(f.v4_train["win"].mean())
    out = {"home edge only": np.full(len(t), home),
           "v1 core (delta, b2b)": fitpred(f.v4_train, V1, t),
           "v4 core (+ delta*phase)": fitpred(f.v4_train, V4, t),
           "v6 (production)": routed(f, V6, AV6)}
    for term, _ in TERMS:
        out[f"v6 - {term}"] = routed(f, [x for x in V6 if x != term],
                                     [x for x in AV6 if x != term])
    out["v6 - availability"] = routed(f, V6, AV6, use_avail=False)
    return out


def paired(a, b, y):
    """Mean log loss difference a - b and its ±95 on the rows both cover."""
    k = np.isfinite(a) & np.isfinite(b)
    return mean_ci(ll(a[k], y[k]) - ll(b[k], y[k]))


def part_a(folds):
    L = ["== A. Walk-forward, production v6 routing, games 10+ (every fit on "
         "earlier seasons only)"]
    rows = []
    for y, f in sorted(folds.items()):
        a = arms(f)
        t = f.test.assign(**{f"p::{k}": v for k, v in a.items()})
        rows.append(t)
    d = pd.concat(rows, ignore_index=True)
    y = d["win"].to_numpy(float)
    P = {k[3:]: d[k].to_numpy(float) for k in d.columns if k.startswith("p::")}
    v6 = P["v6 (production)"]

    L.append("  A1. Log loss by season (lower is better); v6 minus each simpler arm "
             "± 95%")
    L.append(f"    {'season':<9}{'n':>6}  {'home only':>9} {'v1 core':>8} {'v4 core':>8}"
             f" {'v6':>8}   {'v6 - v4 core':>18} {'v6 - home only':>18}")
    for s, g in d.groupby("year"):
        yy = g["win"].to_numpy(float)
        p = {k: g[f"p::{k}"].to_numpy(float) for k in P}
        m4, c4 = paired(p["v6 (production)"], p["v4 core (+ delta*phase)"], yy)
        mh, ch = paired(p["v6 (production)"], p["home edge only"], yy)
        L.append(f"    {season_name(s):<9}{len(g):>6}  "
                 f"{ll(p['home edge only'], yy).mean():9.4f} "
                 f"{ll(p['v1 core (delta, b2b)'], yy).mean():8.4f} "
                 f"{ll(p['v4 core (+ delta*phase)'], yy).mean():8.4f} "
                 f"{ll(p['v6 (production)'], yy).mean():8.4f}   "
                 f"{m4:+.4f} ± {c4:.4f}   {mh:+.4f} ± {ch:.4f}")
    m4, c4 = paired(v6, P["v4 core (+ delta*phase)"], y)
    mh, ch = paired(v6, P["home edge only"], y)
    L.append(f"    {'pooled':<9}{len(d):>6}  " + " ".join(
        f"{ll(P[k], y).mean():{w}.4f}" for k, w in
        (("home edge only", 9), ("v1 core (delta, b2b)", 8),
         ("v4 core (+ delta*phase)", 8), ("v6 (production)", 8)))
        + f"   {m4:+.4f} ± {c4:.4f}   {mh:+.4f} ± {ch:.4f}")
    mb, cb = mean_ci(brier(v6, y) - brier(P["v4 core (+ delta*phase)"], y))
    L.append(f"    Brier v6 - v4 core, pooled: {mb:+.4f} ± {cb:.4f}")

    L.append("")
    L.append("  A2. Calibration of v6 against outcomes: slope (1 = calibrated shape) "
             "and intercept (0), ± 95%; predicted vs actual home win %")
    for s, g in d.groupby("year"):
        sl, slc, ic, icc = calib(g["p::v6 (production)"], g["win"])
        L.append(f"    {season_name(s):<9} n={len(g):<5} slope {sl:.2f} ± {slc:.2f}  "
                 f"intercept {ic:+.2f} ± {icc:.2f}  home won "
                 f"{100 * g['win'].mean():.1f}% vs predicted "
                 f"{100 * np.nanmean(g['p::v6 (production)']):.1f}%")
    first = d["year"] == d["year"].min()
    for name, k in (("pooled", np.ones(len(d), bool)),
                    ("pooled, without the first fold (one training season)",
                     ~first.to_numpy())):
        sl, slc, ic, icc = calib(v6[k], y[k])
        L.append(f"    {name}: n={int(k.sum())}  slope {sl:.2f} ± {slc:.2f}  intercept "
                 f"{ic:+.2f} ± {icc:.2f}  home won {100 * y[k].mean():.1f}% vs "
                 f"predicted {100 * np.nanmean(v6[k]):.1f}%")

    L.append("")
    L.append("  A3. Calibration slope by season phase, pooled seasons (the v1 core has "
             "no phase term; v4 and v6 do)")
    for lo, hi, name in PHASES:
        k = ((d["phase"] >= lo) & (d["phase"] < hi)).to_numpy()
        cells = []
        for arm in ("v1 core (delta, b2b)", "v4 core (+ delta*phase)", "v6 (production)"):
            sl, slc, _, _ = calib(P[arm][k], y[k])
            cells.append(f"{arm.split(' (')[0]} {sl:.2f} ± {slc:.2f}")
        L.append(f"    {name:<14} n={int(k.sum()):<5} " + "   ".join(cells))

    L.append("")
    L.append("  A4. Reliability of v6, the favourite's side, pooled (Wilson 95%)")
    fav = np.maximum(v6, 1 - v6)
    won = np.where(v6 >= 0.5, y, 1 - y)
    for lo, hi in zip(FAV_BINS[:-1], FAV_BINS[1:]):
        k = (fav >= lo) & ((fav < hi) if hi < 1 else (fav <= hi))
        n = int(k.sum())
        if n < 20:
            continue
        w = int(won[k].sum())
        a, b = c.wilson(w, n)
        L.append(f"    {lo:.2f}-{hi:.2f}  n={n:<5} predicted {100 * fav[k].mean():5.1f}%"
                 f"  won {100 * w / n:5.1f}% ({100 * a:.1f}-{100 * b:.1f})")

    L.append("")
    L.append("  A5. By route (production routing), pooled")
    for route, g in d.groupby("route"):
        sl, slc, _, _ = calib(g["p::v6 (production)"], g["win"])
        L.append(f"    {route:<6} n={len(g):<5} log loss "
                 f"{ll(g['p::v6 (production)'], g['win']).mean():.4f}  slope "
                 f"{sl:.2f} ± {slc:.2f}")

    L.append("")
    L.append("  A6. Back-to-back games: home win % actual vs v6 predicted, pooled")
    for v, name in ((1, "away team on a back-to-back"), (-1, "home team on a back-to-back"),
                    (0, "neither or both")):
        k = (d["b2b_net"] == v).to_numpy()
        if k.sum() < 20:
            continue
        a, b = c.wilson(int(y[k].sum()), int(k.sum()))
        L.append(f"    {name:<30} n={int(k.sum()):<5} won {100 * y[k].mean():5.1f}% "
                 f"({100 * a:.1f}-{100 * b:.1f})  predicted {100 * np.nanmean(v6[k]):5.1f}%")

    L.append("")
    L.append("  A7. Each v6 term dropped in turn (refit without it): log loss without "
             "- with, ± 95% (positive = the term helps out of sample), on the games "
             "the term reaches (v6 terms: base and availability routes; availability: "
             "its own route, which exists from the first season with a report season "
             "to train on)")
    route = d["route"].to_numpy()
    for key, name in TERMS + (("availability", "injury-report availability"),):
        k = (route == "avail") if key == "availability" else (route != "v4")
        p = P[f"v6 - {key}"]
        m, ci = paired(p[k], v6[k], y[k])
        g = d[k]
        per = [paired(h[f"p::v6 - {key}"].to_numpy(float),
                      h["p::v6 (production)"].to_numpy(float),
                      h["win"].to_numpy(float))[0] for _, h in g.groupby("year")]
        L.append(f"    {name:<28} {m:+.4f} ± {ci:.4f}  n={int(k.sum()):<5} (helps in "
                 f"{sum(x > 0 for x in per)} of {len(per)} seasons)")
    return L, d


def fit_offset(p, y, iters=50):
    """(b, se): the season-level logit offset of y on p with slope 1."""
    lp, y = c.logit(p), np.asarray(y, float)
    b = 0.0
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(lp + b)))
        g, h = float((y - mu).sum()), float((mu * (1 - mu)).sum())
        b += g / h
        if abs(g / h) < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(lp + b)))
    return b, 1 / np.sqrt(float((mu * (1 - mu)).sum()))


def eb_tau(offsets):
    """Empirical-Bayes sd of the true season offsets from [(b, se)]; NaN with
    fewer than two seasons, 0 when the spread is all noise."""
    if len(offsets) < 2:
        return float("nan")
    b = np.array([o[0] for o in offsets])
    se = np.array([o[1] for o in offsets])
    return float(np.sqrt(max(b.var(ddof=1) - (se ** 2).mean(), 0.0)))


def home_offset(dates, y, p, tau):
    """p shifted by the season's residual on strictly earlier dates:
    sigma(logit p + b), b = sum(y - p) / (sum p(1-p) + 1/tau^2). tau NaN or
    0 leaves p unchanged."""
    p = np.asarray(p, float)
    if not (np.isfinite(tau) and tau > 0):
        return p.copy()
    t = pd.DataFrame({"d": pd.to_datetime(dates), "r": np.asarray(y, float) - p,
                      "v": p * (1 - p)})
    day = t.groupby("d")[["r", "v"]].sum().sort_index()
    before = day.cumsum().shift(1, fill_value=0.0)
    r, v = before.loc[t["d"], "r"].to_numpy(), before.loc[t["d"], "v"].to_numpy()
    b = r / (v + 1 / tau ** 2)
    return 1 / (1 + np.exp(-(c.logit(p) + b)))


def market_lines(d, cols):
    """Each priced book x season: n, col - close log loss, and the paired
    difference of every col against the first."""
    m = c.with_market(d[["year", "slate_date", "home", "away", "win"] + cols])
    m = m[m["close_q_home"].notna()]
    L = []
    for (yr, book), g in m.groupby(["year", "close_book"]):
        yy = g["win"].to_numpy(float)
        q = g["close_q_home"].to_numpy(float)
        cells = [f"{k} - close {np.mean(ll(g[k], yy) - ll(q, yy)):+.4f}" for k in cols]
        md, cd = paired(g[cols[1]].to_numpy(float), g[cols[0]].to_numpy(float), yy)
        L.append(f"    {season_name(yr)} {book:<8} n={len(g):<5} " + "  ".join(cells)
                 + f"   {cols[1]} - {cols[0]} {md:+.4f} ± {cd:.4f}")
    return L


def part_a8(d):
    """In-season home-edge update on the walk-forward v6 predictions."""
    L = ["  A8. In-season home edge: v6 shifted by this season's own residual on "
         "earlier dates (tau walk-forward, empirical Bayes from the earlier test "
         "seasons); log loss adjusted - v6, ± 95% (negative = the update helps)"]
    base = "p::v6 (production)"
    d = d.sort_values("date").reset_index(drop=True)
    d["p_adj"] = d[base]
    offsets, rows = [], []
    for yr, g in d.groupby("year", sort=True):
        tau = eb_tau(offsets)
        d.loc[g.index, "p_adj"] = home_offset(g["date"], g["win"], g[base], tau)
        for t in TAU_GRID:
            d.loc[g.index, f"p_tau{t}"] = home_offset(g["date"], g["win"], g[base], t)
        b, se = fit_offset(g[base].to_numpy(float), g["win"].to_numpy(float))
        offsets.append((b, se))
        yy = g["win"].to_numpy(float)
        m, ci = paired(d.loc[g.index, "p_adj"].to_numpy(float),
                       g[base].to_numpy(float), yy)
        rows.append(m)
        L.append(f"    {season_name(yr):<9} n={len(g):<5} tau {tau:5.3f}  season offset "
                 f"{b:+.3f} ± {Z95 * se:.3f}  adjusted - v6 {m:+.4f} ± {ci:.4f}  "
                 f"home won {100 * yy.mean():.1f}% vs v6 {100 * g[base].mean():.1f}% "
                 f"/ adjusted {100 * d.loc[g.index, 'p_adj'].mean():.1f}%")
    y = d["win"].to_numpy(float)
    m, ci = paired(d["p_adj"].to_numpy(float), d[base].to_numpy(float), y)
    sl, slc, ic, icc = calib(d["p_adj"], y)
    L.append(f"    pooled    n={len(d):<5} adjusted - v6 {m:+.4f} ± {ci:.4f}  (helps in "
             f"{sum(x < 0 for x in rows)} of {len(rows)} seasons)  adjusted slope "
             f"{sl:.2f} ± {slc:.2f}, intercept {ic:+.2f} ± {icc:.2f}")
    for t in TAU_GRID:
        m, ci = paired(d[f"p_tau{t}"].to_numpy(float), d[base].to_numpy(float), y)
        L.append(f"    fixed tau {t:.2f} (sensitivity, chosen in sample): adjusted - "
                 f"v6 {m:+.4f} ± {ci:.4f}")
    tau_all = eb_tau(offsets)
    L.append(f"    tau from all {len(offsets)} seasons (used out of era, B): {tau_all:.3f}")
    L.append("    against the close, priced seasons (same games):")
    try:
        L += market_lines(d.rename(columns={base: "v6", "p_adj": "adjusted"}),
                          ["v6", "adjusted"])
    except Exception as e:  # noqa: BLE001
        L.append(f"    (no market rows: {e!r})")
    return L, tau_all


def frozen_core():
    """The committed composite weights and v4 logit."""
    with open(os.path.join(ROOT, "model", "weights.json")) as fh:
        w = json.load(fh)
    with open(os.path.join(ROOT, "model", "logit_v4.json")) as fh:
        lg = json.load(fh)
    return w, lg


def part_b(era, tau=0.10):
    w, lg = frozen_core()
    feats, coef = lg["features"], np.asarray(lg["coef"], float)
    L = ["== B. Out of era, the core (delta, b2b, delta*phase), games 10+",
         f"  frozen = the committed weights ({w['years'][0]}-{w['years'][-1]}, no "
         f"2020) and v4 logit ({lg['years'][0]}-{lg['years'][-1]}), never refitted; "
         "logit refit = committed weights, logit fitted on the era's earlier "
         "seasons; full refit = composite weights and logit both fitted on the "
         "era's earlier seasons; home = the earlier era seasons' home win rate.",
         "  Log loss per game; differences ± 95% (positive = frozen worse)."]
    raw = {}
    for t in era:
        try:
            raw[t] = c.season_table(t, None)
        except Exception as e:  # noqa: BLE001
            L.append(f"  {season_name(t)}: not built ({e!r})")
    tabs = {t: c.with_weights(r, w) for t, r in raw.items()}
    pooled = []
    L.append(f"    {'season':<9}{'n':>6}  {'home won':>8} {'frozen pred':>11}  "
             f"{'frozen':>7} {'logit':>7} {'full':>7} {'home':>7}  {'frozen slope':>14} "
             f"{'frozen - full refit':>20}")
    for t in sorted(tabs):
        g = tabs[t][c.finite(tabs[t], feats)].reset_index(drop=True)
        yy = g["win"].to_numpy(float)
        pf = 1 / (1 + np.exp(-(lg["intercept"] + g[feats].to_numpy(float) @ coef)))
        prior = [s for s in sorted(tabs) if s < t]
        pr = pfull = ph = np.full(len(g), np.nan)
        if prior:
            tr = pd.concat([tabs[s] for s in prior], ignore_index=True)
            pr = fitpred(tr, feats, g)
            ph = np.full(len(g), tr["win"].mean())
            we = nc.fit_weights(prior)
            gt = c.with_weights(raw[t], we)
            gt = gt[c.finite(tabs[t], feats)].reset_index(drop=True)
            tre = pd.concat([c.with_weights(raw[s], we) for s in prior], ignore_index=True)
            pfull = fitpred(tre, feats, gt)
        sl, slc, _, _ = calib(pf, yy)
        md, cd = paired(pf, pfull, yy) if prior else (np.nan, np.nan)
        L.append(f"    {season_name(t):<9}{len(g):>6}  {100 * yy.mean():7.1f}% "
                 f"{100 * pf.mean():10.1f}%  {ll(pf, yy).mean():7.4f} "
                 f"{np.nanmean(ll(pr, yy)) if prior else np.nan:7.4f} "
                 f"{np.nanmean(ll(pfull, yy)) if prior else np.nan:7.4f} "
                 f"{np.nanmean(ll(ph, yy)) if prior else np.nan:7.4f}  "
                 f"{sl:6.2f} ± {slc:.2f}   {md:+.4f} ± {cd:.4f}")
        pooled.append(pd.DataFrame(dict(y=yy, pf=pf, pr=pr, pfull=pfull, ph=ph,
                                        phase=g["phase"], b2b=g["b2b_net"],
                                        padj=home_offset(g["date"], yy, pf, tau))))
    if not pooled:
        return L
    d = pd.concat(pooled, ignore_index=True)
    y, pf = d["y"].to_numpy(float), d["pf"].to_numpy(float)
    sl, slc, ic, icc = calib(pf, y)
    L.append(f"    pooled: n={len(d)}  frozen slope {sl:.2f} ± {slc:.2f}  intercept "
             f"{ic:+.2f} ± {icc:.2f}  home won {100 * y.mean():.1f}% vs predicted "
             f"{100 * pf.mean():.1f}%")
    k = np.isfinite(d["pfull"].to_numpy(float))
    for name, col in (("logit refit", "pr"), ("full refit", "pfull"), ("home only", "ph")):
        m, ci = paired(pf[k], d[col].to_numpy(float)[k], y[k])
        L.append(f"      frozen - {name:<12} {m:+.4f} ± {ci:.4f}  (n={int(k.sum())}, "
                 "seasons after the first)")
    pa = d["padj"].to_numpy(float)
    m, ci = paired(pa, pf, y)
    sa, sac, ia, iac = calib(pa, y)
    helps = sum(paired(g["padj"].to_numpy(float), g["pf"].to_numpy(float),
                       g["y"].to_numpy(float))[0] < 0
                for _, g in d.groupby(np.repeat(np.arange(len(pooled)),
                                                [len(x) for x in pooled])))
    L.append(f"      frozen + in-season home edge (tau {tau:.3f}, from the modern "
             f"seasons) - frozen {m:+.4f} ± {ci:.4f} (helps in {helps} of "
             f"{len(pooled)} seasons); intercept after {ia:+.2f} ± {iac:.2f}, home "
             f"predicted {100 * pa.mean():.1f}%")
    for lo, hi, name in PHASES:
        kk = ((d["phase"] >= lo) & (d["phase"] < hi)).to_numpy()
        s2, c2, _, _ = calib(pf[kk], y[kk])
        L.append(f"      {name:<14} n={int(kk.sum()):<5} frozen slope {s2:.2f} ± {c2:.2f}")
    for v, name in ((1, "away team on a back-to-back"), (-1, "home team on a back-to-back")):
        kk = (d["b2b"] == v).to_numpy()
        if kk.sum() < 20:
            continue
        a, b = c.wilson(int(y[kk].sum()), int(kk.sum()))
        L.append(f"      {name:<30} n={int(kk.sum()):<5} home won {100 * y[kk].mean():5.1f}% "
                 f"({100 * a:.1f}-{100 * b:.1f})  frozen predicted {100 * pf[kk].mean():5.1f}%")
    L.append("")
    try:
        we = nc.fit_weights(sorted(raw))
    except Exception as e:  # noqa: BLE001
        L.append(f"  era weights: not fitted ({e!r})")
        return L
    L.append(f"  B2. Composite weights, wins per +1 SD over 82 games: committed vs "
             f"fitted on {season_name(min(raw))} to {season_name(max(raw))} "
             f"(in-sample R2 {w.get('r2_in_sample', np.nan):.3f} vs "
             f"{we['r2_in_sample']:.3f})")
    for n, a, b in zip(nc.FEATURES, w["w"], we["w"]):
        L.append(f"    {n:<16} {82 * a:+6.2f}  {82 * b:+6.2f}")
    L.append(f"    correlation of the two weight vectors: "
             f"{np.corrcoef(w['w'], we['w'])[0, 1]:+.2f}")
    return L


def early_lines(label, p, y, ph=None):
    sl, slc, ic, icc = calib(p, y)
    lean, lci = mean_ci(np.asarray(p, float) - np.asarray(y, float))
    s = (f"    {label:<9} n={len(y):<5} home won {100 * np.mean(y):5.1f}%  predicted "
         f"{100 * np.mean(p):5.1f}%  lean {100 * lean:+.1f} ± {100 * lci:.1f} pts  "
         f"slope {sl:.2f} ± {slc:.2f}  log loss {ll(p, y).mean():.4f}")
    if ph is not None:
        m, ci = paired(np.asarray(p, float), np.asarray(ph, float), np.asarray(y, float))
        s += f"  vs home only {m:+.4f} ± {ci:.4f}"
    return s


def part_c(folds, era):
    """Games 1-9: the carryover model against outcomes."""
    L = ["== C. Games 1-9, the carryover model (cold_start.py), against outcomes",
         "  Walk-forward: each fold's composite weights; the early logit fitted on "
         "the earlier seasons of " + f"{EARLY_TRAIN[0]}-{EARLY_TRAIN[-1]} (no 2020); "
         "home only = those seasons' early home rate. lean = predicted - actual "
         "home win %, ± 95%."]
    rows = []
    for yr, f in sorted(folds.items()):
        train = [s for s in EARLY_TRAIN if s < yr]
        if not train:
            continue
        try:
            em = cold_start.fit_early(train, f.weights)
            tg = pd.concat([cold_start.early_games(s, f.weights) for s in train])
            e = cold_start.early_games(yr, f.weights, min_gp=1, window=nc.MIN_GAMES)
        except Exception as ex:  # noqa: BLE001
            L.append(f"    {season_name(yr)}: not built ({ex!r})")
            continue
        if not len(e):
            continue
        p = nc.predict(em, e[cold_start.FEATURES].to_numpy(float))
        ph = np.full(len(e), tg["win"].mean())
        y = e["win"].to_numpy(float)
        L.append(early_lines(season_name(yr), p, y, ph) +
                 f"  (intercept {em['intercept']:+.3f})")
        rows.append(pd.DataFrame(dict(p=p, ph=ph, y=y)))
    if rows:
        d = pd.concat(rows, ignore_index=True)
        L.append(early_lines("pooled", d["p"], d["y"], d["ph"]))
    w = json.load(open(os.path.join(ROOT, "model", "weights.json")))
    # v7's logit_early.json adds talent_diff, which needs box scores the
    # out-of-era seasons lack: score them with the frozen carryover logit.
    em = json.load(open(os.path.join(ROOT, "model", cold_start.FROZEN_FILE)))
    L.append(f"  Out of era: the frozen carryover logit ({em['years'][0]}-{em['years'][-1]}, "
             f"intercept {em['intercept']:+.3f}) with the committed weights")
    rows = []
    for yr in era:
        try:
            e = cold_start.early_games(yr, w, min_gp=1, window=nc.MIN_GAMES)
        except Exception as ex:  # noqa: BLE001
            L.append(f"    {season_name(yr)}: not built ({ex!r})")
            continue
        if not len(e):
            continue
        p = nc.predict(em, e[em["features"]].to_numpy(float))
        y = e["win"].to_numpy(float)
        L.append(early_lines(season_name(yr), p, y))
        rows.append(pd.DataFrame(dict(p=p, y=y)))
    if rows:
        d = pd.concat(rows, ignore_index=True)
        L.append(early_lines("pooled", d["p"], d["y"]))
    return L


def draft_picks(year):
    """norm_name -> pick in the year's NBA draft (BBR draft page, cached)."""
    h = nc.fetch(f"{nc.BASE}/draft/NBA_{year}.html", nc.CACHE / f"draft_{year}.html")
    m = re.search(r'<table[^>]*id="stats".*?</table>', nc.uncomment(h), re.S)
    df = pd.read_html(io.StringIO(m.group(0)))[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [x[-1] for x in df.columns]
    df = df[pd.to_numeric(df["Pk"], errors="coerce").notna()]
    return {pav.norm_name(n): int(k) for n, k in zip(df["Player"], df["Pk"])
            if isinstance(n, str)}


def rookie_curve(seasons):
    """(a, b, n): rookie-season BPM ~ a + b log(pick), weighted by minutes,
    over the rookies of `seasons` (drafted the summer before each)."""
    X, Y, W = [], [], []
    for t in seasons:
        bpm = pav.load_bpm(t)
        for name, pick in draft_picks(t - 1).items():
            rec = bpm.get(name)
            if rec is None or not (np.isfinite(rec[0]) and rec[1] > 0):
                continue
            X.append(np.log(pick))
            Y.append(rec[0])
            W.append(rec[1])
    X, Y, sw = np.asarray(X), np.asarray(Y), np.sqrt(np.asarray(W, float))
    A = np.column_stack([np.ones_like(X), X]) * sw[:, None]
    a, b = np.linalg.lstsq(A, Y * sw, rcond=None)[0]
    return float(a), float(b), len(X)


def rookie_values(value, box, bpm, picks, curve):
    """value with each drafted rookie (no BPM row last season) at the
    rookie BPM his pick predicts, above replacement; (value, n rookies)."""
    out, n = dict(value), 0
    a, b = curve[:2]
    for pid, name in box.drop_duplicates("player_id").set_index("player_id")["name"].items():
        key = pav.norm_name(name)
        if bpm.get(key) is None and key in picks:
            out[pid] = a + b * np.log(picks[key]) - pav.REPLACEMENT_BPM
            n += 1
    return out, n


def rookie_column(folds):
    """{(slate_date, home, away): talent_diff with drafted rookies by pick},
    each season's curve fitted on earlier seasons only; and the curve lines."""
    seasons = sorted({int(t) for f in folds.values()
                      for fr in (f.base_train, f.test) for t in fr["year"].unique()})
    col, L = {}, []
    for t in seasons:
        curve = rookie_curve(range(2015, t))
        box = pav.fetch_box(t)
        bpm = pav.load_bpm(t - 1)
        value, _, _ = pav.player_values(box, bpm)
        value, n = rookie_values(value, box, bpm, draft_picks(t - 1), curve)
        tal = pav.talent_fn(box, value, pav.arrival_roles(box, bpm))
        keys = set()
        for f in folds.values():
            for fr in (f.base_train, f.test):
                g = fr[fr["year"] == t]
                keys |= set(zip(g["slate_date"], g["home"], g["away"]))
        for sd, h, a in keys:
            col[(sd, h, a)] = tal(h, sd) - tal(a, sd)
        L.append(f"    {season_name(t):<9} curve BPM = {curve[0]:+.2f} {curve[1]:+.2f} "
                 f"log(pick) (fit on {curve[2]} rookies of 2014-15 to "
                 f"{season_name(t - 1)}); {n} rookies valued")
    return col, L


def part_a9(folds, d):
    """v6 with drafted rookies valued by pick, refit, against v6."""
    L = ["  A9. Rookies by draft slot: v6 refitted with talent valuing drafted "
         "rookies by pick (instead of at replacement); log loss rookie - v6, ± 95%"]
    try:
        col, lines = rookie_column(folds)
    except Exception as e:  # noqa: BLE001
        return L + [f"    not built ({e!r})"]
    L += lines
    v6r = [x if x != "talent_diff" else "talent_rk" for x in V6]
    av6r = [x if x != "talent_diff" else "talent_rk" for x in AV6]
    ps = []
    for yr, f in sorted(folds.items()):
        for fr in (f.base_train, f.avail_train, f.v4_train, f.test):
            if fr is not None:
                fr["talent_rk"] = [col.get(k, np.nan) for k in
                                   zip(fr["slate_date"], fr["home"], fr["away"])]
        ps.append(routed(f, v6r, av6r))
    d = d.assign(rookie=np.concatenate(ps))
    k = (d["route"] != "v4").to_numpy()
    y = d["win"].to_numpy(float)
    v6 = d["p::v6 (production)"].to_numpy(float)
    m, ci = paired(d["rookie"].to_numpy(float)[k], v6[k], y[k])
    per = [paired(g["rookie"].to_numpy(float), g["p::v6 (production)"].to_numpy(float),
                  g["win"].to_numpy(float))[0] for _, g in d[k].groupby("year")]
    L.append(f"    pooled    n={int(k.sum()):<5} rookie - v6 {m:+.4f} ± {ci:.4f}  (better "
             f"in {sum(x < 0 for x in per)} of {len(per)} seasons)")
    L.append("    against the close, priced seasons (same games):")
    try:
        L += market_lines(d.rename(columns={"p::v6 (production)": "v6"}), ["v6", "rookie"])
    except Exception as e:  # noqa: BLE001
        L.append(f"    (no market rows: {e!r})")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", nargs="+", type=int, default=SEASONS)
    ap.add_argument("--era", default="2005-2014", help="BBR years for part B ('' skips)")
    ap.add_argument("--skip-a", action="store_true")
    a = ap.parse_args(argv)
    L = ["History backtest -- the current model against outcomes only (no market "
         "prices), games 10+. ± is 95%.", ""]
    tau = TAU_GRID[1]
    era = nc.parse_years(a.era.split()) if a.era else []
    if not a.skip_a:
        folds = c.build_folds(a.seasons, with_e=False)
        la, d = part_a(folds)
        l8, tau = part_a8(d)
        L += la + [""] + l8 + [""] + part_a9(folds, d) + [""]
    if era:
        L += part_b(era, tau if np.isfinite(tau) and tau > 0 else TAU_GRID[1]) + [""]
    if not a.skip_a:
        L += part_c(folds, [t for t in era if t - 1 in era]) + [""]
    text = "\n".join(L)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
