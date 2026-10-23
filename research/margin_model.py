#!/usr/bin/env python3
"""Margin model: v8's inputs fitted on the final margin instead of the win.

  python research/margin_model.py [--seasons 2017 ... 2026]

Owner's question (2026-10-23): can a model fitted directly on margins give
the spread, and would it also be accurate for win probability? v8 fits
P(home win) by logistic regression and the pages map it to a margin with a
fixed 13.5 x Phi^-1(P) (analysis.ATS_SIGMA, MODEL.md §5: an unvalidated
assumption). The margin model M uses the same inputs, routing and training
seasons as v8 (research/unified.py, schedule fade10, the shipped K = 10):

  * the base terms on every game, the availability terms on covered games
    from g = 1, the v4 core when a term is missing (v8's routes);
  * ordinary least squares on the final home margin (overtime included)
    in place of the logistic fit on the win, per route;
  * its win probability Phi(m / sigma), sigma = the route's residual SD on
    the training seasons (no test-season outcome enters it);
  * a secondary reading, M-probit: P = Phi(a + b m) with a, b fitted on the
    training seasons' win/loss against their in-sample m.

Questions and readings, fixed before the run:

  Q1  Win probability: M's P against v8's P on the same games, log loss and
      Brier (M - v8; negative = M better), games 1+ pooled and by season,
      game buckets, calibration slope; against the close on the priced
      seasons, each book x season alone. M passes as "as accurate" only if
      it is not worse than v8 on either large book-season (2024-25 ESPN
      BET, 2025-26 DK) with Brier agreeing (G7); a gain over 0.008 a season
      is the leakage tripwire. Passing G7 makes it a candidate, not a model
      change: that is the owner's decision, with a new MODEL_TAG.
  Q2  Spread: M's margin against v8's 13.5 x Phi^-1(P) and against the
      closing and opening spreads (minus the home line), MAE and RMSE on
      the same games (M - other; negative = M better).
  Q3  Against the spread (descriptive, no bet rule): the slope of the
      home cover margin (margin + home line) on each model's edge (model
      margin + home line) at the open and at the close, M beside v8.

Walk-forward: every fit (composite weights, every regression, sigma) uses
seasons strictly before the test season. Research only: nothing here touches
the ledgers, the pages or model/*.json. Writes diagnostics/margin_model/
RESULTS.txt and predictions.csv. ± is 95%, game-level (paired differences).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "v5_audit"))
import analysis                                   # noqa: E402
import backfill_history as bf                     # noqa: E402
import ledger                                     # noqa: E402
import nba_composite as nc                        # noqa: E402
import player_availability as pav                 # noqa: E402
import unified as u                               # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "margin_model")
SCHEDULE = "fade10"
SIGMA_FIXED = 13.5                 # analysis.ATS_SIGMA, the pages' conversion
KEY = ["slate_date", "home", "away"]
TRIPWIRE = 0.008
log = u.log


def norm_cdf(x):
    return u.c.norm_cdf(np.asarray(x, float))


def norm_ppf(p):
    """Phi^-1, NaN where P is missing (analysis.ats_sides' NormalDist)."""
    from statistics import NormalDist
    nd = NormalDist()
    p = np.asarray(p, float)
    return np.array([nd.inv_cdf(min(max(v, 1e-9), 1 - 1e-9)) if np.isfinite(v) else np.nan
                     for v in p.ravel()]).reshape(p.shape)


# --------------------------------------------------------------- margins --
def margins(season):
    """Final home margin per game (overtime included) from a Season's logs."""
    rows = []
    for tm, cl in season.logs.items():
        h = cl[cl["home"].astype(bool)]
        rows += [{"slate_date": pd.Timestamp(r.date).strftime("%Y-%m-%d"), "home": tm,
                  "away": r.opp, "margin": float(r.pts - r.opp_pts)}
                 for r in h.itertuples()]
    return pd.DataFrame(rows, columns=KEY + ["margin"]).drop_duplicates(KEY)


# ------------------------------------------------------------------- fits --
def fit_ols(train, feats, min_n=50):
    """OLS of margin on feats (with an intercept) and its residual SD; None
    with fewer than min_n complete rows."""
    tr = train[u.finite(train, feats) & np.isfinite(train["margin"].to_numpy(float))]
    if len(tr) < min_n:
        return None
    X = np.column_stack([np.ones(len(tr)), tr[feats].to_numpy(float)])
    y = tr["margin"].to_numpy(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    sigma = float(np.sqrt((res ** 2).sum() / max(1, len(tr) - X.shape[1])))
    m = {"intercept": float(b[0]), "coef": [float(v) for v in b[1:]],
         "features": list(feats), "n": len(tr), "sigma": sigma}
    # M-probit: P(win) = Phi(a + b * m), fitted on the same training rows
    fit = u.c.fit((X @ b)[:, None], tr["win"].to_numpy(float), l2=0.0, link="probit")
    m["probit"] = (float(fit["intercept"]), float(fit["coef"][0]))
    return m


def pred_margin(m, df):
    out = np.full(len(df), np.nan)
    if m is None or not len(df):
        return out
    ok = u.finite(df, m["features"])
    if ok.any():
        out[ok] = m["intercept"] + df.loc[ok, m["features"]].to_numpy(float) @ np.asarray(m["coef"])
    return out


def margin_route(tw_s, Y):
    """M on season Y, v8's routing and training seasons (unified.unified):
    (margin, sigma, probit P, route, models)."""
    ph = [t for t in bf.PHASE_YEARS if t < Y and t in tw_s]
    allb = pd.concat([tw_s[t] for t in ph], ignore_index=True)
    m_base, m_v4 = fit_ols(allb, u.BASE_F), fit_ols(allb, u.V4_F)
    rep = [t for t in bf.REPORT_YEARS if t < Y and t in tw_s]
    m_av = None
    if rep:
        alla = pd.concat([tw_s[t] for t in rep], ignore_index=True)
        m_av = fit_ols(alla[(alla["g"] >= 1) & alla["av_min"].notna()], u.AVAIL_F)
    te = tw_s[Y]
    ma = pred_margin(m_av, te)
    ma[(te["g"] < 1).to_numpy()] = np.nan
    mb, m4 = pred_margin(m_base, te), pred_margin(m_v4, te)
    n = len(te)
    mg, sg, pp = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    route = np.full(n, "", dtype=object)
    for name, mm, v in (("v4", m_v4, m4), ("base", m_base, mb), ("avail", m_av, ma)):
        k = np.isfinite(v)
        if mm is None or not k.any():
            continue
        mg[k], sg[k], route[k] = v[k], mm["sigma"], name
        a, b = mm["probit"]
        pp[k] = norm_cdf(a + b * v[k])
    return mg, sg, pp, route, {"base": m_base, "avail": m_av, "v4": m_v4}


def run(seasons, cache_dir=pav.DEFAULT_CACHE):
    need = sorted({t for Y in seasons for t in bf.PHASE_YEARS if t <= Y} | set(seasons))
    data = {t: u.Season(t, cache_dir) for t in need}
    terms = {}
    archive = pav.ReportArchive(os.path.join(cache_dir, "injury_reports"))
    try:
        for t in [t for t in bf.REPORT_YEARS if t in data]:
            try:
                terms[t], s = pav.season_terms(t, archive, cache_dir)
                terms[t] = terms[t].drop_duplicates(KEY)
                log(f"report season {t}: {len(s['covered'])}/{s['games']} covered")
            except Exception as e:  # noqa: BLE001
                log(f"report season {t}: no terms ({e!r})")
    finally:
        archive.save()
    k = u.SCHEDULES[SCHEDULE]
    tabs = {t: u.add_terms(s.table(k), terms.get(t)).merge(margins(s), on=KEY, how="left")
            for t, s in data.items()}
    out, models = [], {}
    for Y in seasons:
        wts = nc.fit_weights(bf.training_years(bf.WEIGHT_YEARS, Y, True))
        tw = {t: u.with_weights(tabs[t], wts) for t in tabs if t <= Y}
        p8, r8, _ = u.unified(tw, Y)
        mg, sg, pp, rm, mm = margin_route(tw, Y)
        te = tw[Y][KEY + ["year", "g", "win", "margin"]].copy()
        te["p_v8"], te["route_v8"] = p8, r8
        te["m_M"], te["sigma_M"], te["route_M"] = mg, sg, rm
        te["p_M"] = norm_cdf(mg / sg)
        te["p_Mprobit"] = pp
        te["m_v8"] = SIGMA_FIXED * norm_ppf(p8)
        models[Y] = mm
        log(f"fold {Y}: {len(te)} games, routes {pd.Series(rm).value_counts().to_dict()}, "
            f"sigma " + ", ".join(f"{r} {m['sigma']:.2f}" for r, m in mm.items() if m))
        out.append(te)
    return pd.concat(out, ignore_index=True), models


# ----------------------------------------------------------------- report --
def prob_lines(d, label):
    """Q1 rows: M and M-probit against v8 on the same games."""
    L = []
    y = d["win"].to_numpy(float)
    for arm in ("p_M", "p_Mprobit"):
        ok = np.isfinite(d[arm]) & np.isfinite(d["p_v8"])
        z, yy = d[ok], y[ok]
        dl, dlc = u.mean_ci(u.ll(z[arm], yy) - u.ll(z["p_v8"], yy))
        db, dbc = u.mean_ci((z[arm] - yy) ** 2 - (z["p_v8"] - yy) ** 2)
        L.append(f"  {label:<14} {arm[2:]:<8} n {ok.sum():>6}  log loss M - v8 {dl:+.4f} ± "
                 f"{dlc:.4f}  Brier {db:+.4f} ± {dbc:.4f}  slope {u.slope(z[arm], yy):.3f} "
                 f"(v8 {u.slope(z['p_v8'], yy):.3f})")
    return L


def q1_outcomes(d):
    L = ["== Q1. Win probability against outcomes (walk-forward; M - v8, negative = M "
         "better, ± 95%)"]
    g1 = d[d["g"] >= 1]
    L += prob_lines(g1, "games 1+")
    for lo, hi, name in u.BUCKETS:
        L += prob_lines(d[d["g"].between(lo, hi)], name)
    L.append("  by season (games 1+, Phi(m / sigma)):")
    wins = 0
    for Y, z in g1.groupby("year"):
        yy = z["win"].to_numpy(float)
        dl, dlc = u.mean_ci(u.ll(z["p_M"], yy) - u.ll(z["p_v8"], yy))
        wins += dl < 0
        L.append(f"    {u.season_name(Y)}  n {len(z):>5}  {dl:+.4f} ± {dlc:.4f}")
    L.append(f"    M better in {wins} of {g1['year'].nunique()} seasons")
    return L


def priced(d):
    """Test rows joined to graded reconstructed rows with a close (and the
    same book's open spread, diagnostics/ats_signal/open_spreads.csv)."""
    g = ledger.graded(ledger.load(os.path.join(ROOT, ledger.RECON_PATH))).copy()
    g["q_close"] = analysis.price_q_home(g, "close")
    g = g[g["q_close"].between(0, 1, inclusive="neither")]
    cols = KEY + ["game_id", "season", "close_book", "q_close", "close_spread"]
    m = d.merge(g[cols], on=KEY, how="inner")
    try:
        o = pd.read_csv(os.path.join(ROOT, "diagnostics", "ats_signal", "open_spreads.csv"))
        o = o.rename(columns={"book": "close_book"})[["game_id", "close_book", "open_spread"]]
        m["game_id"] = m["game_id"].astype(str)
        o["game_id"] = o["game_id"].astype(str)
        m = m.merge(o, on=["game_id", "close_book"], how="left")
    except FileNotFoundError:
        m["open_spread"] = np.nan
    return m


def q1_close(m):
    L = ["", "== Q1 (G7). Against the close, each book x season alone (± 95%)"]
    for (season, book), x in m.groupby(["season", "close_book"]):
        L.append(f"  {u.BOOK_NAMES.get(book, book)} {u.season_name(season)}")
        for lo, hi, name in ((1, 9, "games 1-9"), (10, 999, "games 10+"), (1, 999, "games 1+")):
            z = x[x["g"].between(lo, hi) & np.isfinite(x["p_M"]) & np.isfinite(x["p_v8"])]
            if len(z) < 5:
                continue
            yy, q = z["win"].to_numpy(float), z["q_close"].to_numpy(float)
            mc, mcc = u.mean_ci(u.ll(z["p_M"], yy) - u.ll(q, yy))
            vc, vcc = u.mean_ci(u.ll(z["p_v8"], yy) - u.ll(q, yy))
            dl, dlc = u.mean_ci(u.ll(z["p_M"], yy) - u.ll(z["p_v8"], yy))
            db, dbc = u.mean_ci((z["p_M"] - yy) ** 2 - (z["p_v8"] - yy) ** 2)
            flag = "  TRIPWIRE" if dl < -TRIPWIRE else ""
            L.append(f"    {name:<10} n {len(z):>5}  v8 - close {vc:+.4f} ± {vcc:.4f}  "
                     f"M - close {mc:+.4f} ± {mcc:.4f}  M - v8 {dl:+.4f} ± {dlc:.4f} "
                     f"(Brier {db:+.4f} ± {dbc:.4f}){flag}")
    return L


def err_row(name, pred, actual, ref=None):
    e = np.abs(pred - actual)
    s = f"    {name:<26} MAE {np.mean(e):6.3f}  RMSE {np.sqrt(np.mean((pred - actual) ** 2)):6.3f}" \
        f"  bias {np.mean(pred - actual):+.3f}"
    if ref is not None:
        dm, dmc = u.mean_ci(e - np.abs(ref - actual))
        ds, dsc = u.mean_ci((pred - actual) ** 2 - (ref - actual) ** 2)
        s += f"   M - this: MAE {-dm:+.3f} ± {dmc:.3f}, MSE {-ds:+.2f} ± {dsc:.2f}"
    return s


def q2_spread(d, m):
    L = ["", "== Q2. The spread: predicted home margin vs the final margin "
             "(M - other, negative = M better, ± 95%)"]
    z = d[(d["g"] >= 1) & np.isfinite(d["m_M"]) & np.isfinite(d["m_v8"])]
    a = z["margin"].to_numpy(float)
    L.append(f"  all test games, games 1+ (n {len(z)})")
    L.append(err_row("M (OLS on margin)", z["m_M"].to_numpy(float), a))
    L.append(err_row("v8, 13.5 x Phi^-1(P)", z["m_v8"].to_numpy(float), a, z["m_M"].to_numpy(float)))
    for (season, book), x in m.groupby(["season", "close_book"]):
        x = x[(x["g"] >= 1) & np.isfinite(x["m_M"]) & np.isfinite(x["m_v8"])
              & np.isfinite(x["close_spread"])]
        if len(x) < 30:
            continue
        a, mm = x["margin"].to_numpy(float), x["m_M"].to_numpy(float)
        L.append(f"  {u.BOOK_NAMES.get(book, book)} {u.season_name(season)} (n {len(x)})")
        L.append(err_row("M (OLS on margin)", mm, a))
        L.append(err_row("v8, 13.5 x Phi^-1(P)", x["m_v8"].to_numpy(float), a, mm))
        L.append(err_row("close (minus home line)", -x["close_spread"].to_numpy(float), a, mm))
        o = np.isfinite(x["open_spread"].to_numpy(float))
        if o.sum() >= 30:
            L.append(err_row(f"open, n {o.sum()}", -x["open_spread"].to_numpy(float)[o], a[o], mm[o]))
    return L


def ats_slope(edge, cover):
    k = np.isfinite(edge) & np.isfinite(cover)
    if k.sum() < 50:
        return float("nan"), float("nan"), 0
    X = np.column_stack([np.ones(k.sum()), edge[k]])
    b, *_ = np.linalg.lstsq(X, cover[k], rcond=None)
    r = cover[k] - X @ b
    cov = (r @ r / (k.sum() - 2)) * np.linalg.inv(X.T @ X)
    return float(b[1]), float(u.Z95 * np.sqrt(cov[1, 1])), int(k.sum())


def q3_ats(m):
    L = ["", "== Q3. Against the spread (descriptive): slope of the home cover margin on "
             "the model's edge, points per point, ± 95%"]
    for (season, book), x in m.groupby(["season", "close_book"]):
        x = x[x["g"] >= 10]
        L.append(f"  {u.BOOK_NAMES.get(book, book)} {u.season_name(season)}, games 10+")
        for at in ("open", "close"):
            line = x[f"{at}_spread"].to_numpy(float)
            cover = x["margin"].to_numpy(float) + line
            for arm in ("m_M", "m_v8"):
                b, bc, n = ats_slope(x[arm].to_numpy(float) + line, cover)
                if n:
                    L.append(f"    at the {at:<5} {arm[2:]:<3} n {n:>5}  slope {b:+.3f} ± {bc:.3f}")
    return L


def coef_lines(models, Y):
    L = ["", f"== Coefficients, the last fold (trained on seasons before {u.season_name(Y)}; "
             "points of home margin per unit)"]
    for route, mm in models[Y].items():
        if mm:
            a, b = mm["probit"]
            L.append(f"  {route:<6} n {mm['n']:>6}  sigma {mm['sigma']:.2f}  probit "
                     f"{a:+.3f} + {b:.4f} m (1/b {1 / b:.2f})  int {mm['intercept']:+.2f}  "
                     + "  ".join(f"{f} {v:+.3f}" for f, v in zip(mm["features"], mm["coef"])))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", nargs="+", type=int, default=u.SEASONS)
    ap.add_argument("--cache", default=pav.DEFAULT_CACHE)
    a = ap.parse_args(argv)
    d, models = run(a.seasons, a.cache)
    m = priced(d)
    L = ["Margin model (research/margin_model.py): v8's inputs and routing, OLS on the "
         "final margin, P = Phi(m / sigma). Walk-forward; ± is 95%.", ""]
    L += q1_outcomes(d) + q1_close(m) + q2_spread(d, m) + q3_ats(m)
    L += coef_lines(models, max(a.seasons))
    text = "\n".join(L)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as fh:
        fh.write(text + "\n")
    d.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
