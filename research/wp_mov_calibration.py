#!/usr/bin/env python3
"""Research only: nested walk-forward win probability -> final home margin.

The probability model is fitted on seasons before Y. The margin mapping for
Y is fitted ONLY on earlier seasons' out-of-sample probabilities and scores.
The first probability season is a mapping warm-up, excluded from all paired
comparisons. No market data, production models, tags or ledgers are changed.
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
import history_backtest as hb  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "wp_mov_calibration")
METHODS = ("fixed_13_5", "scale_only", "affine")
KEYS = ["year", "date", "home", "away"]


def probit(p):
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or ((p <= 0) | (p >= 1)).any():
        raise ValueError("Probabilities must be finite and strictly between 0 and 1")
    return np.array([NormalDist().inv_cdf(float(x)) for x in p])


def fit_mapping(z, margin, method):
    z, margin = np.asarray(z, float), np.asarray(margin, float)
    if method == "fixed_13_5":
        return 0.0, 13.5
    if not len(z) or not np.isfinite(z).all() or not np.isfinite(margin).all():
        raise ValueError("Mapping fit requires finite training observations")
    if method == "scale_only":
        if z @ z <= 0:
            raise ValueError("No training variation in probability")
        return 0.0, float(z @ margin / (z @ z))
    if method == "affine":
        X = np.column_stack([np.ones(len(z)), z])
        if np.linalg.matrix_rank(X) < 2:
            raise ValueError("No training variation in probability")
        a, b = np.linalg.lstsq(X, margin, rcond=None)[0]
        return float(a), float(b)
    raise ValueError(f"Unknown mapping: {method}")


def score_table(years, load_logs=None):
    """Exact BBR game identity; final scores include overtime. Outcome only."""
    load_logs = load_logs or hb.nc.load_logs
    rows = []
    for year in sorted(set(years)):
        for team, log in load_logs(int(year)).items():
            for r in log[log["home"]].itertuples(index=False):
                rows.append(dict(year=int(year), date=pd.Timestamp(r.date),
                                 home=team, away=r.opp,
                                 margin=float(r.pts - r.opp_pts)))
    result = pd.DataFrame(rows)
    if result.duplicated(KEYS).any():
        raise ValueError("Duplicate home game in scoring margins")
    return result


def attach_margins(predictions, scores):
    p = predictions.copy()
    p["date"] = pd.to_datetime(p["date"])
    scores = scores.copy()
    scores["date"] = pd.to_datetime(scores["date"])
    p = p.merge(scores, on=KEYS, how="left", validate="one_to_one")
    if not np.isfinite(p["margin"]).all():
        raise ValueError("Missing or nonfinite final scoring margin")
    if (p["margin"] == 0).any() or not np.array_equal(
            p["margin"].to_numpy() > 0, p["win"].to_numpy() == 1):
        raise ValueError("Final margin disagrees with probability-fold outcome")
    return p


def build_probabilities(years):
    folds = hb.c.build_folds(years, with_e=False)
    rows = []
    for year, f in sorted(folds.items()):
        p = hb.routed(f, hb.V6, hb.AV6)
        g = f.test[["year", "date", "home", "away", "win", "route", "phase"]].copy()
        g["p_home"] = p
        # V5-based route labels can hide a v6 missing-term fallback.
        miss = ((g["route"].eq("base") & ~hb.c.finite(f.test, hb.V6)) |
                (g["route"].eq("avail") & ~hb.c.finite(f.test, hb.AV6)))
        g.loc[miss, "route"] = "v4"
        train_years = set(f.v4_train["year"]) | set(f.base_train["year"])
        if f.avail_train is not None:
            train_years |= set(f.avail_train["year"])
        if max(train_years) >= year or max(f.weights["years"]) >= year:
            raise ValueError("Probability model training leaks into test season")
        g["probability_train_max_year"] = max(train_years)
        g["weights_train_max_year"] = max(f.weights["years"])
        rows.append(g)
        print(f"probabilities {year}: {len(g)} games, previous seasons only", flush=True)
    return attach_margins(pd.concat(rows, ignore_index=True), score_table(years))


def validate_input(d):
    required = KEYS + ["p_home", "margin", "route", "phase", "win",
                       "probability_train_max_year", "weights_train_max_year"]
    if not set(required) <= set(d):
        raise ValueError("Input missing probability provenance or outcome columns")
    if d.empty or d.duplicated(KEYS).any():
        raise ValueError("Empty or duplicate probability observations")
    for key in ("probability_train_max_year", "weights_train_max_year"):
        if not np.isfinite(d[key]).all() or (d[key] >= d["year"]).any():
            raise ValueError("Probability model training leaks into test season")
    if not np.isfinite(d["margin"]).all() or (d["margin"] == 0).any():
        raise ValueError("Invalid final margins")
    if not np.array_equal(d["margin"].to_numpy() > 0, d["win"].to_numpy() == 1):
        raise ValueError("Margin and win outcome disagree")
    probit(d["p_home"])


def walk_forward(d):
    validate_input(d)
    d = d.sort_values(KEYS).reset_index(drop=True).copy()
    d["z"] = probit(d["p_home"])
    predictions, coefficients = [], []
    for year in sorted(d["year"].unique()):
        train, test = d[d["year"] < year], d[d["year"] == year]
        if train.empty:
            continue
        for method in METHODS:
            a, b = fit_mapping(train["z"], train["margin"], method)
            g = test.copy()
            g["method"], g["pred_margin"] = method, a + b * g["z"]
            g["mapping_train_max_year"] = int(train["year"].max())
            predictions.append(g)
            coefficients.append(dict(year=int(year), method=method,
                                     train_n=len(train), train_max_year=int(train["year"].max()),
                                     intercept=a, scale=b))
    if not predictions:
        raise ValueError("Need at least two out-of-sample probability seasons")
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(coefficients)


def metrics(g):
    error = g["pred_margin"].to_numpy(float) - g["margin"].to_numpy(float)
    a, b = fit_mapping(g["pred_margin"], g["margin"], "affine")
    return dict(n=len(g), bias=float(error.mean()), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.mean(error ** 2))),
                actual_on_pred_intercept=a, actual_on_pred_slope=b)


def paired_bootstrap(g, base, draws=2000, seed=20261021):
    """Paired date-cluster bootstrap, candidate minus fixed; negative is better.

    Resample dates with all games kept together. Report percentile 95% CIs.
    This does not capture serial dependence across dates or model-selection risk.
    """
    a = g[KEYS + ["pred_margin", "margin"]].merge(
        base[KEYS + ["pred_margin"]], on=KEYS, validate="one_to_one", suffixes=("", "_base"))
    if len(a) != len(g) or len(a) != len(base):
        raise ValueError("Comparison samples must match exactly")
    e = a["pred_margin"] - a["margin"]
    eb = a["pred_margin_base"] - a["margin"]
    x = pd.DataFrame(dict(date=a["date"], n=1, ae=np.abs(e), ab=np.abs(eb),
                          se=e ** 2, sb=eb ** 2)).groupby("date").sum().to_numpy(float)
    rng = np.random.default_rng(seed)
    maes, rmses = [], []
    for start in range(0, draws, 100):
        k = rng.integers(0, len(x), size=(min(100, draws - start), len(x)))
        sums = x[k].sum(axis=1)
        maes.extend((sums[:, 1] - sums[:, 2]) / sums[:, 0])
        rmses.extend(np.sqrt(sums[:, 3] / sums[:, 0]) - np.sqrt(sums[:, 4] / sums[:, 0]))
    return dict(delta_mae=float(np.abs(e).mean() - np.abs(eb).mean()),
                mae_ci_low=float(np.quantile(maes, .025)), mae_ci_high=float(np.quantile(maes, .975)),
                delta_rmse=float(np.sqrt((e ** 2).mean()) - np.sqrt((eb ** 2).mean())),
                rmse_ci_low=float(np.quantile(rmses, .025)), rmse_ci_high=float(np.quantile(rmses, .975)),
                dates=len(x), bootstrap_draws=draws)


def summarize(pred):
    rows = []
    for method, g in pred.groupby("method"):
        rows.append(dict(scope="pooled", group="all", method=method, **metrics(g)))
        for year, s in g.groupby("year"):
            rows.append(dict(scope="season", group=str(year), method=method, **metrics(s)))
        for route, s in g.groupby("route"):
            rows.append(dict(scope="route", group=route, method=method, **metrics(s)))
        for lo, hi, label in hb.PHASES:
            s = g[(g["phase"] >= lo) & (g["phase"] < hi)]
            if len(s) >= 20:
                rows.append(dict(scope="phase", group=label, method=method, **metrics(s)))
        fav = np.maximum(g["p_home"], 1 - g["p_home"])
        for lo, hi in ((.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)):
            s = g[(fav >= lo) & (fav < hi)]
            if len(s) >= 20:
                rows.append(dict(scope="strength", group=f"{lo:.1f}-{min(hi, 1):.1f}",
                                 method=method, **metrics(s)))
    return pd.DataFrame(rows)


def reliability(pred):
    rows = []
    for method, g in pred.groupby("method"):
        for lo in np.arange(0, 1, .1):
            hi = lo + .1
            s = g[(g["p_home"] >= lo) & (g["p_home"] < hi)]
            if len(s) < 20:
                continue
            rows.append(dict(method=method, p_low=lo, p_high=hi, n=len(s),
                             p_mean=s["p_home"].mean(), predicted=s["pred_margin"].mean(),
                             actual=s["margin"].mean(), bias=(s["pred_margin"] - s["margin"]).mean()))
    return pd.DataFrame(rows)


def render_report(d, pred, coef, summary, comparisons, bins):
    warmup = int(d["year"].min())
    L = ["# WP to MOV calibration: nested walk-forward", "",
         "Research only. Final home points minus away points, including overtime. No market data.",
         f"Probability sample: {len(d):,} games across {d['year'].nunique()} seasons. "
         f"Mapping warm-up: {hb.season_name(warmup)} ({int((d['year'] == warmup).sum()):,} games).",
         f"Matched evaluation: {int((pred['method'] == METHODS[0]).sum()):,} games across "
         f"{pred['year'].nunique()} seasons. Games before both teams have 10 prior games are excluded.", "",
         "Every WP uses composite weights and logits fitted on earlier seasons only. "
         "For season Y, margin coefficients use ONLY out-of-sample WP/outcome pairs from seasons < Y. "
         "No same-season or in-sample WP is used to fit a mapping. "
         "The earliest WP season is excluded equally for all methods.", "",
         "Mappings, z = inverse standard normal CDF(WP): fixed = 13.5*z; "
         "scale_only = b*z; affine = a+b*z (ordinary least squares). "
         "The fitted b is a conditional-mean conversion scale, not an independently estimated residual SD. "
         "Bias means prediction minus actual; negative comparison deltas favor the fitted mapping.", "",
         "## Matched out-of-sample results", "",
         "| Method | Games | MAE | RMSE | Bias | Actual-on-pred intercept | Actual-on-pred slope |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary[summary["scope"] == "pooled"].itertuples(index=False):
        L.append(f"| {r.method} | {r.n} | {r.mae:.4f} | {r.rmse:.4f} | {r.bias:+.4f} | "
                 f"{r.actual_on_pred_intercept:+.4f} | {r.actual_on_pred_slope:.4f} |")
    L += ["", "## Paired changes versus fixed 13.5", "",
          "95% percentile CIs from paired date-cluster bootstrap (2,000 draws, seed 20261021). "
          "All games on a sampled date stay together; cross-date dependence and feature-selection risk are not captured.", "",
          "| Method | Delta MAE [95% CI] | Delta RMSE [95% CI] |",
          "|---|---:|---:|"]
    for r in comparisons.itertuples(index=False):
        L.append(f"| {r.method} | {r.delta_mae:+.4f} [{r.mae_ci_low:+.4f}, {r.mae_ci_high:+.4f}] | "
                 f"{r.delta_rmse:+.4f} [{r.rmse_ci_low:+.4f}, {r.rmse_ci_high:+.4f}] |")
    for scope, title in (("season", "By season"), ("route", "By model route"),
                         ("phase", "By season phase"), ("strength", "By favorite probability")):
        L += ["", f"## {title}", "", "| Group | Method | Games | MAE | RMSE | Bias |",
              "|---|---|---:|---:|---:|---:|"]
        for r in summary[summary["scope"] == scope].itertuples(index=False):
            label = hb.season_name(int(r.group)) if scope == "season" else r.group
            L.append(f"| {label} | {r.method} | {r.n} | {r.mae:.3f} | {r.rmse:.3f} | {r.bias:+.3f} |")
    L += ["", "## Mapping coefficients fitted before each test season", "",
          "| Test season | Method | Training games | Last training season (BBR year) | Intercept | Scale |",
          "|---|---|---:|---:|---:|---:|"]
    for r in coef.itertuples(index=False):
        L.append(f"| {hb.season_name(r.year)} | {r.method} | {r.train_n} | {r.train_max_year} | "
                 f"{r.intercept:+.4f} | {r.scale:.4f} |")
    L += ["", "## Reliability by home win probability", "",
          "Descriptive averages on the identical held-out games; not used to fit or select a mapping.", "",
          "| WP band | Method | Games | Predicted home margin | Actual home margin | Bias |",
          "|---|---|---:|---:|---:|---:|"]
    for r in bins.itertuples(index=False):
        L.append(f"| {r.p_low:.1f}-{r.p_high:.1f} | {r.method} | {r.n} | "
                 f"{r.predicted:+.3f} | {r.actual:+.3f} | {r.bias:+.3f} |")
    # Descriptive deployment coefficients: explicitly not the held-out result.
    L += ["", "## Full-sample research fit (not an out-of-sample score)", ""]
    for method in METHODS[1:]:
        a, b = fit_mapping(probit(d["p_home"]), d["margin"], method)
        L.append(f"- {method}: a={a:+.4f}, b={b:.4f}; all {len(d)} OOS WP/outcome pairs. "
                 "Not deployed and not used to score any season above.")
    L += ["", "## Limits", "",
          "This tests expected final margin, not a complete score/margin distribution, cover probabilities, "
          "or financial returns. A calibrated WP alone does not identify the mean margin. "
          "An affine mean mapping need not correspond to a Normal distribution whose win probability equals WP. "
          "Model choices predate this test but were developed retrospectively; walk-forward fitting is not "
          "an untouched prospective holdout. Availability routing is concentrated in recent seasons, "
          "so route comparisons are also era comparisons. The 2019-20 bubble season is excluded, "
          "as in the existing history backtest. Subgroup tables are exploratory, not selection rules.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=hb.SEASONS)
    ap.add_argument("--input", help="Previously generated OOS probability CSV with provenance and margins")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    d = pd.read_csv(a.input, parse_dates=["date"]) if a.input else build_probabilities(a.seasons)
    pred, coef = walk_forward(d)
    summary, bins = summarize(pred), reliability(pred)
    base = pred[pred["method"] == METHODS[0]]
    comparisons = pd.DataFrame([dict(method=m, **paired_bootstrap(pred[pred["method"] == m], base))
                                for m in METHODS[1:]])
    report = render_report(d, pred, coef, summary, comparisons, bins)
    os.makedirs(a.out, exist_ok=True)
    for name, table in (("oos_probabilities", d), ("predictions", pred), ("coefficients", coef),
                        ("metrics", summary), ("comparisons", comparisons), ("reliability", bins)):
        table.to_csv(os.path.join(a.out, name + ".csv"), index=False)
    with open(os.path.join(a.out, "REPORT.md"), "w") as f:
        f.write(report)
    with open(os.path.join(a.out, "metadata.json"), "w") as f:
        json.dump(dict(seasons=sorted(map(int, d["year"].unique())),
                       mappings=list(METHODS), bootstrap_seed=20261021,
                       bootstrap_draws=2000, bootstrap_cluster="date",
                       margin="home final points minus away final points; includes overtime",
                       source_commit=os.environ.get("GITHUB_SHA", "local")), f, indent=2)
    print(report, flush=True)


if __name__ == "__main__":
    main()
