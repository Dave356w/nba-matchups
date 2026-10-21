#!/usr/bin/env python3
"""Frozen earlier-season band medians vs 2025-26 final margins and DK lines.

Research forecast benchmark only: no selections, wagering outcomes, or changes
to production. Ten-percentage-point bands are fixed before scoring. Both home
WP bands and favorite WP bands are evaluated; neither is tuned on the holdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import wp_mov_calibration as wm  # noqa: E402

OUT = ROOT / "diagnostics" / "median_margin_benchmark"
INPUT = ROOT / "diagnostics" / "wp_mov_calibration" / "oos_probabilities.csv.gz"
METHODS = ("home_band_median", "favorite_band_median", "fixed_13_5", "dk_open", "dk_close")


def band_index(p, perspective):
    p = np.asarray(p, float)
    wm.probit(p)  # reject missing, infinite and endpoint probabilities
    if perspective == "home":
        edges = np.arange(11) / 10
    elif perspective == "favorite":
        p = np.maximum(p, 1 - p)
        edges = np.arange(5, 11) / 10
    else:
        raise ValueError("Unknown probability perspective")
    return np.searchsorted(edges, p, side="right") - 1


def fit_bands(train, test_year, perspective):
    wm.validate_input(train)
    if (train["year"] >= test_year).any():
        raise ValueError("Median training includes the holdout season")
    p = train["p_home"].to_numpy(float)
    margin = train["margin"].to_numpy(float)
    if perspective == "favorite":
        margin = np.where(p >= .5, margin, -margin)
    groups = pd.DataFrame(dict(band=band_index(p, perspective), margin=margin))
    result = groups.groupby("band")["margin"].agg(["size", "median", "mean"]).reset_index()
    result = result.rename(columns={"size": "train_n"})
    start = 0 if perspective == "home" else .5
    result["p_low"] = start + result["band"] / 10
    result["p_high"] = result["p_low"] + .1
    result["perspective"] = perspective
    result["train_max_year"] = int(train["year"].max())
    return result


def predict_bands(p, fitted, perspective):
    """Freeze the lookup; an empty training band uses the nearest nonempty band.

    That fallback depends only on training data. Its use is recorded per game.
    In the committed 2025-26 experiment every evaluated band has training data.
    """
    p = np.asarray(p, float)
    ix = band_index(p, perspective)
    available = fitted["band"].to_numpy(int)
    if not len(available):
        raise ValueError("No training bands")
    nearest = available[np.abs(ix[:, None] - available).argmin(axis=1)]
    lookup = fitted.set_index("band")["median"]
    pred = lookup.loc[nearest].to_numpy(float)
    if perspective == "favorite":
        pred = np.where(p >= .5, pred, -pred)
    return pred, nearest != ix


def matched_market(test, recon, openings, test_year):
    """One common complete-case sample and exact score/probability checks."""
    wm.validate_input(test)
    if not test["year"].eq(test_year).all():
        raise ValueError("Probability test input contains another season")
    r = recon[(recon["season"] == test_year) & recon["close_book"].eq("dk")].copy()
    audit = {"test_year": int(test_year), "oos_holdout_games": len(test),
             "dk_reconstructed_rows": len(r)}
    if r["game_id"].duplicated().any() or openings["game_id"].duplicated().any():
        raise ValueError("Duplicate market game identity")
    r = r.merge(openings[["game_id", "book", "open_spread"]], on="game_id",
                how="left", validate="one_to_one")
    numeric = ["close_spread", "open_spread", "home_pts", "away_pts", "p_home"]
    for name in numeric:
        r[name] = pd.to_numeric(r[name], errors="coerce")
    eligible = r["book"].eq("dk") & np.isfinite(r[numeric]).all(axis=1)
    audit["dk_complete_open_close_scores"] = int(eligible.sum())
    r = r[eligible].rename(columns={"p_home": "ledger_p_home"})
    p = test.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["slate_date"] = p["date"].dt.strftime("%Y-%m-%d")
    cols = ["game_id", "slate_date", "home", "away", "ledger_p_home", "open_spread",
            "close_spread", "home_pts", "away_pts", "tip_utc", "snapshot_utc"]
    p = p.merge(r[cols], on=["slate_date", "home", "away"], how="inner", validate="one_to_one")
    if p.empty:
        raise ValueError("No matched market observations")
    if not np.array_equal(p["margin"].to_numpy(), (p["home_pts"] - p["away_pts"]).to_numpy()):
        raise ValueError("Final scores disagree between probability and market sources")
    max_diff = float(np.max(np.abs(p["p_home"] - p["ledger_p_home"])))
    if max_diff > .000051:
        raise ValueError("Ledger WP disagrees beyond its four-decimal rounding")
    # Quoted home spread is the handicap added to home score, not home margin.
    p["dk_open"] = -p["open_spread"]
    p["dk_close"] = -p["close_spread"]
    audit.update(matched_games=len(p), unmatched_complete_dk_rows=len(r) - len(p),
                 max_probability_rounding_difference=max_diff,
                 first_date=p["slate_date"].min(), last_date=p["slate_date"].max(),
                 reconstructed_snapshot_timestamps=int(p["snapshot_utc"].notna().sum()))
    return p, audit


def median_metrics(g):
    y, p = g["margin"].to_numpy(float), g["pred_margin"].to_numpy(float)
    error = p - y
    return dict(n=len(g), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt((error ** 2).mean())), bias=float(error.mean()),
                median_error=float(np.median(error)), below=float((y < p).mean()),
                equal=float((y == p).mean()), at_or_below=float((y <= p).mean()))


def evaluate(probabilities, recon, openings, test_year=2026):
    wm.validate_input(probabilities)
    train = probabilities[probabilities["year"] < test_year].copy()
    test = probabilities[probabilities["year"] == test_year].copy()
    fitted = {s: fit_bands(train, test_year, s) for s in ("home", "favorite")}
    common, audit = matched_market(test, recon, openings, test_year)
    audit.update(training_games=len(train), training_seasons=sorted(map(int, train["year"].unique())),
                 training_max_year=int(train["year"].max()), holdout_routes=common["route"].value_counts().to_dict())
    for s in ("home", "favorite"):
        common[f"{s}_band_median"], fallback = predict_bands(common["p_home"], fitted[s], s)
        audit[f"{s}_empty_band_fallback_games"] = int(fallback.sum())
    common["fixed_13_5"] = 13.5 * wm.probit(common["p_home"])
    predictions = []
    for method in METHODS:
        g = common.copy()
        g["method"], g["pred_margin"] = method, g[method]
        predictions.append(g)
    pred = pd.concat(predictions, ignore_index=True)
    summary = pd.DataFrame([dict(method=m, **median_metrics(g)) for m, g in pred.groupby("method")])
    comparisons = []
    for method in METHODS[:2]:
        for benchmark in METHODS[2:]:
            comparisons.append(dict(method=method, benchmark=benchmark,
                                    **wm.paired_bootstrap(pred[pred["method"] == method],
                                                         pred[pred["method"] == benchmark])))
    # Descriptive holdout medians are never fed back into the training lookup.
    rows = []
    home_ix = band_index(common["p_home"], "home")
    for band in sorted(set(home_ix)):
        g = common[home_ix == band]
        rows.append(dict(p_low=band / 10, p_high=(band + 1) / 10, n=len(g),
                         actual_median=float(g["margin"].median()),
                         frozen_home_median=float(g["home_band_median"].iloc[0]),
                         actual_mean=float(g["margin"].mean()),
                         market_open_median=float(g["dk_open"].median()),
                         market_close_median=float(g["dk_close"].median())))
    return (pred, pd.concat(fitted.values(), ignore_index=True), summary,
            pd.DataFrame(comparisons), pd.DataFrame(rows), audit)


def report(fitted, metrics, comparisons, bands, audit):
    a = audit
    L = ["# Median margin vs DraftKings: 2025–26 forecast benchmark", "",
         f"Frozen training: **{a['training_games']:,}** earlier-season OOS probability/outcome pairs, "
         f"last BBR year {a['training_max_year']}. Held-out common sample: **{a['matched_games']:,} games**, "
         f"{a['first_date']} to {a['last_date']}.", "",
         "Two fixed 10-percentage-point lookup candidates: actual home-margin medians by home WP, "
         "and favorite-margin medians by favorite WP (converted back to home margin). "
         "Both lookups are fitted only on seasons before 2025–26. No bands, thresholds, or candidates "
         "are selected using holdout performance. This is retrospective research: prior analysis had "
         "already inspected this season, so chronological training exclusion is not an untouched prospective holdout.", "",
         "## Held-out forecast accuracy", "",
         "MAE is the primary median-forecast score (equivalent to twice the 0.5-quantile loss). "
         "RMSE and average bias are secondary diagnostics. All values are final home-minus-away points, "
         "including overtime. Market forecast margin = minus the quoted home handicap.", "",
         "| Forecast | Games | MAE | RMSE | Mean bias | Median error |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in metrics.itertuples(index=False):
        L.append(f"| {r.method} | {r.n} | {r.mae:.4f} | {r.rmse:.4f} | {r.bias:+.4f} | {r.median_error:+.4f} |")
    L += ["", "## Paired uncertainty", "",
          "Candidate minus benchmark; negative favors the candidate. Paired date-cluster bootstrap: "
          "2,000 draws, seed 20261021; 95% percentile intervals. Intervals do not capture cross-date serial "
          "dependence, model-development selection, or multiple-comparison effects.", "",
          "| Candidate | Benchmark | Delta MAE [95% CI] | Delta RMSE [95% CI] |",
          "|---|---|---:|---:|"]
    for r in comparisons.itertuples(index=False):
        L.append(f"| {r.method} | {r.benchmark} | {r.delta_mae:+.4f} "
                 f"[{r.mae_ci_low:+.4f}, {r.mae_ci_high:+.4f}] | {r.delta_rmse:+.4f} "
                 f"[{r.rmse_ci_low:+.4f}, {r.rmse_ci_high:+.4f}] |")
    L += ["", "## Marginal median calibration", "",
          "For a calibrated median with discrete scores, P(actual < forecast) ≤ 50% ≤ "
          "P(actual ≤ forecast). A 50% value inside that interval is a necessary aggregate diagnostic, "
          "not proof of conditional calibration. Exact equality is reported separately.", "",
          "| Forecast | Actual below | Equal | Actual at or below |",
          "|---|---:|---:|---:|"]
    for r in metrics.itertuples(index=False):
        L.append(f"| {r.method} | {100*r.below:.2f}% | {100*r.equal:.2f}% | {100*r.at_or_below:.2f}% |")
    L += ["", "## Frozen earlier-season lookup", "",
          "Rare bands are left visible, not silently combined. Empty-band fallback uses the nearest "
          "nonempty training band; no evaluated game needed it in this experiment.", "",
          "| Perspective | WP band | Training games | Training median margin |",
          "|---|---|---:|---:|"]
    for r in fitted.itertuples(index=False):
        L.append(f"| {r.perspective} | {100*r.p_low:.0f}–{100*r.p_high:.0f}% | {r.train_n} | {r.median:+.2f} |")
    L += ["", "## Descriptive holdout by home WP band", "",
          "Holdout outcomes below are diagnostics, not refitted forecast values. Market medians summarize "
          "varying game-specific lines inside each band; the accuracy table scores every original line.", "",
          "| Home WP | Games | Frozen home-band forecast | Actual median | DK opening median | DK closing median |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in bands.itertuples(index=False):
        L.append(f"| {100*r.p_low:.0f}–{100*r.p_high:.0f}% | {r.n} | {r.frozen_home_median:+.2f} | "
                 f"{r.actual_median:+.2f} | {r.market_open_median:+.2f} | {r.market_close_median:+.2f} |")
    L += ["", "## Provenance and limitations", "",
          f"- DraftKings reconstructed rows: {a['dk_reconstructed_rows']}; complete DK opening/closing "
          f"lines and scores: {a['dk_complete_open_close_scores']}; matched OOS rows: {a['matched_games']}; "
          f"unmatched complete DK rows: {a['unmatched_complete_dk_rows']}.",
          f"- Probability-model fits and composite weights precede their own seasons. The median lookup's "
          f"latest training year is {a['training_max_year']}, before test year {a['test_year']}.",
          f"- Final scores agree exactly across sources. Maximum difference between reconstructed CSV WP "
          f"and archived OOS WP: {a['max_probability_rounding_difference']:.8f}, within four-decimal rounding.",
          f"- Matched route counts: {a['holdout_routes']}. Empty-band fallback games: "
          f"home={a['home_empty_band_fallback_games']}, favorite={a['favorite_empty_band_fallback_games']}.",
          f"- Actual reconstructed snapshot timestamps: {a['reconstructed_snapshot_timestamps']}. "
          "The repo archives opening and closing endpoints, not a timestamp-aligned historical model "
          "prediction at each opening line. Model WP can include pregame information unavailable at "
          "the opening endpoint; the opening comparison is not an equal-information-time experiment.",
          "- This is forecast accuracy and midpoint calibration only. It does not estimate a full "
          "conditional margin distribution or validate predictions at arbitrary margin thresholds.",
          "- Games before both teams have 10 prior games, the first month without DK coverage, and "
          "2019–20 remain outside the relevant source samples. Training mixes historical model routes "
          "and eras; the target season's routing mix may differ.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probabilities", type=Path, default=INPUT)
    ap.add_argument("--reconstructed", type=Path, default=ROOT / "data" / "nba_reconstructed.csv")
    ap.add_argument("--openings", type=Path, default=ROOT / "diagnostics" / "ats_signal" / "open_spreads.csv")
    ap.add_argument("--test-year", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    results = evaluate(pd.read_csv(a.probabilities, parse_dates=["date"]),
                       pd.read_csv(a.reconstructed), pd.read_csv(a.openings), a.test_year)
    pred, fitted, summary, comparisons, bands, audit = results
    text = report(fitted, summary, comparisons, bands, audit)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("frozen_bands", fitted), ("metrics", summary),
                        ("comparisons", comparisons), ("holdout_bands", bands)):
        frame.to_csv(a.out / f"{name}.csv", index=False)
    pred.to_csv(a.out / "predictions.csv.gz", index=False,
                compression={"method": "gzip", "mtime": 0})
    audit["source_commit"] = os.environ.get("GITHUB_SHA", "local")
    (a.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (a.out / "REPORT.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
