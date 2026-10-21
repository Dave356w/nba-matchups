#!/usr/bin/env python3
"""ATS cover rates for frozen mean/median margin forecasts on the DK holdout.

Use the identical matched 2025-26 sample as median_margin_benchmark. A forecast
above the market-implied home margin selects home; below selects away; equality
passes. Pushes are excluded from the cover-rate denominator. No threshold scan,
price assumptions, return calculations, or production changes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import median_margin_benchmark as mm  # noqa: E402

METHODS = ("home_band_median", "favorite_band_median", "fixed_13_5",
           "fitted_scale", "fitted_affine", "dk_open", "dk_close")
LABELS = {"home_band_median": "Home-band median", "favorite_band_median": "Favorite-band median",
          "fixed_13_5": "Fixed 13.5", "fitted_scale": "Fitted mean scale",
          "fitted_affine": "Fitted mean intercept + scale", "dk_open": "DK opening forecast",
          "dk_close": "DK closing forecast"}


def select_and_grade(margin, forecast, home_spread):
    """Signed home handicap is added to home margin; mirror-invariant grading."""
    m, f, s = (np.asarray(x, float) for x in (margin, forecast, home_spread))
    if m.shape != f.shape or m.shape != s.shape or not all(np.isfinite(x).all() for x in (m, f, s)):
        raise ValueError("Scoring requires same-shaped finite observations")
    edge = f + s
    side = np.where(np.isclose(edge, 0, atol=1e-10, rtol=0), 0, np.sign(edge)).astype(int)
    adjusted = m + s
    selected = side != 0
    pushed = selected & np.isclose(adjusted, 0, atol=1e-10, rtol=0)
    result = np.where(~selected, "pass", np.where(pushed, "push",
                      np.where(side * adjusted > 0, "cover", "noncover")))
    return side, result


def summary(result):
    result = np.asarray(result)
    covers, losses = int((result == "cover").sum()), int((result == "noncover").sum())
    pushes, passes = int((result == "push").sum()), int((result == "pass").sum())
    n = covers + losses
    low, high = mm.wm.hb.c.wilson(covers, n) if n else (np.nan, np.nan)
    return dict(games=len(result), covers=covers, noncovers=losses, pushes=pushes, passes=passes,
                selected=covers + losses + pushes, graded=n,
                cover_rate=covers / n if n else np.nan, ci_low=low, ci_high=high)


def fitted_means(probabilities, test_year):
    mm.wm.validate_input(probabilities)
    train = probabilities[probabilities["year"] < test_year]
    if train.empty:
        raise ValueError("No earlier-season OOS training observations")
    out = []
    for name, method in (("fitted_scale", "scale_only"), ("fitted_affine", "affine")):
        a, b = mm.wm.fit_mapping(mm.wm.probit(train["p_home"]), train["margin"], method)
        out.append(dict(method=name, intercept=a, scale=b, train_n=len(train),
                        train_max_year=int(train["year"].max())))
    return pd.DataFrame(out)


def timing(method, target):
    if method == "dk_close" and target == "open":
        return "lookahead_only"
    if (method, target) in (("dk_open", "open"), ("dk_close", "close")):
        return "same_line_all_pass"
    if target == "open":
        return "reconstructed_WP_not_opening_time_aligned"
    return "earlier_opening_line" if method == "dk_open" else "reconstructed_pregame_WP"


def evaluate(base, coefficients, test_year=2026):
    base = base.copy()
    if base.empty or not base["year"].eq(test_year).all() or base.duplicated(mm.wm.KEYS).any():
        raise ValueError("Need unique games from only the held-out season")
    mm.wm.validate_input(base)
    for r in coefficients.itertuples(index=False):
        if r.train_max_year >= test_year:
            raise ValueError("Mean mapping training includes holdout season")
        base[r.method] = r.intercept + r.scale * mm.wm.probit(base["p_home"])
    rows, picks = [], []
    for target in ("open", "close"):
        for method in METHODS:
            side, result = select_and_grade(base["margin"], base[method], base[f"{target}_spread"])
            rows.append(dict(target=target, method=method, timing=timing(method, target),
                             home_selections=int((side == 1).sum()), away_selections=int((side == -1).sum()),
                             **summary(result)))
            g = base[mm.wm.KEYS + ["game_id", "margin", "p_home"]].copy()
            g["target"], g["method"], g["timing"] = target, method, timing(method, target)
            g["forecast_margin"], g["home_spread"] = base[method], base[f"{target}_spread"]
            g["side"], g["result"] = side, result
            picks.append(g)
    return pd.DataFrame(rows), pd.concat(picks, ignore_index=True)


def report(scores, coefficients, source_commit):
    L = ["# ATS cover-rate comparison: frozen forecasts, 2025–26", "",
         "The same 964 matched DraftKings games as the median-margin benchmark. Median bands and "
         "mean-conversion coefficients are fitted using 8,438 OOS WP/outcome pairs from seasons before "
         "2025–26. These cover rates are retrospective diagnostics, not an untouched prospective test.", "",
         "Selection rule: compare the predicted signed home margin with minus the quoted home handicap. "
         "Forecast above that margin selects home ATS; below selects away ATS; exact equality passes. "
         "Numerical equality tolerance is 1e-10 points. Cover rate = covers / (covers + noncovers); "
         "pushes and passes are reported separately. No minimum edge is added or optimized.", ""]
    for target in ("open", "close"):
        L += ["", f"## At DraftKings {target}ing lines" if target == "open" else "## At DraftKings closing lines", "",
              "| Forecast | Covers–noncovers–pushes | Passes | Cover rate | Wilson 95% interval |",
              "|---|---:|---:|---:|---:|"]
        for r in scores[scores["target"] == target].itertuples(index=False):
            label = LABELS[r.method] + (" **(LOOK-AHEAD ONLY)**" if r.timing == "lookahead_only" else "")
            rate = f"{100*r.cover_rate:.2f}%" if r.graded else "—"
            ci = f"{100*r.ci_low:.2f}–{100*r.ci_high:.2f}%" if r.graded else "—"
            L.append(f"| {label} | {r.covers}–{r.noncovers}–{r.pushes} | {r.passes} | {rate} | {ci} |")
    L += ["", "## Timing and interpretation", "",
          "- **The closing forecast at the opening line uses future information.** Its result is a "
          "look-ahead diagnostic only and must not be ranked as an available opening-time strategy.",
          "- The opening forecast at its own line, and the closing forecast at its own line, have zero "
          "forecast disagreement on every game. All 964 pass; a 50% rate must not be invented for them.",
          "- Opening and closing lines differ on 731 games and match on 233. The opening-line forecast "
          "evaluated at the closing line therefore has 731 selections and 233 passes.",
          "- Reconstructed model WP is pregame but has no archived opening-time snapshot. Opening-line "
          "model rates are not timestamp-aligned opening-time performance. Closing-line comparisons "
          "also remain reconstructed historical research, not native live records.",
          "- All recorded lines in this sample are half points and final margins are integer points; "
          "there are no outcome pushes. The grader nevertheless handles pushes and passes independently.",
          "- Lower margin MAE and higher ATS cover rate are different objectives. The median forecasts' "
          "lower average home bias does not establish higher ATS accuracy.",
          "- Wilson intervals treat graded games as independent Bernoulli observations; they do not "
          "account for same-date/serial dependence, model development, multiple comparisons, or varying "
          "selection samples. Cover rate alone does not establish profitability; prices are not evaluated.", "",
          "## Frozen mean-conversion coefficients", "",
          "The two fitted mean mappings supplement the five forecasts in the earlier median benchmark. "
          "They use the same prior-season-only protocol from the original WP-to-MOV study.", "",
          "| Mapping | Training games | Last training BBR year | Intercept | Scale |",
          "|---|---:|---:|---:|---:|"]
    for r in coefficients.itertuples(index=False):
        L.append(f"| {LABELS[r.method]} | {r.train_n} | {r.train_max_year} | {r.intercept:+.4f} | {r.scale:.4f} |")
    L += ["", "## Reproduction", "",
          "```bash", "python research/median_ats_comparison.py --out /tmp/median-ats-comparison", "```", "",
          "Outputs: `ATS_REPORT.md`, `ats_cover_rates.csv`, `ats_mean_coefficients.csv`, and "
          "`ats_predictions.csv.gz`. The per-game output includes target line, forecast, selected side, "
          "result, and timing label. The underlying median lookup values and data audit are in "
          "[REPORT.md](REPORT.md). Production code, models and ledgers are unchanged.", "",
          f"Source commit: `{source_commit}`.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", type=Path, help="Existing median benchmark per-game CSV (optional)")
    ap.add_argument("--out", type=Path, default=mm.OUT)
    a = ap.parse_args(argv)
    probabilities = pd.read_csv(mm.INPUT, parse_dates=["date"])
    if a.predictions:
        p = pd.read_csv(a.predictions, parse_dates=["date"])
    else:
        p = mm.evaluate(probabilities, pd.read_csv(ROOT / "data" / "nba_reconstructed.csv"),
                        pd.read_csv(ROOT / "diagnostics" / "ats_signal" / "open_spreads.csv"))[0]
    base = p[p["method"] == "fixed_13_5"].copy()
    coefficients = fitted_means(probabilities, 2026)
    scores, predictions = evaluate(base, coefficients)
    text = report(scores, coefficients, os.environ.get("GITHUB_SHA", "local"))
    a.out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(a.out / "ats_cover_rates.csv", index=False)
    coefficients.to_csv(a.out / "ats_mean_coefficients.csv", index=False)
    predictions.to_csv(a.out / "ats_predictions.csv.gz", index=False,
                       compression={"method": "gzip", "mtime": 0})
    (a.out / "ATS_REPORT.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
