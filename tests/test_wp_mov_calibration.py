"""Chronology, final-score identity, and recovery of a known margin mapping."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "research"))
import wp_mov_calibration as wm  # noqa: E402


def sample():
    rows = []
    # Deliberately nonlinear-in-WP, exactly linear-in-probit margins.
    for year in (2022, 2023, 2024):
        for i, p in enumerate((.1, .2, .3, .7, .8, .9)):
            z = wm.probit([p])[0]
            margin = 1 + 11 * z
            rows.append(dict(year=year, date=pd.Timestamp(f"{year}-01-{i + 1:02d}"),
                             home="BOS", away="NYK", p_home=p, margin=margin,
                             win=int(margin > 0), phase=.5, route="base",
                             probability_train_max_year=year - 1, weights_train_max_year=year - 1))
    return pd.DataFrame(rows)


def test_nested_chronology_and_known_mapping():
    d = sample()
    pred, coef = wm.walk_forward(d)
    assert set(pred["year"]) == {2023, 2024}  # warm-up never scored
    assert (coef["train_max_year"] < coef["year"]).all()
    fit = coef[coef["method"] == "affine"]
    assert np.allclose(fit["intercept"], 1)
    assert np.allclose(fit["scale"], 11)
    g = pred[pred["method"] == "affine"]
    assert wm.metrics(g)["rmse"] < 1e-12
    # Perturb the held-out outcomes: its fitted coefficients cannot change.
    d.loc[d["year"] == 2024, "margin"] *= 2
    _, perturbed = wm.walk_forward(d)
    pd.testing.assert_frame_equal(coef, perturbed)
    fixed = pred[pred["method"] == "fixed_13_5"]
    assert np.allclose(fixed["pred_margin"], 13.5 * wm.probit(fixed["p_home"]))


def test_refuses_leaking_probabilities_and_duplicate_games():
    d = sample()
    d.loc[0, "probability_train_max_year"] = d.loc[0, "year"]
    with pytest.raises(ValueError, match="leaks"):
        wm.walk_forward(d)
    d = sample()
    with pytest.raises(ValueError, match="duplicate"):
        wm.walk_forward(pd.concat([d, d.iloc[:1]]))


def test_score_join_is_exact_and_checks_sign():
    d = sample()
    scores = d[wm.KEYS + ["margin"]]
    joined = wm.attach_margins(d.drop(columns="margin"), scores.sample(frac=1, random_state=2))
    assert np.allclose(joined["margin"], d["margin"])
    bad = scores.copy()
    bad.loc[0, "margin"] *= -1
    with pytest.raises(ValueError, match="disagrees"):
        wm.attach_margins(d.drop(columns="margin"), bad)
    with pytest.raises(ValueError, match="Missing"):
        wm.attach_margins(d.drop(columns="margin"), scores.iloc[1:])


def test_bootstrap_is_paired_reproducible_and_identical_baseline_zero():
    pred, _ = wm.walk_forward(sample())
    fixed = pred[pred["method"] == "fixed_13_5"]
    result = wm.paired_bootstrap(fixed, fixed, draws=100)
    assert result["delta_mae"] == result["delta_rmse"] == 0
    assert result["mae_ci_low"] == result["rmse_ci_high"] == 0
    affine = pred[pred["method"] == "affine"]
    a = wm.paired_bootstrap(affine, fixed, draws=100)
    assert a == wm.paired_bootstrap(affine, fixed, draws=100)
    assert a["mae_ci_high"] < 0 and a["rmse_ci_high"] < 0
    with pytest.raises(ValueError, match="match exactly"):
        wm.paired_bootstrap(affine.iloc[1:], fixed, draws=100)


def test_report_and_csv_roundtrip(tmp_path):
    d = sample()
    path = tmp_path / "input.csv"
    d.to_csv(path, index=False)
    out = tmp_path / "report"
    wm.main(["--input", str(path), "--out", str(out)])
    text = (out / "REPORT.md").read_text()
    assert "Matched evaluation: 12 games" in text
    assert "not an out-of-sample score" in text
    assert "By model route" in text and "By season phase" in text
    scored = pd.read_csv(out / "predictions.csv")
    assert (scored["mapping_train_max_year"] < scored["year"]).all()
