"""Sanity invariants of the deployed model's logic (owner request, 2026-11-12).

Direction and symmetry checks on the committed fits and the shared input
functions: none depends on a result, and each would fail on a sign slip, a
home/away mix-up or a later game leaking into a rating.
"""
import math

import numpy as np
import pandas as pd
import pytest

import nba_composite as nc


@pytest.fixture(scope="module")
def weights():
    return nc.load_json("weights.json")


def _log(n, shift, start, seed):
    """A team's game log: shoots `shift` better than, and holds opponents
    `shift` below, a league-average eFG."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"date": pd.date_range(start, periods=n, freq="2D")})
    for side, sh in (("T", shift), ("O", -shift)):
        d[side + "FGA"] = rng.normal(88, 4, n)
        d[side + "FG"] = d[side + "FGA"] * (0.46 + sh)
        d[side + "3PA"] = rng.normal(36, 3, n)
        d[side + "3P"] = d[side + "3PA"] * 0.36
        d[side + "FTA"] = rng.normal(22, 3, n)
        d[side + "FT"] = d[side + "FTA"] * (0.78 + sh)
        d[side + "ORB"] = rng.normal(10, 1, n)
        d[side + "DRB"] = rng.normal(33, 2, n)
        d[side + "TOV"] = rng.normal(13, 1, n)
    return d


GOOD, BAD = _log(30, 0.02, "2026-10-20", 1), _log(30, -0.02, "2026-10-20", 2)
GOOD_PRIOR, BAD_PRIOR = _log(82, 0.02, "2025-10-21", 3), _log(82, -0.02, "2025-10-21", 4)


def test_swapping_home_and_away_negates_every_rating_input(weights):
    a = nc.fade_inputs(GOOD, 12, GOOD_PRIOR, BAD, 12, BAD_PRIOR, 0.36, weights)
    b = nc.fade_inputs(BAD, 12, BAD_PRIOR, GOOD, 12, GOOD_PRIOR, 0.36, weights)
    assert a["delta"] > 0 and a["ft_diff"] > 0          # the better team at home
    for k in ("delta", "luck_def", "ft_diff"):
        assert a[k] == pytest.approx(-b[k], abs=1e-9)


def test_a_rating_reads_only_games_before_the_one_scored(weights):
    later = GOOD.copy()
    later.loc[12:, [c for c in later if c != "date"]] *= 3   # games 12+ rewritten
    a = nc.fade_inputs(GOOD, 12, GOOD_PRIOR, BAD, 12, BAD_PRIOR, 0.36, weights)
    b = nc.fade_inputs(later, 12, GOOD_PRIOR, BAD, 12, BAD_PRIOR, 0.36, weights)
    assert a == b


def test_last_season_leads_early_and_is_gone_by_game_ten(weights):
    # this season says GOOD is better; last season said the opposite
    def delta(i, ph=BAD_PRIOR, pa=GOOD_PRIOR):
        return nc.fade_inputs(GOOD, i, ph, BAD, i, pa, 0.36, weights)["delta"]
    assert delta(0) < 0 and delta(1) < 0                 # last season's rating
    assert delta(9) > 0
    assert delta(10) == pytest.approx(delta(10, None, None), abs=1e-9)
    assert [nc.fade_rho(i) for i in (0, 5, 10, 30)] == [0.25, 0.125, 0.0, 0.0]


def test_opponents_hot_three_point_shooting_counts_as_bad_luck(weights):
    hot = GOOD.assign(O3P=GOOD.O3PA * 0.42, OFG=GOOD.OFG + GOOD.O3PA * 0.06)
    cold = GOOD.assign(O3P=GOOD.O3PA * 0.30, OFG=GOOD.OFG - GOOD.O3PA * 0.06)
    assert nc.opp_luck(hot, 20, 0.36, weights) > 0      # the rating should rise
    assert nc.opp_luck(cold, 20, 0.36, weights) < 0
    assert abs(nc.opp_luck(GOOD, 20, 0.36, weights)) < 1e-6


def test_back_to_back_and_phase_inputs_point_the_right_way():
    opening = pd.Timestamp("2026-10-20")
    assert nc.logit_inputs(0, 4, 0, "2026-12-01", opening)["b2b_net"] == 1    # away tired
    assert nc.logit_inputs(0, 0, 4, "2026-12-01", opening)["b2b_net"] == -1   # home tired
    assert nc.logit_inputs(0, 0, 0, "2026-12-01", opening)["b2b_net"] == 0
    assert nc.season_phase(opening, opening) == 0.0
    assert nc.season_phase("2027-06-01", opening) == 1.0
    assert nc.logit_inputs(10, 1, 1, "2027-01-15", opening)["d_phase"] == pytest.approx(
        10 * nc.season_phase("2027-01-15", opening))
    two = pd.DataFrame({"date": pd.to_datetime(["2026-11-01", "2026-11-02"])})
    assert nc.rest_days(two, 1) == 0 and nc.rest_days(two, 0) == 4


@pytest.mark.parametrize("name", ["logit.json", "logit_avail.json"])
def test_deployed_fits_have_sensible_signs_and_scale(name):
    m = nc.load_json(name)
    assert nc.is_margin(m) and m["version"] >= 11
    coef = dict(zip(m["features"], m["coef"]))
    # every input is "home minus away, higher is better for the home team"
    assert all(v > 0 for v in coef.values()), coef
    assert 1.0 < m["intercept"] < 3.5                    # home edge, points
    assert 1.0 < coef["b2b_net"] < 3.5                   # a back-to-back, points
    assert 11.0 < m["sigma"] < 15.0                      # NBA margin residual SD


def test_probability_is_the_normal_cdf_of_margin_over_sigma():
    m = nc.load_json("logit.json")
    zero = np.zeros((1, len(m["features"])))
    p0 = float(nc.predict(m, zero)[0])
    assert p0 == pytest.approx(0.5 * (1 + math.erf(m["intercept"] / m["sigma"] / math.sqrt(2))))
    assert 0.53 < p0 < 0.60                              # equal teams: the home edge only
    x = np.random.default_rng(0).normal(0, 1, (50, len(m["features"]))) * [20, 1, 10, 5, 3, 3]
    margin, p = nc.predict_margin(m, x), nc.predict(m, x)
    assert np.allclose(p, nc.norm_cdf(margin / m["sigma"]))
    assert ((margin > 0) == (p > 0.5)).all()
    for j in range(x.shape[1]):                          # monotone in every input
        up = x.copy()
        up[:, j] += 1
        assert (nc.predict(m, up) > p).all()
