"""Model v9: v8's inputs, routes and seasons fitted by least squares on the
final home margin, P = Phi(m / sigma), with the margin recorded beside P."""
import os

import numpy as np
import pandas as pd

import analysis
import backfill_history as bf
import build_site
import ledger
import nba_composite as nc
import validate_data_files as vdf
from test_v8 import AVAIL, BASE, CARRY, V4, score, teams

# the v8 logit coefficients x ~ 30 points per logit unit, as a margin model
MBASE = {**BASE, "link": "margin", "intercept": 2.1,
         "coef": [0.12, 2.3, 0.13, 0.08, 0.42, 0.09], "sigma": 13.2}
MAVAIL = {**AVAIL, "link": "margin", "intercept": 2.2,
          "coef": [0.14, 2.5, 0.7, 0.35, 0.15, 0.1, 0.31, 0.11], "sigma": 13.4}


def test_fit_margin_recovers_the_margin_sigma_and_the_win_probability():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 10, (40000, 2))
    m = 2.5 + X @ [0.5, -0.2] + rng.normal(0, 12, len(X))
    fit = nc.fit_margin(X, m, ["delta", "b2b_net"])
    assert nc.is_margin(fit) and fit["features"] == ["delta", "b2b_net"]
    assert abs(fit["intercept"] - 2.5) < 0.2 and np.allclose(fit["coef"], [0.5, -0.2], atol=0.01)
    assert abs(fit["sigma"] - 12) < 0.15
    x = np.array([[10.0, 1.0]])
    mu = 2.5 + 5 - 0.2
    assert abs(nc.predict_margin(fit, x)[0] - mu) < 0.3
    assert abs(nc.predict(fit, x)[0] - nc.norm_cdf(mu / 12)) < 0.01
    # a logit is unchanged and has no margin
    assert np.isnan(nc.predict_margin(BASE, [[0.0] * 6])[0])
    assert abs(nc.predict(BASE, [[0.0] * 6])[0] - 1 / (1 + np.exp(-0.3))) < 1e-12
    assert not nc.is_margin(BASE) and not nc.is_margin(None)


def test_v9_rows_record_the_margin_their_p_comes_from(weights):
    cur, prior = teams(30)
    r = build_site.score_game(cur, "A", "B", "2027-02-01", weights, MBASE, prior_logs=prior,
                              talent=lambda tm, d: {"A": 3.0, "B": -1.0}[tm],
                              carry=CARRY, fallback=V4)
    assert r["model_tag"] == build_site.MODEL_TAG_V9
    assert np.isfinite(r["margin_home"])
    # the margin is stored to 0.01 point
    assert abs(r["p_home"] - nc.norm_cdf(r["margin_home"] / MBASE["sigma"])) < 3e-4
    assert (r["margin_home"] > 0) == (r["p_home"] > 0.5)
    av = build_site.score_game(cur, "A", "B", "2027-02-01", weights, MBASE, prior_logs=prior,
                               talent=lambda tm, d: {"A": 3.0, "B": -1.0}[tm],
                               avail_model=MAVAIL, avail={"av_min": -0.4, "av_bpm": -1.0},
                               carry=CARRY, fallback=V4)
    assert av["model_tag"] == build_site.MODEL_TAG_V9_AVAIL and np.isfinite(av["margin_home"])
    # the logit fallbacks give no margin and keep their tags
    fb = build_site.score_game(cur, "A", "B", "2027-02-01", weights, MBASE, prior_logs=prior,
                               talent=None, carry=CARRY, fallback=V4)
    assert fb["model_tag"] == build_site.MODEL_TAG_V4 and np.isnan(fb["margin_home"])
    # v8 logits give no margin either
    assert np.isnan(score(cur, prior, "2027-02-01", weights)["margin_home"])


def test_v9_tags_routes_and_version():
    assert build_site.is_v9(MBASE) and build_site.is_v8(MBASE) and not build_site.is_v9(BASE)
    tags = build_site.active_tags(MBASE, MAVAIL, V4)
    assert tags[:2] == [build_site.MODEL_TAG_V9, build_site.MODEL_TAG_V9_AVAIL]
    assert build_site.model_version(tags) == "v9"
    assert build_site.MODEL_TAG_V9_AVAIL in build_site.AVAIL_TAGS
    assert build_site.route({"p_home": 0.6, "model_tag": build_site.MODEL_TAG_V9}, MBASE) == "base"
    assert build_site.route({"p_home": 0.6, "model_tag": build_site.MODEL_TAG_V9_AVAIL},
                            MBASE) == "avail"
    # every tag stays unique
    assert len({build_site.MODEL_TAG_V8, build_site.MODEL_TAG_V8_AVAIL,
                build_site.MODEL_TAG_V9, build_site.MODEL_TAG_V9_AVAIL}) == 4


def _pregame(gid="1", tip="2027-01-10T01:00Z", p=0.62, m=4.1):
    return {"game_id": gid, "slate_date": "2027-01-09", "season": 2027, "tip_utc": tip,
            "home": "AAA", "away": "BBB", "model_tag": build_site.MODEL_TAG_V9,
            "gp_home": 30, "gp_away": 30, "delta": 5.0, "p_home": p, "margin_home": m,
            "lean": "AAA", "p_lean": p, "pre_book": "dk", "pre_home_ml": -150,
            "pre_away_ml": 130, "pre_q_home": 0.58}


def test_margin_is_refreshed_until_tip_and_frozen_after():
    from datetime import datetime, timezone
    before = datetime(2027, 1, 9, 20, tzinfo=timezone.utc)
    led, acc, _ = ledger.upsert_pregame(ledger.empty(), [_pregame()], now=before)
    assert acc == ["1"] and led["margin_home"].iloc[0] == 4.1
    later = datetime(2027, 1, 9, 23, tzinfo=timezone.utc)
    led, _, _ = ledger.upsert_pregame(led, [_pregame(p=0.64, m=4.8)], now=later)
    assert led["margin_home"].iloc[0] == 4.8
    after = datetime(2027, 1, 10, 2, tzinfo=timezone.utc)
    led, acc, rej = ledger.upsert_pregame(led, [_pregame(p=0.7, m=7.0)], now=after)
    assert not acc and led["margin_home"].iloc[0] == 4.8
    assert "margin_home" in ledger.PREGAME_COLUMNS and "margin_home" not in ledger.GRADE_COLUMNS


def test_validator_checks_the_margin_and_reads_the_pre_v9_schema(tmp_path):
    path = tmp_path / "l.csv"
    good = pd.DataFrame([_pregame()], columns=ledger.COLUMNS)
    good.to_csv(path, index=False)
    assert vdf.check(str(path)) == []
    old = good[ledger.PRE_MARGIN_COLUMNS]
    old.to_csv(path, index=False)
    assert vdf.check(str(path)) == []
    for bad in ({"margin_home": -4.1}, {"p_home": np.nan, "lean": np.nan, "p_lean": np.nan},
                {"margin_home": 75.0}):
        b = good.copy()
        for k, v in bad.items():
            b[k] = v
        b.to_csv(path, index=False)
        assert vdf.check(str(path)), bad


def test_card_spread_lean_uses_the_models_own_margin():
    row = pd.Series({**_pregame(p=0.62, m=4.1), "pre_spread": -2.5,
                     "pre_home_spread_odds": -110, "pre_away_spread_odds": -110,
                     "first_p_home": 0.62, "first_spread": -2.5,
                     "first_home_spread_odds": -110, "first_away_spread_odds": -110})
    lean = analysis.spread_lean(row)
    assert lean["own"] and abs(lean["model_line"] + 4.1) < 1e-9 and lean["side"] == "AAA"
    assert abs(lean["edge"] - 1.6) < 1e-9
    # the row's own sigma: P = Phi(m / sigma) -> the cover probability
    sigma = 4.1 / nc_ppf(0.62)
    assert abs(lean["model_p"] - nc.norm_cdf(1.6 / sigma)) < 1e-6
    # without a recorded margin: the 13.5 conversion, as before
    plain = analysis.spread_lean(row.drop("margin_home"))
    assert not plain["own"] and abs(plain["model_line"] + 13.5 * nc_ppf(0.62)) < 1e-9
    # H5 / H6 never read it: the registered conversion at the first snapshot
    s = analysis.ats_sides(pd.DataFrame([row]), "first")
    assert not s["own"].iloc[0] and abs(s["model_line"].iloc[0] + 13.5 * nc_ppf(0.62)) < 1e-9


def nc_ppf(p):
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def test_rescore_writes_the_margin_and_keeps_prices():
    recon = pd.DataFrame([{**_pregame(), "basis": "reconstructed", "close_q_home": 0.6,
                           "home_won": 1, "margin_home": np.nan}], columns=ledger.COLUMNS)
    test = pd.DataFrame([{"date": "2027-01-09", "home": "AAA", "away": "BBB", "gp_home": 30,
                          "gp_away": 30, "delta": 6.0, "p_home": 0.66, "margin_home": 5.4321,
                          "model_tag": build_site.MODEL_TAG_V9, "carry_p_home": np.nan}])
    out, n = bf.rescore(recon, test)
    assert n == 1 and out["margin_home"].iloc[0] == 5.43 and out["close_q_home"].iloc[0] == 0.6
    assert out["model_tag"].iloc[0] == build_site.MODEL_TAG_V9


def test_rebuild_refuses_completed_seasons_and_rescore():
    import pytest
    with pytest.raises(SystemExit):
        bf.main(["--seasons", "2026", "--rebuild", "--no-avail"])
    with pytest.raises(SystemExit):
        bf.main(["--seasons", "2027", "--rebuild", "--rescore", "--no-avail"])


def _graded(gid, p, m, won, basis, close_spread=-3.5, **kw):
    return {**_pregame(gid=gid, p=p, m=m), "basis": basis, "close_book": "dk",
            "close_home_ml": -150, "close_away_ml": 130, "close_q_home": 0.58,
            "close_spread": close_spread, "close_home_spread_odds": -110,
            "close_away_spread_odds": -110, "home_pts": 110 if won else 100,
            "away_pts": 101, "home_won": won, "gp_home": 12, "gp_away": 12, **kw}


def test_results_show_the_rebuilt_season_apart_from_the_live_record():
    native = pd.DataFrame([_graded("1", 0.62, np.nan, 1, "native"),
                           _graded("2", np.nan, np.nan, 0, "native", p_home=np.nan,
                                   lean=np.nan, p_lean=np.nan, model_tag=np.nan)],
                          columns=ledger.COLUMNS)
    rebuilt = pd.DataFrame([_graded(g, p, m, w, "rebuilt", pre_book=np.nan,
                                    pre_home_ml=np.nan, pre_away_ml=np.nan, pre_q_home=np.nan)
                            for g, p, m, w in (("1", 0.64, 4.6, 1), ("2", 0.40, -3.3, 0))],
                           columns=ledger.COLUMNS)
    html = build_site.render_results(native, "t", "2027-01-10", rebuilt)
    live, rb = html.split("id='rebuilt'")
    assert "1–0" in live and "1 bet ·" in live            # the live record: game 1 only
    assert "1 graded game this season had no live pick" in live
    assert "hindsight" in rb and "Lean · moneyline" in rb and "Spread miss" in rb
    assert "<td class='small'>no</td>" in rb and "<td class='small'>yes</td>" in rb
    assert "AAA -4.6" in rb                                # the model's own margin
    empty = build_site.render_results(native, "t", "2027-01-10", ledger.empty())
    assert "Appears once this season" in empty
    perf = build_site.render_performance(native, "t", "2027-01-10", rebuilt)
    assert "This season, rebuilt after the fact" in perf and "results.html#rebuilt" in perf


def test_card_names_fit_and_show_the_model_spread():
    row = pd.DataFrame([{**_pregame(p=0.62, m=4.1), "pre_spread": -2.5,
                         "pre_home_spread_odds": -110, "pre_away_spread_odds": -110,
                         "home_b2b": 0, "away_b2b": 0, "basis": "native"}],
                       columns=ledger.COLUMNS).iloc[0]
    card = build_site.game_card(row)
    assert "<span class='nm'><span class='tn'>" in card and "class='lean-pill'" in card
    assert "Model spread</td>" in card and "-4.1" in card     # its own margin
    assert "the model's predicted margin" in card              # the spread lean says so
    v8_row = row.copy()
    v8_row["margin_home"] = np.nan
    plain = build_site.game_card(v8_row)
    assert "Model spread (from P)" in plain and "13.5 × Φ⁻¹(P)" in plain


def test_validator_checks_rebuilt_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    row = {**_pregame(), "basis": "rebuilt", "pre_book": np.nan, "pre_home_ml": np.nan,
           "pre_away_ml": np.nan, "pre_q_home": np.nan, "close_book": "dk",
           "close_home_ml": -150, "close_away_ml": 130, "close_q_home": 0.58,
           "home_pts": 110, "away_pts": 101, "home_won": 1}
    pd.DataFrame([row], columns=ledger.COLUMNS).to_csv(ledger.REBUILT_PATH, index=False)
    assert vdf.main() == 0
    pd.DataFrame([{**row, "home_won": np.nan, "home_pts": np.nan, "away_pts": np.nan,
                   "close_q_home": np.nan, "close_home_ml": np.nan, "close_away_ml": np.nan,
                   "close_book": np.nan}],
                 columns=ledger.COLUMNS).to_csv(ledger.REBUILT_PATH, index=False)
    assert vdf.main() == 1
