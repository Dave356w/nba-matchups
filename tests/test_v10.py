"""Model v10 (CLAUDE.md amendment 2026-10-28): ORB% in the composite,
drafted rookies valued by pick, the report route refitted before each
slate, its tags, and test membership; v9 files score exactly as before."""
import numpy as np
import pandas as pd

import analysis
import backfill_history as bf
import build_site
import nba_composite as nc
import player_availability as pav
import public_site
from test_analysis_and_pages import _ats_rows
from test_v8 import teams
from test_v9 import MAVAIL, MBASE

VBASE = {**MBASE, "version": nc.V10, "orb": nc.ORB_PCT}
VAVAIL = {**MAVAIL, "version": nc.V10, "orb": nc.ORB_PCT}
T = np.array([40, 88, 12, 17, 22, 11, 30, 14, 39, 86, 13, 16, 21, 10, 31, 15], float)


def test_orb_pct_replaces_only_the_rebounding_factors():
    f, g = nc.features_from_totals(T), nc.features_from_totals(T, orb_pct=True)
    assert np.allclose(np.delete(f, [2, 6]), np.delete(g, [2, 6]))
    assert np.isclose(g[2], 11 / (11 + 31) * 100) and np.isclose(g[6], -10 / (10 + 30) * 100)
    o = pd.DataFrame({"FG": [40.0], "3P": [12.0], "FGA": [88.0], "TOV": [14.0],
                      "ORB": [11.0], "DRB": [30.0], "FTA": [22.0]})
    p = pd.DataFrame({"FG": [39.0], "3P": [13.0], "FGA": [86.0], "TOV": [15.0],
                      "ORB": [10.0], "DRB": [31.0], "FTA": [21.0]})
    a, b = nc.rates_from_per100(o, p), nc.rates_from_per100(o, p, orb_pct=True)
    assert np.allclose(np.delete(a, [2, 6], 1), np.delete(b, [2, 6], 1))
    assert np.isclose(b[0, 2], 11 / 42 * 100) and np.isclose(b[0, 6], -10 / 40 * 100)


def test_weights_say_how_the_composite_reads_rebounds(weights):
    assert not nc.is_orb_pct(weights) and nc.is_orb_pct({**weights, "orb": "pct"})
    lh, la = teams(30)[0]["A"], teams(30)[0]["B"]
    v9 = nc.fade_inputs(lh, 20, None, la, 20, None, 0.36, weights)
    v10 = nc.fade_inputs(lh, 20, None, la, 20, None, 0.36, {**weights, "orb": "pct"})
    assert v9["ft_diff"] == v10["ft_diff"] and v9["delta"] != v10["delta"]


def test_rookies_are_valued_by_pick_only_without_a_bpm_row():
    box = pd.DataFrame({"player_id": ["1", "2", "3"], "name": ["Vet One", "Top Pick", "Undrafted"],
                        "minutes": [30.0, 25.0, 10.0]})
    bpm = {pav.norm_name("Vet One"): (1.0, 2000.0, 70.0)}
    picks = {pav.norm_name("Top Pick"): 2, pav.norm_name("Vet One"): 5}
    curve = (-1.0, -0.5, 100)
    v9, _, _ = pav.player_values(box, bpm)
    v10, _, _ = pav.player_values(box, bpm, (picks, curve))
    assert v9["2"] == 0.0 and v9["3"] == 0.0
    assert np.isclose(v10["2"], -1.0 - 0.5 * np.log(2) - pav.REPLACEMENT_BPM)
    assert v10["1"] == v9["1"] and v10["3"] == 0.0      # a veteran keeps his BPM
    assert pav.rookies_for(2027, {"season": 2027, "picks": picks, "curve": list(curve)}) == \
        (picks, curve)


def test_refit_adds_this_seasons_games_and_keeps_the_model_otherwise():
    rng = np.random.default_rng(1)
    feats = ["delta", "b2b_net"]
    model = {"features": feats, "link": "margin", "intercept": 2.0, "coef": [0.1, 2.0],
             "sigma": 13.0, "version": nc.V10}
    X = rng.normal(0, 10, (3000, 2))
    train = pd.DataFrame(X, columns=feats).assign(margin=1.0 + X @ [0.1, 2.0]
                                                  + rng.normal(0, 13, len(X)))
    same = pav.refit(model, pd.DataFrame(columns=feats + ["margin"]), train)
    assert same["n_refit"] == 0 and same["coef"] == model["coef"]
    extra = pd.DataFrame(rng.normal(0, 10, (400, 2)), columns=feats)
    extra["margin"] = 9.0 + extra.to_numpy() @ [0.1, 2.0]
    extra.loc[0, "delta"] = np.nan                      # incomplete rows are skipped
    m = pav.refit(model, extra, train)
    assert m["n_refit"] == 399 and m["intercept"] > 1.5 and m["features"] == feats
    assert nc.is_margin(m) and m["version"] == nc.V10 and model["intercept"] == 2.0


def test_slate_refits_use_only_games_before_each_slate():
    feats = ["delta"]
    model = {"features": feats, "link": "margin", "intercept": 0.0, "coef": [1.0],
             "sigma": 10.0}
    train = pd.DataFrame({"delta": np.linspace(-5, 5, 200)}).assign(margin=lambda d: d.delta)
    te = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"] * 2 + ["2026-01-02"] * 2),
                       "delta": [0.0, 0.0, 0.0, 0.0], "margin": [50.0, 50.0, 0.0, 0.0]})
    p, m = bf.slate_refits(model, train, te, te[feats])
    assert np.allclose(m[:2], 0.0, atol=1e-9)           # day 1 sees no day-1 result
    assert m[2] > 0.3                                   # day 2 learnt from day 1


def test_v10_tags_routes_and_neutral_fill(weights):
    assert build_site.is_v10(VBASE) and build_site.is_v9(VBASE) and not build_site.is_v10(MBASE)
    tags = build_site.active_tags(VBASE, VAVAIL)
    assert tags == [build_site.MODEL_TAG_V10, build_site.MODEL_TAG_V10_AVAIL,
                    build_site.MODEL_TAG_V10_NEUTRAL, build_site.MODEL_TAG_V10_NEUTRAL_AVAIL]
    assert all(t.endswith("_v10") for t in tags) and len(set(tags)) == 4
    assert build_site.model_version(tags) == "v10"
    assert build_site.active_tags(MBASE, MAVAIL)[0] == build_site.MODEL_TAG_V9
    for tag, route in ((build_site.MODEL_TAG_V10, "base"), (build_site.MODEL_TAG_V10_AVAIL, "avail"),
                       (build_site.MODEL_TAG_V10_NEUTRAL, "neutral"),
                       (build_site.MODEL_TAG_V10_NEUTRAL_AVAIL, "neutral")):
        assert build_site.route({"p_home": 0.6, "model_tag": tag}, VBASE) == route
    cur, prior = teams(30)
    tal = lambda tm, d: {"A": 3.0, "B": -1.0}[tm]  # noqa: E731
    w10 = {**weights, "orb": "pct"}
    r = build_site.score_game(cur, "A", "B", "2027-02-01", w10, VBASE, prior_logs=prior, talent=tal)
    assert r["model_tag"] == build_site.MODEL_TAG_V10
    assert abs(r["p_home"] - nc.norm_cdf(r["margin_home"] / VBASE["sigma"])) < 3e-4
    av = build_site.score_game(cur, "A", "B", "2027-02-01", w10, VBASE, prior_logs=prior, talent=tal,
                               avail_model=VAVAIL, avail={"av_min": -0.4, "av_bpm": -1.0})
    assert av["model_tag"] == build_site.MODEL_TAG_V10_AVAIL
    nf = build_site.score_game(cur, "A", "B", "2027-02-01", w10, VBASE, prior_logs=prior, talent=None)
    assert nf["model_tag"] == build_site.MODEL_TAG_V10_NEUTRAL
    # v9 files score exactly as before
    v9 = build_site.score_game(cur, "A", "B", "2027-02-01", weights, MBASE, prior_logs=prior, talent=tal)
    assert v9["model_tag"] == build_site.MODEL_TAG_V9


def test_early_rows_accept_v9_and_v10_first_tags_only():
    tags = [build_site.MODEL_TAG_V8_AVAIL, build_site.MODEL_TAG_V9, build_site.MODEL_TAG_V10,
            build_site.MODEL_TAG_V10_AVAIL, build_site.MODEL_TAG_V10_NEUTRAL, None]
    g = _ats_rows(6).assign(gp_home=3, gp_away=3, first_p_home=0.8, p_home=0.8,
                            first_model_tag=tags, model_tag=build_site.MODEL_TAG_V10,
                            first_home_ml=-150, first_away_ml=130, first_q_home=0.6)
    accepted = set(g.game_id.iloc[1:5])
    for hyp in analysis.active_hypotheses():
        if hyp["early"]:
            assert set(analysis.hypothesis_picks(g, hyp, "first").game_id) == accepted
        else:
            late = g.assign(gp_home=30, gp_away=30)
            assert set(analysis.hypothesis_picks(late, hyp, "first").game_id) == set(g.game_id)
    # thresholds and rules are unchanged by the amendment
    active = {h["key"]: h for h in analysis.active_hypotheses()}
    assert active["H5"]["sigma"] == 13.5 and active["H4"]["min_edge"] == 0.12
    assert analysis.EARLY_TAG_SUFFIXES == ("_v9", "_v10")


def test_how_it_works_dates_the_update_only_when_v10_is_live(monkeypatch):
    native = pd.DataFrame({"basis": ["native"] * 2,
                           "first_model_tag": [build_site.MODEL_TAG_V9, build_site.MODEL_TAG_V10],
                           "first_snapshot_utc": ["2026-10-27T18:00Z", "2026-10-29T18:00Z"]})
    site = public_site.PublicSite.__new__(public_site.PublicSite)
    site.ui, site.native, site.today = build_site, native, "2026-10-30"
    monkeypatch.setattr(build_site, "ACTIVE_TAGS", [build_site.MODEL_TAG_V9])
    assert site.model_note() == ""
    monkeypatch.setattr(build_site, "ACTIVE_TAGS", build_site.active_tags(VBASE, VAVAIL))
    note = site.model_note()
    assert "Model updated October 29, 2026" in note and "previous version" in note
    assert "hindsight" not in note.lower() and "backtest" not in note.lower()
