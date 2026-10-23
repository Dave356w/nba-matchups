"""Membership, price and retirement boundaries of the registered amendment."""
import numpy as np

import analysis
import build_site
import ledger
from test_analysis_and_pages import _ats_rows


def test_three_tests_have_disjoint_early_segments_with_identical_thresholds():
    active = {h["key"]: h for h in analysis.active_hypotheses()}
    assert set(active) == {"H5", "H2·F", "H4", "H5·E", "H2·F·E", "H4·E"}
    g = _ats_rows(5)
    g["gp_home"], g["gp_away"] = [0, 1, 9, 10, 30], 30
    g["first_p_home"], g["p_home"] = 0.8, 0.1
    g["first_home_ml"], g["first_away_ml"], g["first_q_home"] = -150, 130, 0.6
    g["carry_p_home"], g["first_carry_p_home"] = 0.01, 0.01
    for key in ("H5", "H2·F", "H4"):
        late, early = active[key], active[key + "·E"]
        for field in ("native", "hindsight", "min_edge", "favourite", "sigma"):
            assert late.get(field) == early.get(field)
        assert early["native"] == "first" and not early["carry"]
        de = analysis.hypothesis_picks(g, early, "first")
        dl = analysis.hypothesis_picks(g, late, "first")
        assert set(de.game_id) == set(g.game_id.iloc[1:3])
        assert set(dl.game_id) == set(g.game_id.iloc[3:])
        assert not set(de.game_id) & set(dl.game_id)
    assert active["H5"]["sigma"] == 13.5
    assert active["H5"]["min_edge"] == 1
    assert active["H4"]["min_edge"] == 0.12
    assert analysis.ATS_MAIN_LINE == (-135, 115)


def test_retirement_is_frozen_and_cannot_enter_active_grades():
    g = _ats_rows(6).assign(gp_home=3, gp_away=3, first_p_home=0.8)
    rows = analysis.hypothesis_rows(g, g)
    retired = [(h, r, n) for h, r, n in rows if h.get("retired")]
    assert {h["key"] for h, _, _ in retired} == {"H1", "H2", "H3", "H6"}
    assert all(n is None for _, _, n in retired)
    original = analysis.retired_registration()
    assert original["H1"]["native_graded_n"] == 0
    assert len(original["H1"]["native_bets"]) == 4
    assert len(original["H6"]["native_bets"]) == 11
    assert all(r["home_won"] is None for v in original.values() for r in v["native_bets"])
    assert all(not h.get("retired") for h, _ in build_site.public_ats(g))
    assert all(not h.get("retired") for h, _ in analysis.hypothesis_by_season(g, g))
    assert analysis.ats_card(g.iloc[0])["hyp"]["key"] == "H5·E"
    # Refreshes, results and a contrary carryover P cannot rewrite the archive.
    g["first_carry_p_home"] = 0.01
    analysis.hypothesis_rows(g.assign(home_won=0), g)
    assert analysis.retired_registration() == original


def test_early_segments_require_the_model_at_the_graded_snapshot():
    tags = [build_site.MODEL_TAG_V8_AVAIL, build_site.MODEL_TAG_V9,
            build_site.MODEL_TAG_V9_AVAIL, build_site.MODEL_TAG_V9_NEUTRAL,
            build_site.MODEL_TAG_V9_NEUTRAL_AVAIL, None]
    g = _ats_rows(6).assign(gp_home=3, gp_away=3, first_p_home=0.8,
                            p_home=0.8, first_model_tag=tags,
                            model_tag=build_site.MODEL_TAG_V9,
                            first_home_ml=-150, first_away_ml=130,
                            first_q_home=0.6, pre_home_ml=-150,
                            pre_away_ml=130, pre_q_home=0.6, pre_spread=-5.5,
                            pre_home_spread_odds=-110, pre_away_spread_odds=-110)
    accepted = set(g.game_id.iloc[1:5])
    for hyp in analysis.active_hypotheses():
        if hyp["early"]:
            assert set(analysis.hypothesis_picks(g, hyp, "first").game_id) == accepted
            # Reconstructed hindsight uses its own model tag, not a native lock.
            assert set(analysis.hypothesis_picks(g, hyp, "pre").game_id) == set(g.game_id)
        else:
            # Existing games-10+ membership stays exactly as registered.
            late = g.assign(gp_home=30, gp_away=30)
            assert set(analysis.hypothesis_picks(late, hyp, "first").game_id) == set(g.game_id)
    assert analysis.ats_card(g.iloc[0]) is None
    assert analysis.ats_card(g.iloc[5]) is None
    assert all(analysis.ats_card(g.iloc[i])["hyp"]["key"] == "H5·E"
               for i in range(1, 5))
    card = build_site.game_card(g.iloc[0])
    assert "Spread pick · H5·E" not in card
