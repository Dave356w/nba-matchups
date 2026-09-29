import numpy as np

import market


def test_american_parsing_rejects_non_prices():
    assert market.american("+120") == 120
    assert market.american({"american": "-150"}) == -150
    assert market.american(-110.0) == -110
    for bad in (None, "EVEN", 50, "-99", {}):
        assert market.american(bad) is None


def test_devig_removes_hold_and_breakeven_keeps_it():
    q = market.devig(-110, -110)
    assert abs(q - 0.5) < 1e-12
    be = market.implied(-110)
    assert be > 0.5                       # the posted price carries the vig
    assert np.isnan(market.devig(-110, 50))


def test_ev_null_is_minus_the_hold_not_zero():
    q = [0.5, 0.6]
    be = market.breakeven_prob([-110, -160])
    null = market.ev_null(q, be)
    assert null < 0
    assert np.isnan(market.ev_null(q, [0.5, np.nan]))
    assert np.isnan(market.ev_null([], []))


def test_excess_se_uses_prices_not_outcomes():
    # An all-win bucket must not report a zero error bar.
    assert market.excess_se([0.6, 0.6, 0.6]) > 0
    assert abs(market.excess_se([0.5] * 4) - 0.25) < 1e-12


def test_ladder_covers_every_price_and_rejects_invalid():
    assert market.ladder_rung(-300) == "≤ -250"
    assert market.ladder_rung(-100) == "-129 to -100"
    assert market.ladder_rung(100) == "+100 to +129"
    assert market.ladder_rung(400) == "≥ +250"
    assert market.ladder_rung(0) is None and market.ladder_rung(None) is None


def test_unit_profit():
    assert market.unit_profit(150, True) == 1.5
    assert abs(market.unit_profit(-200, True) - 0.5) < 1e-12
    assert market.unit_profit(-200, False) == -1.0
    assert np.isnan(market.unit_profit(20, True))


def test_parse_dk_odds_and_scoreboard_fixtures():
    js = {"items": [
        {"provider": {"id": "58"}, "homeTeamOdds": {"moneyLine": -999}},
        {"provider": {"id": "100"},
         "homeTeamOdds": {"moneyLine": -150,
                          "open": {"moneyLine": {"american": "-140"}},
                          "close": {"moneyLine": {"american": "-160"}},
                          "current": {"moneyLine": {"american": "-155"}}},
         "awayTeamOdds": {"moneyLine": 130,
                          "open": {"moneyLine": {"american": "+120"}},
                          "close": {"moneyLine": {"american": "+135"}}}}]}
    o = market.parse_dk_odds(js)
    assert o["cur_home_ml"] == -155 and o["cur_away_ml"] == 130
    assert o["open_home_ml"] == -140 and o["close_away_ml"] == 135
    assert market.parse_dk_odds({"items": []}) is None

    sb = {"events": [{"id": "401", "date": "2026-11-20T00:30Z",
                      "season": {"type": 2},
                      "competitions": [{"status": {"type": {"state": "post",
                                                            "completed": True}},
                                        "competitors": [
                          {"homeAway": "home", "score": "110",
                           "team": {"abbreviation": "GS"}},
                          {"homeAway": "away", "score": "104",
                           "team": {"abbreviation": "UTAH"}}]}]}]}
    g = market.parse_scoreboard(sb)[0]
    assert (g["home"], g["away"]) == ("GSW", "UTA")
    assert g["completed"] and g["home_pts"] == 110 and g["season_type"] == 2
