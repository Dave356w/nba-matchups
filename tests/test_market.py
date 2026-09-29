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


def _book(pid, name, home, away, close=True):
    side = lambda ml: {"moneyLine": ml, "open": {"moneyLine": ml},
                       **({"close": {"moneyLine": ml}} if close else {})}
    return {"provider": {"id": pid, "name": name},
            "homeTeamOdds": side(home), "awayTeamOdds": side(away)}


def test_parse_book_odds_reads_dk_and_espnbet_never_live():
    js = {"items": [_book("59", "ESPN Bet - Live Odds", -10000, 1800),
                    _book("58", "ESPN BET", -240, 200),
                    _book("100", "Draft Kings", -258, 210)]}
    b = market.parse_book_odds(js)
    assert set(b) == {"dk", "espnbet"}
    assert b["espnbet"]["close_home_ml"] == -240 and b["espnbet"]["book"] == "espnbet"
    assert b["dk"]["close_home_ml"] == -258
    live_only = market.parse_book_odds({"items": [_book("59", "live", -10000, 1800)]})
    assert live_only == {}
    assert market.pick_close(live_only) == {} and market.pick_pregame(live_only) == {}


def test_pick_close_prefers_row_book_then_preference_order():
    both = market.parse_book_odds({"items": [_book("58", "ESPN BET", -240, 200),
                                             _book("100", "DK", -258, 210)]})
    assert market.pick_close(both)["book"] == "dk"
    assert market.pick_close(both, prefer="espnbet")["book"] == "espnbet"
    eb_only = market.parse_book_odds({"items": [_book("58", "ESPN BET", -240, 200),
                                                _book("100", "DK", -258, 210, close=False)]})
    assert market.pick_close(eb_only, prefer="dk")["book"] == "espnbet"
    assert market.pick_pregame(eb_only)["book"] == "dk"   # DK current price exists


def test_spread_line_and_close_spread_parsing():
    assert market.spread_line({"american": "-5.5"}) == -5.5
    assert market.spread_line("+3") == 3.0 and market.spread_line(-1.5) == -1.5
    assert market.spread_line("PK") == 0.0
    for bad in (None, "", "abc", 99, {}):
        assert market.spread_line(bad) is None
    js = {"items": [{"provider": {"id": "100"},
                     "homeTeamOdds": {"close": {
                         "moneyLine": {"american": "-230"},
                         "pointSpread": {"american": "-5.5"},
                         "spread": {"american": "-108"}}},
                     "awayTeamOdds": {"close": {
                         "moneyLine": {"american": "+190"},
                         "pointSpread": {"american": "+5.5"},
                         "spread": {"american": "-112"}}}}]}
    o = market.pick_close(market.parse_book_odds(js))
    assert o["book"] == "dk" and o["close_spread"] == -5.5
    assert o["close_home_spread_odds"] == -108 and o["close_away_spread_odds"] == -112
    # home line derived from the away side; inconsistent lines are dropped
    assert market._home_spread(None, 4.0) == -4.0
    assert market._home_spread(-3.5, 4.5) is None
    # a book without spreads still parses its moneyline
    assert market.parse_dk_odds({"items": [_book("100", "DK", -150, 130)]}
                                )["close_spread"] is None


def test_ats_result():
    assert market.ats_result(7, -5.5) == 1.0       # home -5.5 wins by 7
    assert market.ats_result(5, -5.5) == 0.0
    assert market.ats_result(-3, 3.0) == 0.5       # push
    assert market.ats_result(-2, 3.0) == 1.0       # home +3 loses by 2
    assert np.isnan(market.ats_result(None, -3))


def test_parse_injuries_records_listed_and_none():
    js = {"injuries": [{"team": {"abbreviation": "GS"}, "injuries": [
        {"athlete": {"id": 7, "displayName": "Star"}, "status": "Out",
         "details": {"type": "Knee", "detail": "Soreness"}},
        {"athlete": {"id": 8, "displayName": "Wing"},
         "status": {"name": "Questionable"}, "shortComment": "ankle"}]}]}
    rows = market.parse_injuries(js, "GSW", "UTA")
    assert [(r["team"], r["name"], r["status"]) for r in rows] == [
        ("GSW", "Star", "Out"), ("GSW", "Wing", "Questionable"), ("UTA", "", "NONE")]
    assert rows[0]["player_id"] == "7" and rows[0]["detail"] == "Knee Soreness"
    assert rows[1]["detail"] == "ankle"
    assert market.parse_injuries({"boxscore": {}}, "GSW", "UTA") is None
    assert [r["status"] for r in market.parse_injuries({"injuries": []}, "GSW", "UTA")] \
        == ["NONE", "NONE"]
