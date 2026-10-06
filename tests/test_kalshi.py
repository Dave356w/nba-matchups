"""Kalshi pricing for preseason rows: parsing, fee, matching, close <= tip."""
import market
import kalshi


def test_cost_and_american():
    assert abs(market.kalshi_cost(0.5) - 0.5175) < 1e-12
    assert market.american_from_prob(0.5) == -100
    assert market.american_from_prob(0.25) == 300
    assert market.american_from_prob(0.8) == -400
    assert market.american_from_prob(1.0) is None
    for p in (0.07, 0.31, 0.5, 0.64, 0.93):          # round-trips to ~p
        assert abs(market.implied(market.american_from_prob(p)) - p) < 0.003


def test_parse_events_maps_codes_and_dates():
    js = {"events": [
        {"event_ticker": "KXNBAGAME-26OCT06BKNCHA",
         "markets": [{"ticker": "KXNBAGAME-26OCT06BKNCHA-BKN", "yes_ask_dollars": "0.3000"},
                     {"ticker": "KXNBAGAME-26OCT06BKNCHA-CHA", "yes_ask_dollars": "0.7400"}]},
        {"event_ticker": "KXNBAGAME-26OCT06XYZABC",          # not NBA teams
         "markets": [{"ticker": "KXNBAGAME-26OCT06XYZABC-XYZ"},
                     {"ticker": "KXNBAGAME-26OCT06XYZABC-ABC"}]}]}
    games = kalshi.parse_events(js)
    assert list(games) == [("26OCT06", frozenset({"BRK", "CHO"}))]
    o = kalshi.pregame_odds(games, "2026-10-06", "CHO", "BRK")
    assert o["book"] == "kalshi" and o["cur_home_ml"] < 0 < o["cur_away_ml"]
    assert kalshi.pregame_odds(games, "2026-10-07", "CHO", "BRK") == {}
    assert kalshi.market_ticker("2026-10-06", "BRK", "CHO", "BRK") == \
        "KXNBAGAME-26OCT06BKNCHA-BKN"


def test_market_ticker_uses_kalshi_codes():
    # live tickers (2026-10-04/05): GSW and NYK, not GS / NY
    assert kalshi.market_ticker("2026-10-04", "GSW", "LAC", "GSW") == \
        "KXNBAGAME-26OCT04GSWLAC-GSW"
    assert kalshi.market_ticker("2026-10-05", "PHI", "PHI", "NYK") == \
        "KXNBAGAME-26OCT05NYKPHI-PHI"
    assert kalshi.market_ticker("2026-10-05", "PHO", "DET", "PHO") == \
        "KXNBAGAME-26OCT05PHXDET-PHX"
    for bbr in market.BBR_TEAMS:                     # every code round-trips
        code = kalshi.market_ticker("2026-10-05", bbr, bbr, "ATL").rsplit("-", 1)[-1]
        assert kalshi.team(code) == bbr


def test_one_sided_book_is_unpriced():
    js = {"events": [{"event_ticker": "KXNBAGAME-26OCT03MIATOR", "markets": [
        {"ticker": "KXNBAGAME-26OCT03MIATOR-MIA", "yes_ask_dollars": "0.3600"},
        {"ticker": "KXNBAGAME-26OCT03MIATOR-TOR", "yes_ask_dollars": None}]}]}
    assert kalshi.pregame_odds(kalshi.parse_events(js), "2026-10-03", "TOR", "MIA") == {}


def test_quote_at_ignores_candles_after_tip(monkeypatch):
    def get(path, **p):
        assert p["end_ts"] == 1000 and p["period_interval"] == 1
        return {"candlesticks": [
            {"end_period_ts": 940, "yes_ask": {"close_dollars": "0.6000"}},
            {"end_period_ts": 1000, "yes_ask": {"close": 63},           # cents
             "yes_bid": {"close": 61}},
            {"end_period_ts": 1060, "yes_ask": {"close_dollars": "0.9900"}}]}
    monkeypatch.setattr(kalshi, "get", get)
    assert kalshi.quote_at("T", 1000) == (0.61, 0.63)


def test_q_from_midpoints_not_wide_asks():
    # GSW @ POR as the probe saw it: 18/76 and 24/82 -> asks sum to 1.58
    o = kalshi.odds_from_quotes((0.24, 0.82), (0.18, 0.76), "cur")
    assert abs(o["cur_q_home"] - 0.53 / (0.53 + 0.47)) < 1e-5
    # no bid on a side: fall back to the two costs normalised
    o = kalshi.odds_from_quotes((None, 0.62), (0.30, 0.41), "cur")
    assert abs(o["cur_q_home"] - market.devig(o["cur_home_ml"], o["cur_away_ml"])) < 1e-5
