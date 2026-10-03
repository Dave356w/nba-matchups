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


def test_one_sided_book_is_unpriced():
    js = {"events": [{"event_ticker": "KXNBAGAME-26OCT03MIATOR", "markets": [
        {"ticker": "KXNBAGAME-26OCT03MIATOR-MIA", "yes_ask_dollars": "0.3600"},
        {"ticker": "KXNBAGAME-26OCT03MIATOR-TOR", "yes_ask_dollars": None}]}]}
    assert kalshi.pregame_odds(kalshi.parse_events(js), "2026-10-03", "TOR", "MIA") == {}


def test_ask_at_ignores_candles_after_tip(monkeypatch):
    def get(path, **p):
        assert p["end_ts"] == 1000 and p["period_interval"] == 1
        return {"candlesticks": [
            {"end_period_ts": 940, "yes_ask": {"close_dollars": "0.6000"}},
            {"end_period_ts": 1000, "yes_ask": {"close": 63}},          # cents
            {"end_period_ts": 1060, "yes_ask": {"close_dollars": "0.9900"}}]}
    monkeypatch.setattr(kalshi, "get", get)
    assert kalshi.ask_at("T", 1000) == 0.63
