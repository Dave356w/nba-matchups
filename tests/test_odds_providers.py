import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "odds_providers",
    os.path.join(os.path.dirname(__file__), "..", "research", "odds_providers.py"))
op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(op)


def test_summarize_providers_lists_every_book():
    js = {"items": [
        {"provider": {"id": "58", "name": "ESPN BET", "priority": 1},
         "homeTeamOdds": {"moneyLine": -150,
                          "open": {"moneyLine": {"american": "-140"}},
                          "close": {"moneyLine": {"american": "-160"}}},
         "awayTeamOdds": {"close": {"moneyLine": {"american": "+135"}}}},
        {"provider": {"id": "100", "name": "DraftKings"},
         "homeTeamOdds": {"current": {"moneyLine": {"american": "-155"}}}},
        {"$ref": "http://example/odds/200"},
    ]}
    rows = op.summarize_providers(js)
    assert [r["id"] for r in rows] == ["58", "100", "None"]
    assert rows[0]["home_open"] == -140 and rows[0]["home_close"] == -160
    assert rows[0]["away_close"] == 135 and rows[0]["home_ml"] == -150
    assert rows[1]["home_cur"] == -155 and rows[1]["home_close"] is None
    assert rows[2]["ref_only"] and not rows[0]["ref_only"]
