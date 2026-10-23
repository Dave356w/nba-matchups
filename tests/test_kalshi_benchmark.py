"""Kalshi beside the sportsbook: ledger locks on the kalshi_* columns, the
view's one-market-per-row rule (analysis.market_view, for bases in
KALSHI_BASES: none since 2026-10-21, when the pages moved to DraftKings),
the hypotheses staying on sportsbook prices, the historical candle format,
grading and backfill."""
from conftest import write_research
import urllib.error
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import analysis
import backfill_history
import build_site
import kalshi
import ledger
import market
from test_analysis_and_pages import synth

TIP = "2026-11-20T00:30Z"
BEFORE = datetime(2026, 11, 19, 18, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 11, 19, 23, 0, tzinfo=timezone.utc)
AFTER = datetime(2026, 11, 20, 1, 0, tzinfo=timezone.utc)
KPRE = dict(kalshi_pre_home_ml=-140, kalshi_pre_away_ml=125, kalshi_pre_q_home=0.57)


def row(**kw):
    r = dict(game_id="401", slate_date="2026-11-19", season=2027, tip_utc=TIP,
             model_tag="t", home="GSW", away="UTA", delta=5.0, p_home=0.62,
             lean="GSW", p_lean=0.62, pre_book="dk", pre_home_ml=-150,
             pre_away_ml=130, pre_q_home=0.585, **KPRE)
    r.update(kw)
    return r


def test_pre_kalshi_file_loads_with_blank_kalshi_columns(tmp_path):
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    p = tmp_path / "x.csv"
    led[ledger.PRE_KALSHI_COLUMNS].to_csv(p, index=False)
    back = ledger.load(str(p))
    assert list(back.columns) == ledger.COLUMNS
    assert back[ledger.KALSHI_COLUMNS].isna().all().all()


def with_kalshi(df, frac=1.0, shift=0.03, seed=0):
    """Kalshi prices on a fraction of synth rows, q moved by `shift`."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    on = rng.random(len(df)) < frac
    for kind in ("pre", "close", "first"):
        q = (df["close_q_home"].astype(float) + shift).clip(0.05, 0.95)
        h = [market.american_from_prob(x + 0.02) for x in q]
        a = [market.american_from_prob(1 - x + 0.02) for x in q]
        df.loc[on, f"kalshi_{kind}_home_ml"] = np.array(h, float)[on]
        df.loc[on, f"kalshi_{kind}_away_ml"] = np.array(a, float)[on]
        df.loc[on, f"kalshi_{kind}_q_home"] = q[on]
    df.loc[on, "kalshi_open_home_ml"] = df.loc[on, "kalshi_close_home_ml"]
    df.loc[on, "kalshi_open_away_ml"] = df.loc[on, "kalshi_close_away_ml"]
    return df


def test_pages_grade_every_basis_at_its_sportsbook_by_default():
    # owner's decision, 2026-10-21: live and rebuilt games on DraftKings
    df = pd.concat([with_kalshi(synth(30, 1, "native")),
                    with_kalshi(synth(30, 2, "reconstructed"))], ignore_index=True)
    v = analysis.market_view(df)
    assert v is df
    assert (v["close_book"] == "dk").all() and (v["pre_book"] == "dk").all()
    assert v["close_q_home"].equals(df["close_q_home"])


def test_hypotheses_stay_on_sportsbook_prices(tmp_path, monkeypatch):
    nat = with_kalshi(synth(200, 3, "native"), shift=0.2)
    rec = with_kalshi(synth(300, 4, "reconstructed"), shift=0.2)
    for df in (nat, rec):
        df["open_home_ml"], df["open_away_ml"] = df["close_home_ml"], df["close_away_ml"]
    raw = analysis.hypothesis_rows(nat, rec, book=analysis.HYPOTHESIS_BOOK)
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    write_research(nat, rec, "2026-11-19", model_ok=True)
    grades = (tmp_path / "grades.html").read_text()
    # native and rebuilt sections both on the DraftKings close (2026-10-21)
    nat_html = grades[grades.index("id='forward'"):grades.index("id='rebuilt'")]
    rec_html = grades[grades.index("id='rebuilt'"):]
    assert "DraftKings close" in nat_html and "DraftKings close" in rec_html
    assert "Kalshi close" not in grades
    cal = (tmp_path / "market-calibration.html").read_text()
    assert "Market · Kalshi" not in cal and "Judge every ROI" not in cal
    for hyp, hind, _nat in raw:                     # hindsight n at book prices
        if hind and not hyp.get("retired"):
            assert f"{hind['n']}<br><span class='mut'>at the" in grades
    report = (tmp_path / "ledger_report.txt").read_text()
    assert "Market: DraftKings" in report and "DraftKings close" in report
    assert "Kalshi close" not in report.split("== PRESEASON")[0]


def test_historical_candles_are_dollar_strings_and_fallback(monkeypatch):
    calls = []

    def get(path, **p):
        calls.append(path)
        if path.startswith("/series/"):
            raise urllib.error.HTTPError(path, 404, "nf", None, None)
        return {"candlesticks": [
            {"end_period_ts": 940, "yes_ask": {"close": "0.7700"},
             "yes_bid": {"close": "0.7600"}},
            {"end_period_ts": 1060, "yes_ask": {"close": "0.9900"}}]}
    monkeypatch.setattr(kalshi, "get", get)
    assert kalshi.quote_at("T", 1000) == (0.76, 0.77)        # not 0.0077
    assert calls == ["/series/KXNBAGAME/markets/T/candlesticks",
                     "/historical/markets/T/candlesticks"]
    assert kalshi._close({"yes_ask": {"close": 63}}, "yes_ask") == 0.63


def test_grade_odds_close_and_open(monkeypatch):
    def get(path, **p):
        side = path.split("/")[-2].rsplit("-", 1)[-1]
        ask = {"BOS": 0.66, "LAL": 0.37}[side]
        first = {"BOS": 0.60, "LAL": 0.43}[side]
        if p["period_interval"] == 60:
            return {"candlesticks": [
                {"end_period_ts": p["start_ts"] + 3600,
                 "yes_ask": {"close_dollars": f"{first}"},
                 "yes_bid": {"close_dollars": f"{first - .01:.2f}"}}]}
        return {"candlesticks": [{"end_period_ts": p["end_ts"],
                                  "yes_ask": {"close_dollars": f"{ask}"},
                                  "yes_bid": {"close_dollars": f"{ask - .01:.2f}"}}]}
    monkeypatch.setattr(kalshi, "get", get)
    o = kalshi.grade_odds("2026-01-08", "BOS", "LAL", "2026-01-09T00:30Z")
    assert set(o) == set(ledger.KALSHI_GRADE_COLUMNS)
    assert o["kalshi_close_home_ml"] == market.american_from_prob(market.kalshi_cost(0.66))
    assert o["kalshi_open_home_ml"] == market.american_from_prob(market.kalshi_cost(0.60))
    assert abs(o["kalshi_close_q_home"] - 0.655 / (0.655 + 0.365)) < 1e-4
    monkeypatch.setattr(kalshi, "get", lambda path, **p: {})
    assert kalshi.grade_odds("2026-01-08", "BOS", "LAL", "2026-01-09T00:30Z") == {}


def test_backfill_kalshi_fills_only_kalshi_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = synth(30, 5, "reconstructed")
    df["season"] = [2025] * 10 + [2026] * 20
    ledger.save(df, ledger.RECON_PATH)
    before = ledger.load(ledger.RECON_PATH)
    k = dict(kalshi_close_home_ml=-170, kalshi_close_away_ml=150,
             kalshi_close_q_home=0.62, kalshi_open_home_ml=-140,
             kalshi_open_away_ml=125)
    asked = []
    monkeypatch.setattr(kalshi, "grade_odds",
                        lambda d, h, a, t: asked.append(d) or dict(k))
    assert __import__("research.kalshi_backfill", fromlist=["backfill_kalshi"]).backfill_kalshi([2025, 2026]) == 0
    after = ledger.load(ledger.RECON_PATH)
    assert len(asked) == 20                          # 2024-25 has no Kalshi
    other = [c for c in ledger.COLUMNS if c not in ledger.KALSHI_COLUMNS]
    pd.testing.assert_frame_equal(after[other], before[other])
    s26 = pd.to_numeric(after["season"]) == 2026
    assert (after.loc[s26, "kalshi_close_q_home"] == 0.62).all()
    assert after.loc[~s26, "kalshi_close_q_home"].isna().all()
    asked.clear()
    __import__("research.kalshi_backfill", fromlist=["backfill_kalshi"]).backfill_kalshi([2026])
    assert not asked                                 # filled rows are skipped


def test_postponed_game_keeps_its_original_ticker_date(monkeypatch):
    tip = "2026-01-25T22:30Z"
    tip_ts = ledger.parse_utc(tip).timestamp()

    def get(path, **p):
        if "26JAN24GSWMIN" in path and p["end_ts"] <= tip_ts:
            ask = 0.65 if path.split("/")[-2].endswith("MIN") else 0.37
            return {"candlesticks": [{"end_period_ts": int(tip_ts) - 60,
                                      "yes_ask": {"close_dollars": f"{ask}"},
                                      "yes_bid": {"close_dollars": f"{ask - .01:.2f}"}}]}
        return {}
    monkeypatch.setattr(kalshi, "get", get)
    o = kalshi.close_odds("2026-01-25", "MIN", "GSW", tip)
    assert o["close_home_ml"] == market.american_from_prob(market.kalshi_cost(0.65))
    # a game three days earlier is never picked up
    monkeypatch.setattr(kalshi, "get", lambda path, **p: get(
        path.replace("26JAN24", "26JAN22"), **p) if "26JAN24" in path else {})
    assert kalshi.close_odds("2026-01-25", "MIN", "GSW", tip) == {}
