"""Kalshi as the pages' market (analysis.market_view): ledger locks on the
kalshi_* columns, the view's one-market-per-row rule, the hypotheses staying
on sportsbook prices, the historical candle format, grading and backfill."""
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


def test_kalshi_pregame_columns_follow_the_pregame_and_first_locks():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    r = led.iloc[0]
    assert r["kalshi_pre_home_ml"] == -140 and r["kalshi_first_home_ml"] == -140
    led, _, _ = ledger.upsert_pregame(led, [row(kalshi_pre_home_ml=-160)], now=LATER)
    r = led.iloc[0]
    assert r["kalshi_pre_home_ml"] == -160          # refreshed before tip
    assert r["kalshi_first_home_ml"] == -140         # first snapshot: written once
    led2, acc, _ = ledger.upsert_pregame(led, [row(kalshi_pre_home_ml=-999)], now=AFTER)
    assert not acc and led2.iloc[0]["kalshi_pre_home_ml"] == -160   # frozen at tip
    assert set(ledger.KALSHI_PRE_COLUMNS) <= set(ledger.PREGAME_COLUMNS)
    assert set(ledger.KALSHI_FIRST_COLUMNS) <= set(ledger.WRITE_ONCE_COLUMNS)
    assert set(ledger.KALSHI_GRADE_COLUMNS) <= set(ledger.GRADE_COLUMNS)


def test_kalshi_close_only_on_a_completed_game_and_only_grade_columns():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    before = led[ledger.PREGAME_COLUMNS + ledger.FIRST_COLUMNS].copy()
    k = dict(kalshi_close_home_ml=-170, kalshi_close_away_ml=150,
             kalshi_close_q_home=0.62, kalshi_open_home_ml=-140,
             kalshi_open_away_ml=125)
    assert not ledger.apply_result(led, "401", dict(completed=False, home_pts=50,
                                                    away_pts=40), None, k)
    assert pd.isna(led.iloc[0]["kalshi_close_q_home"])        # never while pending
    assert ledger.apply_result(led, "401", dict(completed=True, home_pts=110,
                                                away_pts=104), None, k)
    pd.testing.assert_frame_equal(led[ledger.PREGAME_COLUMNS + ledger.FIRST_COLUMNS],
                                  before)
    assert led.iloc[0]["kalshi_close_q_home"] == 0.62


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


def test_market_view_prices_each_row_by_one_market():
    df = with_kalshi(synth(200, 1), frac=0.5)
    df["first_snapshot_utc"] = "2026-11-19T15:00:00Z"
    v = analysis.market_view(df)
    k = df["kalshi_close_q_home"].notna()
    assert (v.loc[k, "close_book"] == "kalshi").all()
    assert (v.loc[k, "pre_book"] == "kalshi").all()
    assert (v.loc[k, "first_book"] == "kalshi").all()
    assert (v.loc[k, "close_q_home"] == df.loc[k, "kalshi_close_q_home"]).all()
    assert (v.loc[k, "pre_home_ml"] == df.loc[k, "kalshi_pre_home_ml"]).all()
    pd.testing.assert_frame_equal(v.loc[~k], df.loc[~k].astype(v.dtypes.to_dict()),
                                  check_dtype=False)
    # model, result and the input frame are untouched
    assert (v["p_home"] == df["p_home"]).all() and (v["home_won"] == df["home_won"]).all()
    assert (df.loc[k, "close_book"] == "dk").all()
    # pages split by book: Kalshi rows form their own section, never pooled
    books = dict(analysis.book_split(analysis.with_close(ledger.graded(v))))
    assert set(books) == {"dk", "kalshi"} and len(books["kalshi"]) == k.sum()


def test_market_view_pending_rows_use_kalshi_pregame():
    df = with_kalshi(synth(20, 2))
    df.loc[:9, ["home_won", "home_pts", "away_pts"]] = np.nan
    df.loc[:9, ledger.KALSHI_GRADE_COLUMNS] = np.nan
    v = analysis.market_view(df)
    assert (v.loc[:9, "pre_book"] == "kalshi").all()
    assert v.loc[:9, "close_book"].eq("dk").all()            # no close yet
    assert v.loc[:9, "close_q_home"].equals(df.loc[:9, "close_q_home"].astype(float))


def test_hypotheses_stay_on_sportsbook_prices(tmp_path, monkeypatch):
    nat = with_kalshi(synth(200, 3, "native"), shift=0.2)
    rec = with_kalshi(synth(300, 4, "reconstructed"), shift=0.2)
    for df in (nat, rec):
        df["open_home_ml"], df["open_away_ml"] = df["close_home_ml"], df["close_away_ml"]
    raw = analysis.hypothesis_rows(nat, rec)
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(nat, rec, "2026-11-19", model_ok=True)
    grades = (tmp_path / "grades.html").read_text()
    assert "Kalshi close" in grades and "DraftKings close" not in grades
    for hyp, hind, _nat in raw:                     # hindsight n at book prices
        if hind:
            assert f"{hind['n']} <span class='mut'>at the" in grades
    report = (tmp_path / "ledger_report.txt").read_text()
    assert "Kalshi close" in report and "Market: Kalshi" in report


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
    assert backfill_history.main(["--seasons", "2025", "2026", "--kalshi"]) == 0
    after = ledger.load(ledger.RECON_PATH)
    assert len(asked) == 20                          # 2024-25 has no Kalshi
    other = [c for c in ledger.COLUMNS if c not in ledger.KALSHI_COLUMNS]
    pd.testing.assert_frame_equal(after[other], before[other])
    s26 = pd.to_numeric(after["season"]) == 2026
    assert (after.loc[s26, "kalshi_close_q_home"] == 0.62).all()
    assert after.loc[~s26, "kalshi_close_q_home"].isna().all()
    asked.clear()
    backfill_history.main(["--seasons", "2026", "--kalshi"])
    assert not asked                                 # filled rows are skipped


def test_missing_kalshi_close_is_retried_for_kalshi_seasons():
    df = synth(10, 6)
    df["slate_date"] = "2026-11-18"
    df["season"] = [2025] * 5 + [2027] * 5
    m = ledger.missing_close(df, "2026-11-19", kalshi=True)
    assert len(m) == 5 and (m["season"] == 2027).all()
    assert not len(ledger.missing_close(df, "2026-11-19"))     # book close present


def test_build_adds_kalshi_pregame_and_survives_a_failure(monkeypatch):
    js = {"events": [{"event_ticker": "KXNBAGAME-26NOV19UTAGSW", "markets": [
        {"ticker": "KXNBAGAME-26NOV19UTAGSW-GSW", "yes_ask_dollars": "0.6200",
         "yes_bid_dollars": "0.6100"},
        {"ticker": "KXNBAGAME-26NOV19UTAGSW-UTA", "yes_ask_dollars": "0.3900",
         "yes_bid_dollars": "0.3800"}]}]}
    monkeypatch.setattr(kalshi, "open_games", lambda: kalshi.parse_events(js))
    rows = [dict(slate_date="2026-11-19", home="GSW", away="UTA"),
            dict(slate_date="2026-11-19", home="BOS", away="LAL")]
    build_site.add_kalshi_pregame(rows, "2026-11-19")
    assert rows[0]["kalshi_pre_home_ml"] == market.american_from_prob(
        market.kalshi_cost(0.62))
    assert abs(rows[0]["kalshi_pre_q_home"] - 0.615 / (0.615 + 0.385)) < 1e-4
    assert np.isnan(rows[1]["kalshi_pre_home_ml"])           # not listed
    def boom():
        raise OSError("down")
    monkeypatch.setattr(kalshi, "open_games", boom)
    rows = [dict(slate_date="2026-11-19", home="GSW", away="UTA")]
    build_site.add_kalshi_pregame(rows, "2026-11-19")
    assert "kalshi_pre_home_ml" not in rows[0]               # book only
