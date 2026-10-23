"""Neutral optional inputs preserve the v9 formula and historical data."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import backfill_history as bf
import build_site as bs
import ledger
import nba_composite as nc
import validate_data_files as vdf
from test_v8 import teams
from test_v9 import MBASE, MAVAIL, _pregame


def test_missing_talent_uses_same_margin_formula_at_every_stage(weights):
    for date in ("2026-10-21", "2026-11-02", "2027-02-01"):
        cur, prior = teams(30)
        kw = dict(prior_logs=prior, opening="2026-10-21")
        missing = bs.score_game(cur, "A", "B", date, weights, MBASE, **kw)
        neutral = bs.score_game(cur, "A", "B", date, weights, MBASE,
                                talent=lambda t, d: 0.0, talent0={"A": 0, "B": 0}, **kw)
        assert missing["model_tag"] == bs.MODEL_TAG_V9_NEUTRAL
        assert neutral["model_tag"] == bs.MODEL_TAG_V9
        assert missing["p_home"] == neutral["p_home"]
        assert missing["margin_home"] == neutral["margin_home"]
        assert "carry_p_home" not in missing


def test_missing_injury_terms_are_jointly_neutral_and_tagged(weights):
    cur, prior = teams(12)
    kw = dict(prior_logs=prior, avail_model=MAVAIL, talent=lambda t, d: 1.0)
    a = bs.score_game(cur, "A", "B", "2026-11-10", weights, MBASE,
                      avail={"av_min": np.nan, "av_bpm": 2}, **kw)
    b = bs.score_game(cur, "A", "B", "2026-11-10", weights, MBASE,
                      avail={"av_min": 0, "av_bpm": 0}, **kw)
    assert a["model_tag"] == bs.MODEL_TAG_V9_NEUTRAL_AVAIL
    assert a["p_home"] == b["p_home"] and a["margin_home"] == b["margin_home"]
    assert bs.route(a, MBASE) == "neutral"


def test_missing_core_rating_still_abstains(weights):
    a = bs.score_game({}, "A", "B", "2026-10-21", weights, MBASE)
    assert np.isnan(a["p_home"]) and np.isnan(a["margin_home"])


def test_carry_columns_are_readable_but_no_longer_written(tmp_path):
    old = pd.DataFrame([{**{c: np.nan for c in ledger.COLUMNS}, **_pregame(),
                         "basis": "native", "snapshot_utc": "2027-01-09T19:00Z",
                         "gp_home": 3, "gp_away": 3, "carry_p_home": 0.61,
                         "first_carry_p_home": 0.60}])
    path = tmp_path / "history.csv"
    ledger.save(old, path)
    assert vdf.check(path) == []
    old = ledger.load(path)
    now = datetime(2027, 1, 9, 20, tzinfo=timezone.utc)
    updated, _, _ = ledger.upsert_pregame(old, [_pregame()], now=now)
    assert updated.iloc[0].carry_p_home == 0.61
    assert updated.iloc[0].first_carry_p_home == 0.60
    fresh, _, _ = ledger.upsert_pregame(ledger.empty(), [_pregame()], now=now)
    assert fresh[ledger.CARRY_COLUMNS + ledger.FIRST_CARRY_COLUMNS].isna().all().all()
    assert not set(ledger.CARRY_COLUMNS + ledger.FIRST_CARRY_COLUMNS) & set(ledger.GRADE_COLUMNS)


def test_rescore_preserves_historical_carry_columns():
    source = pd.DataFrame([_pregame()]).assign(carry_p_home=0.61)
    test = pd.DataFrame([dict(date=pd.Timestamp("2027-01-09"), home="AAA", away="BBB",
                              gp_home=30, gp_away=30, p_home=0.7, delta=4,
                              margin_home=5, model_tag=bs.MODEL_TAG_V9)])
    out, n = bf.rescore(source, test)
    assert n == 1 and out.iloc[0].carry_p_home == 0.61
    assert out.iloc[0].p_home == 0.7


def test_reconstruction_matches_live_neutral_policy_without_extra_fits(monkeypatch):
    import player_availability as pav
    def frame(year):
        d = pd.DataFrame({f: [0.3, 0.4, 0.5] for f in nc.V6_FEATURES})
        d["talent_diff"] = [np.nan, 1.0, np.nan]
        d["date"] = pd.to_datetime([f"{year}-01-01", f"{year}-01-02", f"{year}-01-03"])
        return d.assign(home="A", away="B", gp_home=[3, 20, 30], gp_away=30,
                        margin=[4, 6, -3], win=[1, 1, 0])
    monkeypatch.setattr(nc, "fit_weights", lambda years: {})
    monkeypatch.setattr(nc, "fade_v8_games", lambda year, *args, **kw: frame(year))
    fitted = []
    def fit(x, y, features):
        fitted.append(list(features))
        return MAVAIL if features == pav.FEATURES_V6 else MBASE
    monkeypatch.setattr(nc, "fit_margin", fit)
    terms = {y: pd.DataFrame(dict(slate_date=[f"{y}-01-01", f"{y}-01-02"],
                                  home="A", away="B", av_min=[np.nan, 0.2],
                                  av_bpm=[3.0, 0.3])) for y in (2025, 2026)}
    out, _, _ = bf.reconstruct_season_v8(
        2026, terms=terms, weight_years=[2025], phase_years=[2025],
        report_years=[2025], talent={2025: None, 2026: None},
        talent0={2025: {}, 2026: {}}, margin=True)
    assert fitted == [nc.V6_FEATURES, pav.FEATURES_V6]
    assert list(out.model_tag) == [bs.MODEL_TAG_V9_NEUTRAL_AVAIL,
                                    bs.MODEL_TAG_V9_AVAIL, bs.MODEL_TAG_V9_NEUTRAL]
    x = {**frame(2026).iloc[0].to_dict(), "talent_diff": 0, "av_min": 0, "av_bpm": 0}
    expected = nc.predict(MAVAIL, [[x[f] for f in MAVAIL["features"]]])[0]
    assert np.isclose(out.iloc[0].p_home, expected)
    assert "carry_p_home" not in out
