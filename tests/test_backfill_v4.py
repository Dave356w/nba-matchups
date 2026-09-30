import numpy as np
import pandas as pd

import backfill_history as bf
import build_site
import cold_start
import ledger
import nba_composite as nc


def fake_games(y, n=200, seed=0):
    rng = np.random.default_rng(seed + y)
    d = pd.Timestamp(f"{y - 1}-11-15") + pd.to_timedelta(np.arange(n) % 140, "D")
    delta = rng.normal(0, 25, n)
    phase = [nc.season_phase(x, f"{y - 1}-10-22") for x in d]
    p = 1 / (1 + np.exp(-(0.25 + 0.035 * delta)))
    return pd.DataFrame(dict(
        year=y, date=d, home=[f"H{i:03d}" for i in range(n)],
        away=[f"A{i:03d}" for i in range(n)], h_b2b=0, a_b2b=0,
        b2b_net=rng.integers(-1, 2, n), delta=delta, phase=phase,
        d_phase=delta * np.array(phase),
        win=(rng.random(n) < p).astype(int)))


def test_reconstruct_v4_tags_and_availability_arm(monkeypatch):
    monkeypatch.setattr(nc, "fit_weights", lambda years: {})
    monkeypatch.setattr(nc, "build_games", lambda y, w: fake_games(y))
    monkeypatch.setattr(cold_start, "fit_early", lambda years, w: None)
    monkeypatch.setattr(cold_start, "early_games", lambda *a, **k: pd.DataFrame())
    terms = {}
    for t in bf.REPORT_YEARS:
        g = fake_games(t)
        g = g.iloc[:150]                        # a report for 150 of 200 games
        terms[t] = pd.DataFrame(dict(
            slate_date=g["date"].dt.strftime("%Y-%m-%d"), home=g["home"],
            away=g["away"], av_min=0.1, av_bpm=0.5))
    test, _, model = bf.reconstruct_season(2026, terms=terms)
    assert model["features"] == nc.PHASE_FEATURES
    tags = test["model_tag"].value_counts().to_dict()
    assert tags == {build_site.MODEL_TAG_V4_AVAIL: 150, build_site.MODEL_TAG_V4: 50}
    base, _, _ = bf.reconstruct_season(2026, terms=None)
    assert (base["model_tag"] == build_site.MODEL_TAG_V4).all()
    # games without a report keep the base prediction
    assert np.allclose(test["p_home"].iloc[150:], base["p_home"].iloc[150:])


def test_rescore_keeps_prices_and_results():
    recon = pd.DataFrame([dict(
        game_id=str(i), slate_date="2026-01-0%d" % (i + 1), season=2026,
        tip_utc="2026-01-01T00:00Z", model_tag=build_site.MODEL_TAG,
        basis="reconstructed", home="HOM", away="AWY", delta=1.0, p_home=0.6,
        lean="HOM", p_lean=0.6, close_home_ml=-150, close_away_ml=130,
        close_q_home=0.58, close_book="dk", home_pts=100, away_pts=90,
        home_won=1) for i in range(3)], columns=ledger.COLUMNS)
    test = pd.DataFrame(dict(
        date=pd.to_datetime(["2026-01-01", "2026-01-02"]), home="HOM",
        away="AWY", delta=[-5.0, 7.0], p_home=[0.4, 0.7],
        model_tag=[build_site.MODEL_TAG_V4, build_site.MODEL_TAG_V4_AVAIL]))
    out, n = bf.rescore(recon, test)
    assert n == 2
    assert list(out["p_home"]) == [0.4, 0.7, 0.6]
    assert list(out["lean"]) == ["AWY", "HOM", "HOM"]
    assert list(out["p_lean"]) == [0.6, 0.7, 0.6]
    assert list(out["model_tag"]) == [build_site.MODEL_TAG_V4,
                                      build_site.MODEL_TAG_V4_AVAIL,
                                      build_site.MODEL_TAG]
    for c in ["close_home_ml", "close_q_home", "home_won", "close_book"]:
        assert list(out[c]) == list(recon[c])


def test_season_phase_is_capped_and_known_pregame():
    assert nc.season_phase("2025-10-21", "2025-10-21") == 0.0
    assert abs(nc.season_phase("2025-12-30", "2025-10-21") - 70 / 175) < 1e-12
    assert nc.season_phase("2026-06-01", "2025-10-21") == 1.0
    assert nc.season_phase("2025-10-01", "2025-10-21") == 0.0
    assert nc.season_phase("2025-12-30", None) == 0.0
    logs = {"A": pd.DataFrame(dict(date=pd.to_datetime(["2025-10-22", "2025-10-25"]))),
            "B": pd.DataFrame(dict(date=pd.to_datetime(["2025-10-21"])))}
    assert nc.season_opening(logs) == pd.Timestamp("2025-10-21")
