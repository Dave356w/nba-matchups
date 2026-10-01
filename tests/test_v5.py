import json
import os

import numpy as np
import pandas as pd

import backfill_history as bf
import build_site
import cold_start
import nba_composite as nc
import player_availability as pav
from test_team_quality import WEIGHTS, league


def v5_model(avail=False):
    feats = (pav.FEATURES_V5 if avail else nc.V5_FEATURES)
    return {"features": list(feats), "intercept": 0.2,
            "coef": [0.02] * len(feats), "half_life": 25.0}


def v4_model():
    return {"features": list(nc.PHASE_FEATURES), "intercept": 0.29,
            "coef": [0.023, 0.31, 0.027], "half_life": 25.0}


def test_opp_luck_and_league_rate_use_only_earlier_games():
    logs = league(seed=1, rounds=20, opp_three={"T0": 0.50})
    d = sorted({x for lg in logs.values() for x in lg["date"]})[10]
    pct = nc.league_3p_before(logs, d)
    assert abs(pct - nc.league_3p(logs)[d]) < 1e-12 and 0.3 < pct < 0.45
    t0 = logs["T0"]
    i = int((t0["date"] < d).sum())
    luck = nc.opp_luck(t0, i, pct, WEIGHTS)
    assert luck > 0                       # opponents shot hot: T0 is better
    later = t0.copy()
    later.loc[i:, ["O3P", "OFG"]] = 0     # games on/after the date
    assert nc.opp_luck(later, i, pct, WEIGHTS) == luck
    assert np.isnan(nc.opp_luck(t0, 0, pct, WEIGHTS))
    assert np.isnan(nc.opp_luck(t0.drop(columns=nc.X3), i, pct, WEIGHTS))


def test_build_games_carries_v5_terms(monkeypatch):
    logs = league(seed=2, rounds=24)
    monkeypatch.setattr(nc, "load_logs", lambda y, refresh=False: logs)
    g = nc.build_games(2026, WEIGHTS, talent=lambda tm, d: {"T0": 5.0}.get(tm, 1.0))
    assert np.isfinite(g["luck_def"]).all()
    assert set(g["talent_diff"].round(6)) <= {0.0, 4.0, -4.0}
    assert nc.build_games(2026, WEIGHTS)["talent_diff"].isna().all()


def test_build_games_records_games_played_before_the_date(monkeypatch):
    logs = league(seed=3, rounds=24)
    monkeypatch.setattr(nc, "load_logs", lambda y, refresh=False: logs)
    g = nc.build_games(2026, WEIGHTS)
    assert len(g) and g[["gp_home", "gp_away"]].notna().all().all()
    for _, r in g.iterrows():
        for side in ("home", "away"):
            want = int((logs[r[side]]["date"] < r["date"]).sum())
            assert r[f"gp_{side}"] == want >= nc.MIN_GAMES


def test_model_tags_and_active_tags():
    assert build_site.model_tag(v5_model()) == build_site.MODEL_TAG_V5
    assert build_site.model_tag(v5_model(True), avail=True) == build_site.MODEL_TAG_V5_AVAIL
    assert build_site.model_tag(v4_model()) == build_site.MODEL_TAG_V4
    tags = build_site.active_tags(v5_model(), v5_model(True), v4_model())
    assert tags == [build_site.MODEL_TAG_V5, build_site.MODEL_TAG_V5_AVAIL,
                    build_site.MODEL_TAG_V4]
    assert build_site.active_tags(v4_model(), None, v4_model()) == [build_site.MODEL_TAG_V4]


def test_score_game_v5_uses_luck_and_talent_else_falls_back():
    logs = league(seed=3, rounds=24, opp_three={"T1": 0.48})
    date = sorted({x for lg in logs.values() for x in lg["date"]})[-3]
    talent = lambda tm, d: {"T1": 6.0}.get(tm, 2.0)  # noqa: E731
    h, a = "T1", "T2"
    r = build_site.score_game(logs, h, a, date, WEIGHTS, v5_model(),
                              talent=talent, fallback=v4_model())
    assert r["model_tag"] == build_site.MODEL_TAG_V5
    # reproduce: logit of v5 features by hand
    lh, la = logs[h], logs[a]
    ih, ia = int((lh["date"] < date).sum()), int((la["date"] < date).sum())
    pct = nc.league_3p_before(logs, date)
    vals = nc.logit_inputs(r["delta"], nc.rest_days(lh, ih, date),
                           nc.rest_days(la, ia, date), date, nc.season_opening(logs))
    vals["luck_def"] = nc.opp_luck(lh, ih, pct, WEIGHTS) - nc.opp_luck(la, ia, pct, WEIGHTS)
    vals["talent_diff"] = 4.0
    p = nc.predict(v5_model(), [[vals[f] for f in nc.V5_FEATURES]])[0]
    assert abs(r["p_home"] - p) < 1e-4
    # no talent: the frozen v4 logit, tagged v4
    r4 = build_site.score_game(logs, h, a, date, WEIGHTS, v5_model(),
                               talent=None, fallback=v4_model())
    assert r4["model_tag"] == build_site.MODEL_TAG_V4
    p4 = nc.predict(v4_model(), [[vals[f] for f in nc.PHASE_FEATURES]])[0]
    assert abs(r4["p_home"] - p4) < 1e-4
    # no fallback either: abstain rather than score with missing terms
    assert np.isnan(build_site.score_game(logs, h, a, date, WEIGHTS, v5_model())["p_home"])
    # v4 model files: unchanged v4 behaviour, talent ignored
    rv4 = build_site.score_game(logs, h, a, date, WEIGHTS, v4_model(), talent=talent)
    assert rv4["model_tag"] == build_site.MODEL_TAG_V4 and abs(rv4["p_home"] - p4) < 1e-4


def test_score_game_v5_avail_route():
    logs = league(seed=4, rounds=24)
    date = sorted({x for lg in logs.values() for x in lg["date"]})[-2]
    r = build_site.score_game(logs, "T0", "T3", date, WEIGHTS, v5_model(),
                              avail_model=v5_model(True),
                              avail={"av_min": 0.1, "av_bpm": -2.0},
                              talent=lambda tm, d: 1.0, fallback=v4_model())
    assert r["model_tag"] == build_site.MODEL_TAG_V5_AVAIL


def test_talent_fn_and_season_talent(monkeypatch, tmp_path):
    box = pd.DataFrame(dict(
        game_id=["g1", "g1", "g2"], team=["T0"] * 3, name=["A B", "C D", "A B"],
        date=pd.to_datetime(["2025-11-01", "2025-11-01", "2025-11-03"]),
        player_id=["a", "c", "a"], minutes=[30.0, 20.0, 36.0], pm=0.0,
        home=True, opp="T1", margin=0.0))
    monkeypatch.setattr(pav, "fetch_box", lambda y, cache_dir=None: box)
    monkeypatch.setattr(pav, "load_bpm", lambda y: {"a b": (6.0, 2000.0, 70.0)})
    f = pav.season_talent(2026, str(tmp_path))
    v = pav.player_values(box, {"a b": (6.0, 2000.0, 70.0)})[0]
    assert v["a"] > 0 and v["c"] == 0.0
    assert np.isnan(f("T0", "2025-11-01"))
    got = f("T0", "2025-11-02")                          # the 11-01 box
    assert abs(got - (30.0 / 48) * v["a"]) < 1e-9        # role: mean minutes / 48


def test_live_availability_keeps_talent_when_the_report_fails(monkeypatch):
    box = pd.DataFrame(dict(
        game_id=["g1"], team=["T0"], name=["A B"], date=pd.to_datetime(["2026-11-01"]),
        player_id=["a"], minutes=[30.0], pm=0.0, home=True, opp="T1", margin=0.0))
    monkeypatch.setattr(pav, "update_box", lambda *a, **k: box)
    monkeypatch.setattr(pav, "load_bpm", lambda y: {"a b": (6.0, 2000.0, 70.0)})

    class Broken:
        def __init__(self, *a, **k):
            raise OSError("report host down")
    monkeypatch.setattr(pav, "ReportArchive", Broken)
    live = pav.LiveAvailability(2027, "2026-11-05", pd.Timestamp("2026-11-05 17:00"))
    assert live.report is None and live.terms("g", "T0", "T1", "2026-11-05") is None
    assert live.talent("T0", "2026-11-05") > 0


def test_cli_scorer_uses_v4_for_a_v5_model(monkeypatch):
    logs = league(seed=5, rounds=24)
    date = sorted({x for lg in logs.values() for x in lg["date"]})[-2]
    monkeypatch.setattr(nc, "load_json", lambda name: v4_model())
    df = nc.score_slate(2027, date, logs=logs, schedule=[("T0", "T1")],
                        weights=WEIGHTS, model=v5_model())
    site = build_site.score_game(logs, "T0", "T1", date, WEIGHTS, v4_model())
    assert abs(df["p_home"].iloc[0] - site["p_home"]) < 1e-5


def test_frozen_v4_fallback_file_is_the_v4_base_logit():
    root = os.path.dirname(nc.__file__)
    m = json.load(open(os.path.join(root, "model", build_site.FALLBACK_FILE)))
    assert m["features"] == nc.PHASE_FEATURES and len(m["coef"]) == 3


def fake_v5_games(y, n=200, seed=0, talent=True):
    rng = np.random.default_rng(seed + y)
    d = pd.Timestamp(f"{y - 1}-11-15") + pd.to_timedelta(np.arange(n) % 140, "D")
    delta = rng.normal(0, 25, n)
    phase = np.array([nc.season_phase(x, f"{y - 1}-10-22") for x in d])
    tal = rng.normal(0, 3, n)
    luck = rng.normal(0, 2, n)
    p = 1 / (1 + np.exp(-(0.25 + 0.02 * delta + 0.06 * tal)))
    g = pd.DataFrame(dict(
        year=y, date=d, home=[f"H{i:03d}" for i in range(n)],
        away=[f"A{i:03d}" for i in range(n)], h_b2b=0, a_b2b=0,
        b2b_net=rng.integers(-1, 2, n), delta=delta, phase=phase,
        d_phase=delta * phase, luck_def=luck,
        talent_diff=tal if talent else np.nan,
        win=(rng.random(n) < p).astype(int)))
    g.loc[:9, "talent_diff"] = np.nan              # 10 games without box scores
    return g


def test_reconstruct_v5_routes_and_keeps_v4_beside_it(monkeypatch):
    monkeypatch.setattr(nc, "fit_weights", lambda years: {})
    monkeypatch.setattr(nc, "build_games",
                        lambda y, w, talent=None: fake_v5_games(y))
    monkeypatch.setattr(cold_start, "fit_early", lambda years, w: None)
    monkeypatch.setattr(cold_start, "early_games", lambda *a, **k: pd.DataFrame())
    test, _, model = bf.reconstruct_season(2026, v5=True, walk_forward=True,
                                           talent={t: None for t in range(2015, 2027)})
    assert model["features"] == nc.V5_FEATURES
    tags = test["model_tag"].value_counts().to_dict()
    assert tags == {build_site.MODEL_TAG_V5: 190, build_site.MODEL_TAG_V4: 10}
    v4, _, _ = bf.reconstruct_season(2026, v5=False, walk_forward=True)
    assert np.allclose(test["p_v4"], v4["p_home"])        # the gate's v4 arm
    fb = test["model_tag"] == build_site.MODEL_TAG_V4       # fallback rows = v4
    assert np.allclose(test.loc[fb, "p_home"], v4.loc[fb, "p_home"])


def test_opp_luck_and_build_games_accept_season_to_date(monkeypatch):
    # The Fit model backtest builds games with half_life=None (season to date).
    logs = league(seed=6, rounds=20, opp_three={"T0": 0.50})
    d = sorted({x for lg in logs.values() for x in lg["date"]})[10]
    t0 = logs["T0"]
    i = int((t0["date"] < d).sum())
    pct = nc.league_3p_before(logs, d)
    assert nc.opp_luck(t0, i, pct, WEIGHTS, None) > 0
    assert nc.opp_luck(t0, i, pct, WEIGHTS, 1e9) == \
        __import__("pytest").approx(nc.opp_luck(t0, i, pct, WEIGHTS, None))
    monkeypatch.setattr(nc, "load_logs", lambda y, refresh=False: logs)
    assert np.isfinite(nc.build_games(2026, WEIGHTS, None)["luck_def"]).all()


def test_fit_logit_offset_holds_a_coefficient_fixed():
    rng = np.random.default_rng(12)
    n = 4000
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.2 + 0.8 * x1 + 0.5 * x2)))).astype(int)
    full = nc.fit_logit(np.column_stack([x1, x2]), y, ["x1", "x2"], l2=0.0)
    # holding x2 at its full-fit coefficient reproduces the rest of that fit
    part = nc.fit_logit(x1, y, ["x1"], l2=0.0, offset=full["coef"][1] * x2)
    assert abs(part["coef"][0] - full["coef"][0]) < 1e-6
    assert abs(part["intercept"] - full["intercept"]) < 1e-6


def test_fit_fixed_takes_luck_and_talent_from_the_base_fit():
    rng = np.random.default_rng(13)
    n = 1500
    G = pd.DataFrame({f: rng.normal(0, 1, n) for f in pav.FEATURES_V5})
    G["win"] = (rng.random(n) < 1 / (1 + np.exp(-(0.3 + 0.5 * G["delta"]
                                                  + 0.4 * G["talent_diff"])))).astype(int)
    base = {"features": list(nc.V5_FEATURES), "intercept": 0.0,
            "coef": [0.0, 0.0, 0.0, 0.07, 0.25]}
    m = pav.fit_fixed(G, base)
    c = dict(zip(m["features"], m["coef"]))
    assert m["features"] == pav.FEATURES_V5
    assert c["luck_def"] == 0.07 and c["talent_diff"] == 0.25
    assert abs(c["delta"] - 0.5) < 0.2                   # the free terms are fitted
    p = nc.predict(m, G[pav.FEATURES_V5].to_numpy(float))  # scores like any model
    assert np.isfinite(p).all()


def test_reconstruct_p_fixed_equals_p_home_off_the_avail_route(monkeypatch):
    monkeypatch.setattr(nc, "fit_weights", lambda years: {})
    monkeypatch.setattr(nc, "build_games",
                        lambda y, w, talent=None: fake_v5_games(y))
    monkeypatch.setattr(cold_start, "fit_early", lambda years, w: None)
    monkeypatch.setattr(cold_start, "early_games", lambda *a, **k: pd.DataFrame())
    test, _, _ = bf.reconstruct_season(2026, v5=True, walk_forward=True,
                                       talent={t: None for t in range(2015, 2027)})
    assert np.allclose(test["p_fixed"], test["p_home"])


def test_reconstruct_fixed_arm_on_the_avail_route(monkeypatch):
    monkeypatch.setattr(nc, "fit_weights", lambda years: {})
    monkeypatch.setattr(nc, "build_games",
                        lambda y, w, talent=None: fake_v5_games(y))
    monkeypatch.setattr(cold_start, "fit_early", lambda years, w: None)
    monkeypatch.setattr(cold_start, "early_games", lambda *a, **k: pd.DataFrame())
    rng = np.random.default_rng(14)
    terms = {}
    for t in bf.REPORT_YEARS:
        g = fake_v5_games(t).iloc[:150]
        terms[t] = pd.DataFrame(dict(
            slate_date=g["date"].dt.strftime("%Y-%m-%d"), home=g["home"],
            away=g["away"], av_min=rng.normal(0, .1, 150), av_bpm=rng.normal(0, 1, 150)))
    test, _, base5 = bf.reconstruct_season(2026, v5=True, walk_forward=True, terms=terms,
                                           talent={t: None for t in range(2015, 2027)})
    av = (test["route"] == "avail").to_numpy()
    assert av.sum() == 140                        # 150 reported, 10 lack talent
    assert np.allclose(test.loc[~av, "p_fixed"], test.loc[~av, "p_home"])
    assert np.isfinite(test.loc[av, "p_fixed"]).all()
    assert not np.allclose(test.loc[av, "p_fixed"], test.loc[av, "p_home"])
    # by hand: fit_fixed on the earlier report seasons, base5's luck/talent
    tr = pd.concat([pav.with_terms(fake_v5_games(t), terms[t])
                    for t in bf.REPORT_YEARS if t < 2026])
    tr = tr[np.isfinite(tr[pav.FEATURES_V5].to_numpy(float)).all(axis=1)]
    fm = pav.fit_fixed(tr, base5)
    te = pav.with_terms(test[test["route"] == "avail"], terms[2026])
    want = nc.predict(fm, te[pav.FEATURES_V5].to_numpy(float))
    assert np.allclose(np.sort(want), np.sort(test.loc[av, "p_fixed"]))
