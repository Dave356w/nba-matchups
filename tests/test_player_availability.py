from datetime import datetime

import numpy as np
import pandas as pd

import build_site
import player_availability as pav


def box_rows(team, games, opp="O", start="2026-11-01"):
    out = []
    for k, mins in enumerate(games):
        for pid, m in mins.items():
            out.append(dict(game_id=f"{team}{k:02d}", date=pd.Timestamp(start)
                            + pd.Timedelta(days=2 * k), team=team, opp=opp,
                            home=True, margin=3.0, player_id=pid, name=pid.title(),
                            minutes=m, pm=1.0))
    return pd.DataFrame(out)


def test_upcoming_terms_match_the_backtest_computation():
    games = [{"star": 36, "wing": 30, "bench": 12}] * 11 + [{"star": 0, "wing": 30,
                                                            "bench": 12}]
    box = box_rows("T", games)
    val = {"star": 8.0, "wing": 2.0, "bench": 0.0}
    present = {("T11", "star"): 0.0}
    hist = pav.team_availability(box, value=val, present=present)["T11"]
    upcoming = pav.upcoming_terms(box[box["game_id"] != "T11"], "T", "T11",
                                  box["date"].max(), val, present)
    assert np.allclose(hist, upcoming)
    assert upcoming[2] < 0                       # star out -> negative value term
    assert pav.upcoming_terms(box, "NOPE", "g", box["date"].max(), val, {}) is None


def test_od_present_maps_statuses_by_team_and_name():
    box = box_rows("BOS", [{"jaylen brown": 30, "jrue holiday": 30}])
    rep = pd.DataFrame(dict(
        game_date=["11/05/2026"] * 3, matchup=[""] * 3,
        team=["Boston Celtics", "Boston Celtics", "New York Knicks"],
        player=["Brown, Jaylen", "Holiday, Jrue", "Brunson, Jalen"],
        status=["Out", "Questionable", "Out"]))
    got = pav.od_present(rep, box, "g1", {"BOS"}, "2026-11-05")
    assert got == {("g1", "jaylen brown"): 0.0, ("g1", "jrue holiday"): 1.0}
    assert pav.od_present(rep, box, "g1", {"BOS"}, "2026-11-06") == {}


def test_update_box_is_incremental_and_waits_for_pending_games(tmp_path, monkeypatch):
    calls = []

    def fake(d, sleep=0):
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        calls.append(ds)
        if ds == "2026-10-22":
            return ([dict(game_id="1", date=pd.Timestamp(ds), team="BOS", opp="NYK",
                          home=True, margin=5, player_id="7", name="A", minutes=30,
                          pm=2)], 1, 0)
        return [], 0, 1 if ds == "2026-10-23" else 0
    monkeypatch.setattr(pav, "box_rows_for_date", fake)
    monkeypatch.setattr(pav, "season_dates",
                        lambda y: pd.date_range("2026-10-21", "2026-10-24"))
    df = pav.update_box(2027, "2026-10-24", str(tmp_path))
    assert list(df["game_id"]) == ["1"] and len(calls) == 3
    calls.clear()
    df = pav.update_box(2027, "2026-10-25", str(tmp_path))
    assert calls == ["2026-10-23", "2026-10-24"]   # done dates skipped, pending retried
    assert list(df["game_id"]) == ["1"]


def test_recent_report_misses_are_not_remembered(tmp_path):
    now = datetime(2026, 11, 5, 18, 10)
    arc = pav.ReportArchive(str(tmp_path), fetch=lambda url: None, sleep=0, now=now)
    arc.get(datetime(2026, 11, 5, 18, 0))
    arc.get(datetime(2026, 11, 5, 9, 0))
    assert arc.misses == {"2026-11-05_09AM", "2026-11-05_09_00AM"}


def test_score_game_uses_v3_only_with_terms(model, weights):
    from conftest import make_log
    logs = {"GSW": make_log(30, start="2026-10-21", seed=1, strength=1.0),
            "UTA": make_log(30, start="2026-10-22", seed=2, strength=-1.0)}
    avail_model = {"features": pav.FEATURES, "intercept": model["intercept"],
                   "coef": model["coef"] + [0.0, 0.1]}
    date = "2027-01-10"
    v2 = build_site.score_game(logs, "GSW", "UTA", date, weights, model,
                               avail_model=avail_model, avail=None)
    assert v2["model_tag"] == build_site.MODEL_TAG            # no terms: v2
    same = build_site.score_game(logs, "GSW", "UTA", date, weights, model,
                                 avail_model=avail_model,
                                 avail={"av_min": 0.0, "av_bpm": 0.0})
    assert same["model_tag"] == build_site.MODEL_TAG_V3
    assert abs(same["p_home"] - v2["p_home"]) < 1e-9     # zero terms = v2 math
    hurt = build_site.score_game(logs, "GSW", "UTA", date, weights, model,
                                 avail_model=avail_model,
                                 avail={"av_min": -0.7, "av_bpm": -6.0})
    assert hurt["p_home"] < v2["p_home"]


def test_v4_phase_term_and_tags(model, weights):
    from conftest import make_log
    logs = {"GSW": make_log(40, start="2026-10-21", seed=1, strength=1.0),
            "UTA": make_log(40, start="2026-10-22", seed=2, strength=-1.0)}
    v4 = {"features": ["delta", "b2b_net", "d_phase"],
          "intercept": model["intercept"], "coef": model["coef"] + [0.0]}
    v4_avail = {"features": pav.FEATURES_V4, "intercept": model["intercept"],
                "coef": model["coef"] + [0.0, 0.1, 0.0]}
    date = "2027-01-10"
    base = build_site.score_game(logs, "GSW", "UTA", date, weights, model)
    zero = build_site.score_game(logs, "GSW", "UTA", date, weights, v4)
    assert zero["model_tag"] == build_site.MODEL_TAG_V4
    assert abs(zero["p_home"] - base["p_home"]) < 1e-9       # e = 0: v2 math
    steep = dict(v4, coef=model["coef"] + [0.03])
    early_season = build_site.score_game(logs, "GSW", "UTA", date, weights, steep,
                                         opening="2027-01-09")
    late_season = build_site.score_game(logs, "GSW", "UTA", date, weights, steep,
                                        opening="2026-06-01")
    # same delta; the later in the season, the further from 50%
    assert abs(late_season["p_home"] - 0.5) > abs(early_season["p_home"] - 0.5)
    rep = build_site.score_game(logs, "GSW", "UTA", date, weights, v4,
                                avail_model=v4_avail,
                                avail={"av_min": 0.0, "av_bpm": 0.0})
    assert rep["model_tag"] == build_site.MODEL_TAG_V4_AVAIL
    assert build_site.active_tags(v4, v4_avail) == [build_site.MODEL_TAG_V4,
                                                    build_site.MODEL_TAG_V4_AVAIL]
    assert build_site.active_tags(model, None) == [build_site.MODEL_TAG]
