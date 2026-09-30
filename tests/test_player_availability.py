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


def _report(rows):
    return pd.DataFrame(rows, columns=["game_date", "matchup", "team", "player",
                                       "status"])


def test_report_coverage_needs_the_game_and_every_team_filed():
    rep = _report([
        ("11/05/2026", "NYK@BOS", "Boston Celtics", "Brown, Jaylen", "Out"),
        ("11/05/2026", "PHX@BKN", "Phoenix Suns", "", pav.NOT_SUBMITTED),
        ("11/05/2026", "PHX@BKN", "Brooklyn Nets", "Claxton, Nic", "Questionable"),
        ("11/06/2026", "LAL@DEN", "Denver Nuggets", "Braun, Christian", "Out")])
    cov = pav.report_coverage(rep, "2026-11-05")
    assert pav.matchup_teams("PHX@BKN") == ("PHO", "BRK")
    # NYK has nobody listed, but the matchup is on the report: covered
    assert pav.covers(cov, "BOS", "NYK")
    assert not pav.covers(cov, "BRK", "PHO")          # Phoenix not yet submitted
    assert not pav.covers(cov, "DEN", "LAL")          # another slate
    assert not pav.covers(cov, "MIA", "ORL")          # not on the report
    empty = pav.report_coverage(_report([]), "2026-11-05")
    assert not pav.covers(empty, "BOS", "NYK")
    assert not pav.covers(pav.report_coverage(None, "2026-11-05"), "BOS", "NYK")
    # a not-yet-submitted row is coverage, never a player
    box = box_rows("PHO", [{"devin booker": 34}])
    assert pav.od_present(rep, box, "g", {"PHO"}, "2026-11-05") == {}


def _live(report, box):
    live = pav.LiveAvailability.__new__(pav.LiveAvailability)
    live.box, live.value = box, {"jaylen brown": 8.0, "wing": 2.0}
    live.report, live.report_time = report, datetime(2026, 11, 25, 17, 0)
    return live


def test_live_terms_fall_back_when_the_report_does_not_cover_the_game():
    games = [{"jaylen brown": 36, "wing": 30}] * 12
    box = pd.concat([box_rows("BOS", games, opp="NYK"),
                     box_rows("NYK", [{"jalen brunson": 36, "hart": 30}] * 12,
                              opp="BOS").assign(home=False)])
    date = "2026-11-25"
    listed = _report([("11/25/2026", "NYK@BOS", "Boston Celtics", "Brown, Jaylen", "Out")])
    got = _live(listed, box).terms("g", "BOS", "NYK", date)
    assert got is not None and got["av_bpm"] < 0      # BOS star out
    # empty or unparsed report, other slate, team not yet submitted: base model
    assert _live(_report([]), box).terms("g", "BOS", "NYK", date) is None
    other = _report([("11/24/2026", "NYK@BOS", "Boston Celtics", "Brown, Jaylen", "Out")])
    assert _live(other, box).terms("g", "BOS", "NYK", date) is None
    nys = _report([("11/25/2026", "NYK@BOS", "Boston Celtics", "Brown, Jaylen", "Out"),
                   ("11/25/2026", "NYK@BOS", "New York Knicks", "", pav.NOT_SUBMITTED)])
    assert _live(nys, box).terms("g", "BOS", "NYK", date) is None


def test_historical_statuses_mark_only_covered_games(monkeypatch):
    def game(gid, date, home, away):
        return [dict(game_id=gid, date=pd.Timestamp(date), team=t, opp=o, home=h,
                     margin=0.0, player_id=f"{t.lower()} guy", name=f"{t.title()} Guy",
                     minutes=30.0, pm=0.0)
                for t, o, h in ((home, away, True), (away, home, False))]
    box = pd.DataFrame(game("g1", "2026-11-25", "BOS", "NYK")
                       + game("g2", "2026-11-25", "MIA", "ORL"))
    tips = {"g1": "2026-11-26T00:30:00Z", "g2": "2026-11-26T00:30:00Z"}

    class Archive:
        def latest_before(self, cutoff):
            return datetime(2026, 11, 25, 17, 0), "r.pdf"
    rep = _report([("11/25/2026", "NYK@BOS", "Boston Celtics", "Guy, Bos", "Out")])
    monkeypatch.setattr(pav, "parse_report", lambda path: rep)
    st, s = pav.game_statuses(box, tips, Archive(), 30)
    assert s["covered"] == {"g1"} and s["with_report"] == 2
    assert list(st["player_id"]) == ["bos guy"]
