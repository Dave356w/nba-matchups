"""Model v8: one faded rating and one pair of logits for every game, game 0
included (preseason rosters), with H1 / H6's carryover P kept for games 1-9."""
import numpy as np
import pandas as pd
import pytest

import build_site
from research import legacy_cold_start as cold_start
import nba_composite as nc
import player_availability as pav
from conftest import make_log

FEATS = nc.V6_FEATURES
BASE = {"features": FEATS, "intercept": 0.3, "coef": [0.016, 0.3, 0.02, 0.013, 0.06, 0.012],
        "half_life": 25.0, "fade_rho": 0.25, "fade_games": 10}
AVAIL = {"features": pav.FEATURES_V6, "intercept": 0.25,
         "coef": [0.02, 0.33, 0.09, 0.05, 0.016, 0.02, 0.048, 0.007],
         "half_life": 25.0, "fade_rho": 0.25, "fade_games": 10}
CARRY = {"features": ["delta", "b2b_net"], "intercept": 0.39, "coef": [0.035, 0.31],
         "rho": 0.25, "half_life": 25.0}
V4 = {"features": nc.PHASE_FEATURES, "intercept": 0.29, "coef": [0.023, 0.31, 0.027]}


def with_3pa(log, seed):
    rng = np.random.default_rng(seed)
    log = log.copy()
    log["T3PA"] = log["T3P"] + rng.integers(18, 24, len(log))
    log["O3PA"] = log["O3P"] + rng.integers(18, 24, len(log))
    return log


def teams(n_cur):
    cur = {"A": with_3pa(make_log(n_cur, seed=10, strength=1.0), 1),
           "B": with_3pa(make_log(n_cur, start="2026-10-22", seed=11, strength=-1.0), 2)}
    prior = {"A": with_3pa(make_log(82, start="2025-10-21", seed=12, strength=1.5), 3),
             "B": with_3pa(make_log(82, start="2025-10-22", seed=13, strength=-1.5), 4)}
    return cur, prior


def talent(tm, date):
    return {"A": 3.0, "B": -1.0}[tm]


def score(cur, prior, date, weights, **kw):
    args = dict(prior_logs=prior, talent=talent, talent0={"A": 2.0, "B": 0.5},
                carry=CARRY, fallback=V4)
    args.update(kw)
    return build_site.score_game(cur, "A", "B", date, weights, BASE, **args)


def test_game_zero_is_scored_from_last_season_and_the_preseason_roster(weights):
    cur, prior = teams(5)
    r = score(cur, prior, "2026-10-21", weights)
    assert r["gp_home"] == 0 and r["model_tag"] == build_site.MODEL_TAG_V8
    assert np.isfinite(r["p_home"]) and "carry_p_home" not in r     # H1 / H6: games 1-9
    f = lambda lg: nc.features_from_totals(np.array(              # noqa: E731
        [nc.fade_totals(cur[lg], 0, prior[lg], 0.25)[c] for c in nc.COLS]))
    assert abs(r["delta"] - nc.composite(f("A") - f("B"), weights["sd"], weights["w"])) < 1e-3
    r2 = score(cur, prior, "2026-10-21", weights, talent0={"A": 5.0, "B": 0.5})
    assert r2["p_home"] > r["p_home"]                             # the roster counts
    nobody = score({}, prior, "2026-10-20", weights)              # opening night: no logs yet
    assert nobody["gp_home"] == 0 and np.isfinite(nobody["p_home"])


def test_rating_fades_out_by_game_ten(weights):
    cur, prior = teams(30)
    date = "2027-02-01"
    r = score(cur, prior, date, weights)
    ih = int((cur["A"]["date"] < pd.Timestamp(date)).sum())
    ia = int((cur["B"]["date"] < pd.Timestamp(date)).sum())
    d = nc.composite(nc.decayed_features(cur["A"], ih) - nc.decayed_features(cur["B"], ia),
                     weights["sd"], weights["w"])
    assert min(ih, ia) >= 10 and abs(r["delta"] - d) < 1e-3       # this season only
    assert nc.fade_rho(0) == 0.25 and nc.fade_rho(5) == 0.125 and nc.fade_rho(10) == 0.0


def test_availability_from_game_one_and_carry_p_for_games_one_to_nine(weights):
    cur, prior = teams(8)
    av = {"av_min": -0.4, "av_bpm": -1.0}
    g0 = score(cur, prior, "2026-10-21", weights, avail_model=AVAIL, avail=av)
    assert g0["model_tag"] == build_site.MODEL_TAG_V8              # game 0: base
    date = "2026-11-02"
    g = score(cur, prior, date, weights, avail_model=AVAIL, avail=av)
    assert 1 <= min(g["gp_home"], g["gp_away"]) < 10
    assert g["model_tag"] == build_site.MODEL_TAG_V8_AVAIL
    assert "carry_p_home" not in g





def test_v8_uses_only_games_before_the_slate_date(weights):
    cur, prior = teams(12)
    date = cur["A"]["date"].iloc[6]
    r1 = score(cur, prior, date, weights)
    a2 = cur["A"].copy()
    a2.loc[a2["date"] >= date, ["TFG", "OFG", "T3PA"]] = 0
    r2 = score({"A": a2, "B": cur["B"]}, prior, date, weights)
    assert r1["p_home"] == r2["p_home"] and r1["gp_home"] == 6


def test_route_tags_and_version():
    m = BASE
    assert build_site.route({"p_home": 0.6, "model_tag": build_site.MODEL_TAG_V8_AVAIL}, m) == "avail"
    assert build_site.route({"p_home": 0.6, "model_tag": build_site.MODEL_TAG_V8}, m) == "base"
    assert build_site.route({"p_home": 0.6, "model_tag": build_site.MODEL_TAG}, m) == "carry"
    assert build_site.route({"p_home": 0.6, "model_tag": build_site.MODEL_TAG_V4}, m) == "v4"
    assert build_site.route({"p_home": np.nan}, m) == "abstain"
    tags = build_site.active_tags(BASE, AVAIL, V4)
    assert tags[:2] == [build_site.MODEL_TAG_V8, build_site.MODEL_TAG_V8_AVAIL]
    assert build_site.model_version(tags) == "v8"


def test_fade_v8_games_rows_include_game_zero(weights, monkeypatch):
    a = with_3pa(make_log(12, seed=20, strength=1.0), 5)
    b = with_3pa(make_log(12, seed=21, strength=-1.0), 6)
    a["opp"], b["opp"] = "B", "A"
    b["date"], b["home"] = a["date"], ~a["home"]
    prior = {"A": with_3pa(make_log(82, start="2025-10-21", seed=22), 7),
             "B": with_3pa(make_log(82, start="2025-10-21", seed=23), 8)}
    monkeypatch.setattr(nc, "load_logs", lambda y, refresh=False: {"A": a, "B": b})
    g = nc.fade_v8_games(2027, weights, talent=talent, talent0={"A": 2.0, "B": 0.5},
                         prior_logs=prior)
    g0 = g[g[["gp_home", "gp_away"]].min(axis=1) == 0]
    assert len(g0) == 1 and np.isfinite(g0[FEATS].to_numpy(float)).all()
    assert abs(abs(g0["talent_diff"].iloc[0]) - 1.5) < 1e-12        # preseason rosters
    later = g[g[["gp_home", "gp_away"]].min(axis=1) >= 1]
    assert (later["talent_diff"].abs() == 4.0).all()                # talent_fn from game 1


def test_live_preseason_box_is_saved_only_from_a_clean_fetch(tmp_path, monkeypatch):
    rows = pd.DataFrame(dict(game_id=["1"], date=[pd.Timestamp("2026-10-10")], team=["BOS"],
                             player_id=["p"], name=["A B"], minutes=[20.0]))
    monkeypatch.setattr(pav, "preseason_rows", lambda dates, sleep=0.15: (rows, 1))
    pav.update_preseason_box(2027, "2026-10-21", str(tmp_path))
    assert not (tmp_path / "nba_box_pre_2027.csv").exists()          # a request failed
    monkeypatch.setattr(pav, "preseason_rows", lambda dates, sleep=0.15: (rows, 0))
    pav.update_preseason_box(2027, "2026-10-21", str(tmp_path))
    assert (tmp_path / "nba_box_pre_2027.csv").exists()
    monkeypatch.setattr(pav, "preseason_rows",
                        lambda dates, sleep=0.15: pytest.fail("refetched a frozen file"))
    back = pav.update_preseason_box(2027, "2026-10-30", str(tmp_path))
    assert list(back["player_id"]) == ["p"]


def test_validator_flags_a_broken_preseason_box(tmp_path):
    import validate_data_files as v
    df = pd.DataFrame(dict(game_id=["1", "1"], date=pd.to_datetime(["2025-10-10"] * 2),
                           team=["BOS", "NYK"], player_id=["p", "q"], name=["A", "B"],
                           minutes=[20.0, 18.0]))
    path = str(tmp_path / "nba_box_pre_2026.csv")
    pav.save_pre_box(df, path)
    assert v.check_pre_box(path) == []
    pav.save_pre_box(pd.concat([df, df.iloc[:1]]), path)
    assert any("twice in one game" in e for e in v.check_pre_box(path))
    pav.save_pre_box(df.assign(date=pd.to_datetime(["2026-03-01"] * 2)), path)
    assert any("outside the 2025-26 preseason" in e for e in v.check_pre_box(path))


def test_research_preseason_cache_keeps_only_a_clean_fetch(tmp_path, monkeypatch):
    rows = pd.DataFrame(dict(game_id=["1"], date=[pd.Timestamp("2025-10-10")], team=["BOS"],
                             player_id=["p"], name=["A B"], minutes=[20.0]))
    cache, data = tmp_path / "cache", tmp_path / "data"
    monkeypatch.setattr(pav, "preseason_rows", lambda dates, sleep=0.15: (rows, 2))
    got = pav.fetch_preseason_box(2026, "2025-10-21", str(cache), str(data))
    assert len(got) == 1 and pav.pre_box_file(2026, str(cache), str(data)) is None
    monkeypatch.setattr(pav, "preseason_rows", lambda dates, sleep=0.15: (rows, 0))
    pav.fetch_preseason_box(2026, "2025-10-21", str(cache), str(data))
    assert pav.pre_box_file(2026, str(cache), str(data)) == str(cache / "box_pre_2026.csv")


def test_box_history_never_commits_a_partial_preseason_fetch(tmp_path, monkeypatch):
    import importlib.util as iu
    import os
    spec = iu.spec_from_file_location("box_history", os.path.join(
        os.path.dirname(__file__), "..", "research", "box_history.py"))
    bh = iu.module_from_spec(spec)
    spec.loader.exec_module(bh)
    box = pd.DataFrame(dict(game_id=["g1"], date=pd.to_datetime(["2025-10-21"]), team=["BOS"],
                            opp=["NYK"], home=[True], margin=[3], player_id=["p"], name=["A B"],
                            minutes=[30.0], pm=[3.0]))
    rows = pd.DataFrame(dict(game_id=["1"], date=[pd.Timestamp("2025-10-10")], team=["BOS"],
                             player_id=["p"], name=["A B"], minutes=[20.0]))
    monkeypatch.setattr(pav, "preseason_rows", lambda dates, sleep=0.15: (rows, 1))
    data = tmp_path / "data"
    run = lambda: bh.build([2026], str(tmp_path / "cache"), fetch=lambda t, cache_dir=None: box,  # noqa: E731
                           bpm=lambda t: {}, data_dir=str(data), commit=True,
                           today=pd.Timestamp("2026-10-22"))
    out = run()
    assert "preseason incomplete" in out[2026] and not (data / "nba_box_pre_2026.csv").exists()
    monkeypatch.setattr(pav, "preseason_rows", lambda dates, sleep=0.15: (rows, 0))
    out = run()
    assert "preseason 1 games" in out[2026] and (data / "nba_box_pre_2026.csv").exists()


def test_game_zero_rows_enter_no_hypothesis_and_get_their_own_fold():
    # H1 / H6 were registered on games 1-9 and H2-H5 on games 10+; v8's game-0
    # rows belong to neither (their P is not the carryover model's)
    import analysis
    import ledger
    from test_analysis_and_pages import synth
    from test_report_and_ui import _ats_pending
    df = synth(300, 11, basis="reconstructed")
    df["model_tag"] = build_site.MODEL_TAG_V9
    df["p_home"] = df["close_q_home"] + 0.15                 # value side home, edge 0.15
    df["gp_home"] = [0] * 100 + [5] * 100 + [30] * 100
    df["gp_away"] = [4] * 100 + [5] * 100 + [30] * 100
    g = ledger.graded(df)
    ids = lambda lo, hi: set(df["game_id"].iloc[lo:hi])     # noqa: E731
    for hyp in analysis.active_hypotheses():
        if hyp.get("market") == "ats":
            continue
        got = set(analysis.hypothesis_picks(g, hyp, "close")["game_id"])
        assert got and not got & ids(0, 100), hyp["key"]
        assert got <= (ids(100, 200) if hyp["early"] else ids(200, 300)), hyp["key"]
    labels = [r["label"] for r in dict(analysis.roi_summary(g, "close"))[
        analysis.PICK_RULES[0][1]]]
    assert labels == ["All picks", "Games 10+", "Games 1–9", "Game 0"]
    assert analysis.ats_card(_ats_pending(gp=0)) is None             # no spread rule
    assert analysis.ats_card(_ats_pending(gp=3))["hyp"]["key"] == "H5·E"


def test_picks_note_tells_a_tipped_game_from_one_the_next_build_can_score():
    from test_report_and_ui import _slate
    d = _slate(3)
    d["tip_utc"] = ["2026-11-19T23:00Z", "2026-11-20T01:00Z", "2026-11-20T03:00Z"]
    d.loc[:1, "p_home"] = np.nan
    render = lambda now: build_site.render_index(d, "2026-11-19", "b", True, now=now)  # noqa: E731
    html = render(pd.Timestamp("2026-11-20T00:00Z"))       # the first tipped, the second not
    assert ("2 of 3 games have no model probability: 1 tipped without one and stays "
            "without a pick (a row is frozen at tip); for the other, an input") in html
    assert "the next build retries until tip." in html
    assert "have no model probability: they tipped" in render(pd.Timestamp("2026-11-20T02:00Z"))
    assert "tipped" not in render(pd.Timestamp("2026-11-19T12:00Z")).split("flag-note'>")[1][:200]
    # the lean's chip sits beside the name, outside the part that truncates
    assert "<span class='nm'><span class='tn'>" in html
