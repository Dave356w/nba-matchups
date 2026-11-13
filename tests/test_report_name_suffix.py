"""Report-name suffix fix (CLAUDE.md, registered 2026-11-13, and its ou3
amendment).

Since 2023-24 the NBA injury report prints suffixes joined to the last name
('MurphyIII,Trey'). The margin model's report route and the totals model
(ou3) match them once their fitted files record the fix; without it matching
is unchanged.
"""
from datetime import datetime
import json

import pandas as pd
import pytest

import build_site
import player_availability as pav
import totals


@pytest.mark.parametrize("raw, want", [
    ("MurphyIII,Trey", "Trey Murphy III"),
    ("OubreJr.,Kelly", "Kelly Oubre Jr."),
    ("ShannonJr,Terrence", "Terrence Shannon Jr"),
    ("LivelyII,Dereck", "Dereck Lively II"),
    ("JacksonJr.,Jaren", "Jaren Jackson Jr."),
    ("SmithSr.,Joe", "Joe Smith Sr."),
    ("JonesIV,Bob", "Bob Jones IV"),
    ("Porter Jr., Michael", "Michael Porter Jr."),      # 2022-23 spacing: unchanged
    ("Porzingis,Kristaps", "Kristaps Porzingis"),
    ("Ii,Foo", "Foo Ii"),                              # a name, not a suffix
    ("DiVincenzo,Donte", "Donte DiVincenzo"),
])
def test_split_matches_joined_suffixes_only(raw, want):
    assert pav.report_name_to_first_last(raw, split=True) == want


def test_default_matching_is_unchanged():
    assert pav.report_name_to_first_last("MurphyIII,Trey") == "Trey MurphyIII"
    assert pav.report_name_to_first_last("Porter Jr., Michael") == "Michael Porter Jr."
    # the joined form now normalises like the box score's name
    assert pav.norm_name(pav.report_name_to_first_last("MurphyIII,Trey", True)) == \
        pav.norm_name("Trey Murphy III")
    assert pav.norm_name(pav.report_name_to_first_last("MurphyIII,Trey")) != \
        pav.norm_name("Trey Murphy III")


def _report(players, statuses, team="New Orleans Pelicans", matchup="MIN@NOP"):
    return pd.DataFrame(dict(game_date="11/06/2026", matchup=matchup, team=team,
                             player=players, status=statuses))


BOX = pd.DataFrame(dict(game_id="g0", team="NOP", player_id=["1", "2"],
                        name=["Trey Murphy III", "Zion Williamson"]))


def test_od_present_reads_a_joined_suffix_only_with_the_fix():
    rep = _report(["MurphyIII,Trey", "Williamson,Zion"], ["Out", "Questionable"])
    old = pav.od_present(rep, BOX, "g1", {"NOP", "MIN"}, "2026-11-06")
    new = pav.od_present(rep, BOX, "g1", {"NOP", "MIN"}, "2026-11-06", split=True)
    assert old == {("g1", "2"): 1.0}                    # Murphy missed: counted as playing
    assert new == {("g1", "1"): 0.0, ("g1", "2"): 1.0}
    q = pav.od_present(rep, BOX, "g1", {"NOP", "MIN"}, "2026-11-06",
                       {"Questionable": 0.48}, split=True)
    assert q == {("g1", "1"): 0.0, ("g1", "2"): 0.48}


def test_game_statuses_match_joined_suffixes_with_the_fix():
    box = pd.DataFrame(dict(
        game_id="g1", date=pd.Timestamp("2026-11-06"), team=["NOP", "NOP", "MIN"],
        opp=["MIN", "MIN", "NOP"], home=[True, True, False], margin=0.0,
        player_id=["1", "2", "3"], name=["Trey Murphy III", "Zion Williamson", "Anthony Edwards"],
        minutes=[0.0, 30.0, 36.0], pm=0.0))
    rep = _report(["MurphyIII,Trey", "Williamson,Zion"], ["Out", "Probable"])

    class Archive:
        def latest_before(self, cutoff):
            return datetime(2026, 11, 6, 14, 30), "r.pdf"
    orig = pav.parse_report
    try:
        pav.parse_report = lambda path: rep
        tips = {"g1": "2026-11-07T00:00:00Z"}
        st0, s0 = pav.game_statuses(box, tips, Archive(), 30)
        st1, s1 = pav.game_statuses(box, tips, Archive(), 30, split=True)
    finally:
        pav.parse_report = orig
    assert (s0["matched"], s1["matched"]) == (1, 2)
    assert st1.set_index("player_id").loc["1", "status"] == "Out"
    assert s0["covered"] == s1["covered"] == {"g1"}


FIXED = dict(version=11, features=["delta"], fade_games=10, sigma=13.0, intercept=0.0,
             coef=[1.0], link="margin", report_names=pav.SPLIT_SUFFIX,
             q_rates={"300": {"Questionable": 0.47}})


def test_flag_survives_the_per_slate_refit():
    old = {k: v for k, v in FIXED.items() if k != "report_names"}
    assert pav.splits_suffix(FIXED) and not pav.splits_suffix(old)
    assert not pav.splits_suffix(None)
    assert pav.splits_suffix(pav.refit(FIXED, None))


def test_fixed_route_tags_continue_the_v11_family():
    base = {k: v for k, v in FIXED.items() if k != "report_names"}
    assert build_site.is_v11(FIXED) and build_site.is_v11(base)
    assert build_site.model_tag(FIXED, avail=True) == build_site.MODEL_TAG_V11_SFX_AVAIL
    assert build_site.neutral_tag(FIXED, True) == build_site.MODEL_TAG_V11_SFX_NEUTRAL_AVAIL
    # base-route tags are unchanged
    assert build_site.model_tag(FIXED) == build_site.MODEL_TAG_V11
    assert build_site.neutral_tag(FIXED, False) == build_site.MODEL_TAG_V11_NEUTRAL
    # an unfixed report file keeps the existing tags
    assert build_site.model_tag(base, avail=True) == build_site.MODEL_TAG_V11_AVAIL
    new = [build_site.MODEL_TAG_V11_SFX_AVAIL, build_site.MODEL_TAG_V11_SFX_NEUTRAL_AVAIL]
    assert all(t.endswith("_v11") and "_sfx_" in t for t in new)
    assert build_site.MODEL_TAG_V11_SFX_AVAIL in build_site.AVAIL_TAGS
    assert build_site.MODEL_TAG_V11_SFX_NEUTRAL_AVAIL in build_site.NEUTRAL_TAGS
    assert build_site.MODEL_TAG_V11_SFX_NEUTRAL_AVAIL in build_site.AVAIL_TAGS
    assert build_site.active_tags(base, FIXED) == [
        build_site.MODEL_TAG_V11, build_site.MODEL_TAG_V11_SFX_AVAIL,
        build_site.MODEL_TAG_V11_NEUTRAL, build_site.MODEL_TAG_V11_SFX_NEUTRAL_AVAIL]
    assert build_site.model_version(build_site.active_tags(base, FIXED)) == "v11"
    for tag, want in ((build_site.MODEL_TAG_V11_SFX_AVAIL, "avail"),
                      (build_site.MODEL_TAG_V11_SFX_NEUTRAL_AVAIL, "neutral")):
        assert build_site.route(dict(p_home=0.6, model_tag=tag), base) == want


def test_live_terms_pass_the_fix_through():
    class Live(pav.LiveAvailability):
        def __init__(self):
            self.report_time = datetime(2026, 11, 19, 14, 30)
            self.report = "r"
            self.box, self.value = None, {}
    seen = {}
    orig = (pav.covers, pav.report_coverage, pav.od_present, pav.upcoming_terms)
    try:
        pav.covers = lambda *a: True
        pav.report_coverage = lambda *a: None
        pav.od_present = lambda *a, split=False: seen.update(split=split) or {}
        pav.upcoming_terms = lambda *a: (0.0, 0.0, 0.0)
        Live().terms("g", "GSW", "UTA", "2026-11-19")
        assert seen["split"] is False
        Live().terms("g", "GSW", "UTA", "2026-11-19", split=True)
        assert seen["split"] is True
    finally:
        pav.covers, pav.report_coverage, pav.od_present, pav.upcoming_terms = orig


def test_q_rate_files_record_their_matching(tmp_path):
    years = [2023, 2024]
    table = {str(x): {"Questionable": 0.5} for x in pav.Q_LEADS}
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    pav.save_q_rates(table, years, str(old))
    pav.save_q_rates(table, years, str(new), split=True)
    assert "report_names" not in json.loads(old.read_text())
    assert json.loads(new.read_text())["report_names"] == pav.SPLIT_SUFFIX
    assert pav.load_q_rates(str(old), years) == table
    assert pav.load_q_rates(str(new), years, split=True) == table
    with pytest.raises(SystemExit):
        pav.load_q_rates(str(old), years, split=True)
    with pytest.raises(SystemExit):
        pav.load_q_rates(str(new), years)


OU3 = dict(tag=totals.TAG, coef={"intercept": 0.0, "rest_tot": 0.9}, sigma_early=19.0,
           sigma_late=18.5, report=dict(coef={"intercept": 0.3, "av_o": 0.9, "av_d": -1.2,
                                              "av_m": 0.36}, sigma_early=18.0, sigma_late=18.5))


def test_ou3_tags_follow_the_report_stage():
    fixed = dict(OU3, report=dict(OU3["report"], report_names=totals.SPLIT_SUFFIX))
    assert totals.model_tags(OU3) == (totals.TAG, totals.TAG_NEUTRAL)
    assert totals.model_tags(fixed) == (totals.TAG_SFX, totals.TAG_NEUTRAL)
    assert totals.TAG_SFX.endswith(totals.TAG_SUFFIX) and "_sfx_" in totals.TAG_SFX
    assert totals.splits_suffix(fixed) and not totals.splits_suffix(OU3)
    assert not totals.splits_suffix(None)
    av = dict(av_o=1.0, av_d=0.5, av_m=0.2)
    t0, s0, tag0 = totals.adjust(OU3, 220.0, 12, av)
    t1, s1, tag1 = totals.adjust(fixed, 220.0, 12, av)
    assert (t0, s0) == (t1, s1) and (tag0, tag1) == (totals.TAG, totals.TAG_SFX)
    # without report terms: scored as before, the unchanged neutral tag
    assert totals.adjust(fixed, 220.0, 12, None) == totals.adjust(OU3, 220.0, 12, None)
    assert totals.adjust(fixed, 220.0, 12, None)[2] == totals.TAG_NEUTRAL


def test_ou3_fit_report_records_the_fix_and_tag():
    rows = pd.DataFrame(dict(lg_total=[220.0] * 4, rest_tot=[0.0] * 4, gp_home=[12] * 4,
                             gp_away=[12] * 4, total=[221.0, 219.0, 230.0, 210.0],
                             av_o=[1.0, -1.0, 2.0, -2.0], av_d=[0.0, 0.5, -0.5, 0.2],
                             av_m=[0.1, 0.0, 0.3, -0.1], season=2026))
    base = {k: v for k, v in OU3.items() if k != "report"}
    old, new = totals.fit_report(base, rows), totals.fit_report(base, rows, split=True)
    assert old["tag"] == totals.TAG and "report_names" not in old["report"]
    assert new["tag"] == totals.TAG_SFX and new["report"]["report_names"] == totals.SPLIT_SUFFIX
    assert new["report"]["coef"] == old["report"]["coef"]       # same rows, same fit
    assert new["coef"] == base["coef"]                          # T* untouched


def test_ou3_terms_cache_key_separates_the_fix():
    q = {"300": {"Questionable": 0.48}}
    assert totals.terms_key(q) == totals.terms_key(q, split=False)
    assert totals.terms_key(q, split=True) != totals.terms_key(q)


def test_ou3_report_terms_compute_with_the_fix(tmp_path):
    seen = []

    def compute(y, lead, q, health=None, split=False):
        seen.append(split)
        return {}
    q = {"300": {"Questionable": 0.48}}
    totals.report_terms([2027], q, path=tmp_path / "t.csv", compute=compute,
                        today="2026-11-13", log=lambda *a: None, split=True)
    totals.report_terms([2027], q, path=tmp_path / "t.csv", compute=compute,
                        today="2026-11-13", log=lambda *a: None)
    assert seen == [True, False]


def test_ou3_live_terms_pass_the_fix_through():
    box = pd.DataFrame(dict(game_id="g0", date=pd.Timestamp("2026-11-05"), team="NOP",
                            opp="MIN", home=True, margin=0.0, player_id=["1", "2"],
                            name=["Trey Murphy III", "Zion Williamson"], minutes=30.0, pm=0.0))
    rep = _report(["MurphyIII,Trey"], ["Out"])
    values = {"OBPM": {"1": 3.0, "2": 4.0}, "DBPM": {"1": 1.0, "2": 0.5}}
    kw = dict(report_rows=rep, report_time=datetime(2026, 11, 6, 14, 0), box=box, gid="g1",
              home="NOP", away="MIN", date="2026-11-06", tip_utc=None, values=values,
              q_rates=None)
    orig = pav.covers
    try:
        pav.covers = lambda *a: True
        upcoming = pav.upcoming_terms
        pav.upcoming_terms = lambda b, team, *a: (upcoming(b, team, *a) if team == "NOP"
                                                 else (0.0, 0.0, 0.0))
        old = pav.totals_terms(**kw)
        new = pav.totals_terms(**kw, split=True)
    finally:
        pav.covers, pav.upcoming_terms = orig, upcoming
    assert old["av_o"] == 0.0 and new["av_o"] < 0.0              # Murphy out only when matched
