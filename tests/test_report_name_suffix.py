"""Report-name suffix fix (CLAUDE.md, registered 2026-11-13).

Since 2023-24 the NBA injury report prints suffixes joined to the last name
('MurphyIII,Trey'). The margin model's report route matches them once the
fitted file records the fix; without it, and for the totals model, matching
is unchanged.
"""
from datetime import datetime
import json

import pandas as pd
import pytest

import build_site
import player_availability as pav


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
             q_rates={"300": {"Questionable": 0.481}},
             q_rates_sfx={"300": {"Questionable": 0.47}})


def test_route_q_rates_and_flag():
    old = {k: v for k, v in FIXED.items() if k not in ("report_names", "q_rates_sfx")}
    assert pav.splits_suffix(FIXED) and not pav.splits_suffix(old)
    assert not pav.splits_suffix(None)
    assert pav.route_q_rates(FIXED) == FIXED["q_rates_sfx"]
    assert pav.route_q_rates(old) == old["q_rates"]
    assert pav.route_q_rates(None) is None
    # the per-slate refit keeps the flag
    assert pav.splits_suffix(pav.refit(FIXED, None))


def test_fixed_route_tags_continue_the_v11_family():
    base = {k: v for k, v in FIXED.items() if k not in ("report_names", "q_rates_sfx")}
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


def test_totals_keep_the_old_matcher_and_table():
    """ou3 is out of scope: the build hands totals_terms model["q_rates"]
    (the old table) and totals_terms never splits suffixes."""
    import inspect
    src = inspect.getsource(build_site.slate_rows if hasattr(build_site, "slate_rows")
                            else build_site)
    assert 'q_rates=(avail_model or {}).get("q_rates"))' in src
    assert "split" not in inspect.signature(pav.totals_terms).parameters
