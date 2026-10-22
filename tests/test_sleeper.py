import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import sleeper as sl


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests supply every payload; an unpatched request fails loudly."""
    def refuse(*a, **k):
        raise AssertionError("network call in a test")
    monkeypatch.setattr(sl, "urlopen", refuse)


def proj_row(pid, date, team, opp, sp=None, first="A", last="Player", gid="g1", **stats):
    line = dict(stats)
    if sp is not None:
        line["sp"] = sp
    return {"player_id": pid, "date": date, "team": team, "opponent": opp,
            "game_id": gid, "stats": line, "company": "rotowire",
            "last_modified": 1766994004730,
            # today's player object: his team now, not the game's
            "player": {"first_name": first, "last_name": last, "team": "LAL",
                       "injury_status": "Out"}}


def test_seasons_and_team_codes():
    assert sl.sleeper_season(2026) == 2025          # 2025-26
    assert [sl.normalize_team(t) for t in ("BKN", "CHA", "PHX", "gsw", " NOP ")] == \
        ["BRK", "CHO", "PHO", "GSW", "NOP"]
    assert sl.normalize_team("DEF") == "" and sl.normalize_team(None) == ""
    assert sl.normalize_team(float("nan")) == ""


def test_projection_frame_minutes_and_listed_but_out():
    raw = [
        proj_row("1", "2025-12-22", "PHX", "BKN", sp=2040.0, pts=17.8, first="Kevin",
                 last="Durant Jr."),
        proj_row("2", "2025-12-22", "PHX", "BKN"),                # listed, empty line
        proj_row("3", "2025-12-22", "PHX", "BKN", sp=0.0),
        proj_row("4", None, "PHX", "BKN", sp=600.0),              # no date: no game
        proj_row("5", "2025-12-22", "XXX", "BKN", sp=600.0),      # not an NBA team
        {"player_id": None, "date": "2025-12-22"},
        "junk",
    ]
    f = sl.projection_frame(raw)
    assert list(f.columns) == sl.PROJECTION_COLUMNS
    assert f["sleeper_id"].tolist() == ["1", "2", "3"]
    one = f.iloc[0]
    assert one["minutes"] == 34.0 and one["projected"] and one["pts"] == 17.8
    assert (one["team"], one["opp"]) == ("PHO", "BRK")       # the row's team, not today's
    assert one["key"] == "kevin durant" and one["company"] == "rotowire"
    assert one["last_modified_utc"] == "2025-12-29T07:40:04Z"
    assert pd.isna(one["reb"])
    assert f.iloc[1]["minutes"] == 0 and not f.iloc[1]["projected"]
    assert f.iloc[2]["minutes"] == 0 and not f.iloc[2]["projected"]
    with pytest.raises(sl.SleeperError):
        sl.projection_frame({"1": {"sp": 1800}})                # the undated v1 shape


def test_projection_frame_keeps_last_duplicate():
    raw = [proj_row("1", "2025-12-22", "BOS", "NYK", sp=1200.0),
           proj_row("1", "2025-12-22", "BOS", "NYK", sp=1800.0)]
    f = sl.projection_frame(raw)
    assert len(f) == 1 and f.iloc[0]["minutes"] == 30.0


def dump():
    return {
        "10": {"player_id": "10", "active": True, "full_name": "Luka Dončić",
               "position": "PG", "fantasy_positions": ["PG"], "team": "LAL",
               "status": "ACT", "depth_chart_position": "PG", "depth_chart_order": 1,
               "injury_status": None, "news_updated": 1792680011656},
        "11": {"player_id": "11", "active": True, "first_name": "Cameron",
               "last_name": "Carr", "position": "PG", "fantasy_positions": ["PG", "SG"],
               "team": "LAL", "depth_chart_position": "SF", "depth_chart_order": 2,
               "injury_status": "DTD", "injury_body_part": "Oblique",
               "injury_notes": "Strain"},
        "12": {"player_id": "12", "active": True, "full_name": "Free Agent",
               "position": "C", "team": None},
        "13": {"player_id": "13", "active": False, "full_name": "Retired",
               "position": "SF", "team": "BOS"},
        "LAL": {"player_id": "LAL", "active": True, "position": "DEF", "team": "LAL"},
        "14": {"player_id": "14", "active": True, "full_name": "Phoenix Guy",
               "position": "SG", "team": "PHX", "depth_chart_position": "G",
               "depth_chart_order": "x"},
    }


def test_players_frame_fields():
    p = sl.players_frame(dump()).set_index("sleeper_id")
    assert list(p.columns) == sl.PLAYER_COLUMNS[1:]
    assert sorted(p.index) == ["10", "11", "12", "14"]       # inactive and DEF dropped
    assert p.loc["10", "key"] == "luka doncic" and p.loc["10", "depth_order"] == 1
    assert p.loc["10", "news_updated_utc"].startswith("2026-")
    assert p.loc["11", "name"] == "Cameron Carr"
    assert p.loc["11", "fantasy_positions"] == "PG/SG"
    assert (p.loc["11", "injury_status"], p.loc["11", "injury_body_part"]) == ("DTD", "Oblique")
    assert p.loc["12", "team"] == "" and pd.isna(p.loc["12", "depth_order"])
    assert p.loc["14", "team"] == "PHO" and p.loc["14", "depth_slot"] == ""
    assert pd.isna(p.loc["14", "depth_order"])


def test_build_reference_ranks_and_shares():
    raw = [proj_row("10", "2026-10-21", "LAL", "GSW", sp=36 * 60.0, first="Luka", last="Doncic"),
           proj_row("11", "2026-10-21", "LAL", "GSW", sp=24 * 60.0, first="Cameron", last="Carr"),
           proj_row("12", "2026-10-21", "LAL", "GSW", sp=24 * 60.0, first="Free", last="Agent"),
           proj_row("99", "2026-10-21", "LAL", "GSW", first="Not", last="Projected"),
           proj_row("20", "2026-10-21", "GSW", "LAL", sp=40 * 60.0, gid="g1")]
    ref = sl.build_reference(sl.players_frame(dump()), sl.projection_frame(raw))
    ref = ref.set_index("sleeper_id")
    lal = ref[ref["team"] == "LAL"]
    assert lal["minutes_rank"].to_dict() == {"10": 1, "11": 2, "12": 3, "99": 4}
    assert abs(lal["minutes_share"].sum() - 1) < 1e-9 and lal.loc["99", "minutes_share"] == 0
    assert (lal["team_minutes"] == 84).all()
    assert ref.loc["10", "starter"] and not ref.loc["11", "starter"]
    assert not ref.loc["99", "starter"] and ref.loc["11", "injury_status"] == "DTD"
    assert ref.loc["20", "minutes_rank"] == 1 and ref.loc["20", "minutes_share"] == 1


def test_match_espn_unique_names_only():
    frame = pd.DataFrame({"key": ["luka doncic", "jalen williams", "jaylin williams",
                                  "traded guy", "nobody"],
                          "team": ["LAL", "OKC", "OKC", "PHO", "BOS"]})
    espn = pd.DataFrame({
        "player_id": ["1", "2", "3", "4", "5", "1"],
        "name": ["Luka Dončić", "Jalen Williams", "Jalen Williams", "Jaylin Williams",
                 "Traded Guy", "Luka Doncic"],
        "team": ["LAL", "OKC", "OKC", "OKC", "WAS", "LAL"]})
    ids = sl.match_espn(frame, espn)
    assert ids.iloc[0] == "1"                   # two spellings, one ESPN id
    assert pd.isna(ids.iloc[1])                 # two ESPN ids share name and team
    assert ids.iloc[2] == "4"
    assert ids.iloc[3] == "5"                   # name unique league-wide: team moved
    assert pd.isna(ids.iloc[4])


def test_pregame_keeps_last_snapshot_before_tip():
    snaps = pd.DataFrame({
        "date": ["2026-10-22"] * 4 + ["2026-10-23"],
        "team": ["PHI", "PHI", "PHI", "CLE", "BOS"],
        "sleeper_id": ["1", "1", "1", "2", "3"],
        "minutes": [30.0, 0.0, 12.0, 25.0, 20.0],
        "fetched_utc": ["2026-10-22T14:00:00Z", "2026-10-22T22:30:00Z",
                        "2026-10-22T23:00:00Z", "2026-10-22T23:05:00Z",
                        "2026-10-22T14:00:00Z"]})
    tips = pd.DataFrame({"slate_date": ["2026-10-22"], "home": ["PHI"], "away": ["CLE"],
                         "tip_utc": ["2026-10-22T23:00Z"]})
    out = sl.pregame(snaps, tips)
    # PHI: the 22:30 snapshot (the one at tip is not before it); CLE: only after
    # tip; BOS: no tip in the ledger.
    assert out[["team", "sleeper_id", "minutes"]].values.tolist() == [["PHI", "1", 0.0]]


def test_fetch_season_walks_weeks_and_caches(tmp_path):
    calls = []

    def fetch(season, week):
        calls.append((season, week))
        if week in (1, 2):
            return []                                   # empty before the season starts
        if week in (3, 4, 6):
            return [proj_row(str(week), f"2025-11-0{week}", "BOS", "NYK", sp=1800.0)]
        return []                                       # week 5 empty, then 7-8 stop it
    out = sl.fetch_season(2026, cache_dir=tmp_path, fetch=fetch)
    assert {s for s, _ in calls} == {2025}
    assert [w for _, w in calls] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert out["week"].tolist() == [3, 4, 6] and out["minutes"].eq(30).all()
    assert (tmp_path / "proj_2026.csv").exists()
    again = sl.fetch_season(2026, cache_dir=tmp_path,
                            fetch=lambda *a: pytest.fail("cache not used"))
    assert again["sleeper_id"].tolist() == ["3", "4", "6"]
    calls.clear()
    sl.fetch_season(2027, cache_dir=tmp_path, fetch=fetch, completed=False)
    assert not (tmp_path / "proj_2027.csv").exists() and calls  # current season: never cached


def test_fetch_players_cache(tmp_path, monkeypatch):
    now = datetime(2026, 10, 22, 16, 0, tzinfo=timezone.utc)
    got = []
    monkeypatch.setattr(sl, "get_json", lambda url, timeout=30: got.append(url) or {"1": {}})
    players, stamp = sl.fetch_players(tmp_path, now=now)
    assert players == {"1": {}} and got == [sl.BASE_URL + "/players/nba"]
    sl.fetch_players(tmp_path, now=now + timedelta(hours=23))
    assert len(got) == 1                                   # fresh cache reused
    sl.fetch_players(tmp_path, now=now + timedelta(hours=25))
    assert len(got) == 2                                   # stale cache refetched
    (tmp_path / "players_nba.json").write_text("{not json")
    sl.fetch_players(tmp_path, now=now + timedelta(hours=25))
    assert len(got) == 3                                   # corrupt cache refetched
    monkeypatch.setattr(sl, "get_json", lambda url, timeout=30: {})
    (tmp_path / "players_nba.json").unlink()
    with pytest.raises(sl.SleeperError):
        sl.fetch_players(tmp_path, now=now)


def test_get_json_retries_then_raises(monkeypatch):
    monkeypatch.setattr(sl, "MIN_CALL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(sl.time, "sleep", lambda s: None)
    tries = []

    def fail(request, timeout):
        tries.append(request.full_url)
        raise sl.URLError("down")
    monkeypatch.setattr(sl, "urlopen", fail)
    with pytest.raises(sl.SleeperError, match="after 3 attempts"):
        sl.get_json(sl.BASE_URL + "/state/nba")
    assert len(tries) == 3


def test_check_against_box():
    raw = [proj_row("1", "2026-10-21", "LAL", "GSW", sp=30 * 60.0, first="Luka", last="Doncic"),
           proj_row("2", "2026-10-21", "LAL", "GSW", sp=10 * 60.0, first="Late", last="Scratch"),
           proj_row("3", "2026-10-21", "LAL", "GSW", first="Ruled", last="Out"),
           proj_row("4", "2026-10-22", "LAL", "SAC", sp=30 * 60.0, first="Luka", last="Doncic")]
    box = pd.DataFrame({"date": ["2026-10-21"] * 3, "team": ["LAL"] * 3,
                        "player_id": ["e1", "e2", "e9"],
                        "name": ["Luka Doncic", "Late Scratch", "Deep Bench"],
                        "minutes": [34.0, 0.0, 6.0]})
    out = sl.check(sl.projection_frame(raw), box)
    assert out["dates"] == 1 and out["box_rows"] == 3 and out["box_rows_listed"] == 2
    assert out["mae_played"] == 4.0 and out["dnp_projected"] == 1 and out["dnp_listed"] == 1
    assert out["unmatched_projected"] == 1                 # ruled out: not on the box
    assert abs(out["played_minutes_listed"] - 34 / 40) < 1e-9
    assert out["team_minutes_mean"] == 40.0


def test_snapshot_stamps_and_keeps_today_on(tmp_path, monkeypatch):
    now = datetime(2026, 10, 22, 16, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(sl, "fetch_state", lambda timeout=30: {"season": "2026", "week": 1})
    monkeypatch.setattr(sl, "fetch_players", lambda cache_dir, timeout, now=None:
                        (dump(), "2026-10-22T15:00:00+00:00"))
    weeks = {1: [proj_row("10", "2026-10-21", "LAL", "GSW", sp=2000.0),
                 proj_row("10", "2026-10-23", "LAL", "SAC", sp=2100.0, gid="g2")],
             2: [proj_row("10", "2026-10-27", "LAL", "POR", sp=2200.0, gid="g3")]}
    monkeypatch.setattr(sl, "fetch_projections", lambda s, w, timeout=60: weeks.get(w, []))
    ref, ctx = sl.snapshot(tmp_path, now=now)
    assert ctx == {"season": 2026, "week": 1, "today": "2026-10-22"}
    assert ref["date"].tolist() == ["2026-10-23", "2026-10-27"]   # played dates dropped
    assert (ref["fetched_utc"] == "2026-10-22T16:00:00Z").all()
    sl.snapshot(tmp_path, now=now + timedelta(hours=1))
    saved = pd.read_csv(tmp_path / "snapshots.csv", dtype={"sleeper_id": str})
    assert len(saved) == 4 and saved["fetched_utc"].nunique() == 2
    assert json.dumps(sorted(saved["date"].unique().tolist())) == '["2026-10-23", "2026-10-27"]'


def test_live_frame_keeps_status_and_snapshot_rows_are_one_game():
    raw = [proj_row("1", "2026-10-22", "PHI", "CLE", sp=2100.0, first="Joel", last="Embiid",
                    pts=20.47, dd=0.31),
           proj_row("2", "2026-10-22", "CLE", "PHI", sp=1620.0, first="Jarrett", last="Allen",
                    gid="g1"),
           proj_row("3", "2026-10-22", "PHI", "CLE", first="Dominick", last="Barlow"),
           proj_row("4", "2026-10-22", "OKC", "DEN", sp=1980.0, gid="g2")]
    raw[1]["player"].update(injury_status="GTD", injury_body_part="Knee")
    raw[2]["player"].update(injury_status="Out", injury_body_part="Back")
    archive = sl.projection_frame(raw)
    assert "injury_status" not in archive                     # never on archived games
    live = sl.projection_frame(raw, live=True)
    assert live.set_index("sleeper_id").loc["2", "injury_status"] == "GTD"
    rows = sl.snapshot_rows(live, "PHI", "CLE")
    assert [(r["team"], r["name"], r["minutes"]) for r in rows] == \
        [("CLE", "Jarrett Allen", "27.0"), ("PHI", "Joel Embiid", "35.0"),
         ("PHI", "Dominick Barlow", "0.0")]
    joel = rows[1]
    assert joel["pts"] == "20.47" and joel["dd"] == "0.31" and joel["reb"] == ""
    assert rows[2]["injury_status"] == "Out" and rows[2]["injury_body_part"] == "Back"
    assert sl.snapshot_rows(live, "BOS", "NYK") == []


def test_live_projections_reads_this_week_and_next_for_the_date(monkeypatch):
    weeks = {3: [proj_row("1", "2026-11-09", "BOS", "NYK", sp=1800.0),
                 proj_row("2", "2026-11-08", "BOS", "MIA", sp=1800.0)],
             4: [proj_row("3", "2026-11-09", "NYK", "BOS", sp=1500.0)]}
    calls = []
    monkeypatch.setattr(sl, "fetch_state", lambda timeout=30: {"season": "2026", "week": 3})
    monkeypatch.setattr(sl, "fetch_projections",
                        lambda s, w, timeout=60: calls.append((s, w)) or weeks.get(w, []))
    got = sl.live_projections("2026-11-09")
    assert calls == [(2026, 3), (2026, 4)]
    assert sorted(got["sleeper_id"]) == ["1", "3"] and "injury_status" in got
