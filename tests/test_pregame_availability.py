import importlib.util
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research"))
_spec = importlib.util.spec_from_file_location(
    "pregame_availability",
    os.path.join(os.path.dirname(__file__), "..", "research", "pregame_availability.py"))
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)
import availability as av  # noqa: E402


def W(text, x0, top):
    return dict(text=text, x0=x0, top=top)


HEADER = [W("Game", 10, 50), W("Date", 30, 50), W("Game", 80, 50), W("Time", 100, 50),
          W("Matchup", 150, 50), W("Team", 220, 50), W("Player", 330, 50),
          W("Name", 360, 50), W("Current", 450, 50), W("Status", 490, 50),
          W("Reason", 540, 50)]


def test_rows_from_words_carries_game_and_team_down():
    page = HEADER + [
        W("Injury", 10, 20), W("Report:", 40, 20),                       # page title
        W("10/22/2024", 10, 70), W("07:30", 80, 70), W("(ET)", 110, 70),
        W("NYK@BOS", 150, 70), W("Boston", 220, 70), W("Celtics", 260, 70),
        W("Porzingis,", 330, 70), W("Kristaps", 380, 70), W("Out", 450, 70),
        W("Injury/Illness", 540, 70),
        W("Brown,", 330, 85), W("Jaylen", 370, 85), W("Questionable", 450, 85),
        W("New", 220, 100), W("York", 245, 100), W("Knicks", 270, 100),
        W("NOT", 330, 100), W("YET", 350, 100), W("SUBMITTED", 370, 100)]
    page2 = [W("Robinson,", 330, 70), W("Mitchell", 380, 70), W("Doubtful", 450, 70)]
    rows = pa.rows_from_words([page, page2])
    assert [(r["team"], r["player"], r["status"]) for r in rows] == [
        ("Boston Celtics", "Porzingis, Kristaps", "Out"),
        ("Boston Celtics", "Brown, Jaylen", "Questionable"),
        ("New York Knicks", "Robinson, Mitchell", "Doubtful")]
    assert all(r["game_date"] == "10/22/2024" for r in rows)


def test_names_teams_dates():
    assert pa.report_name_to_first_last("Porter Jr., Michael") == "Michael Porter Jr."
    assert av.norm_name(pa.report_name_to_first_last("Porter Jr., Michael")) == \
        av.norm_name("Michael Porter Jr.")
    assert pa.team_code("LA Clippers") == "LAC" and pa.team_code("Portland Trail Blazers") == "POR"
    assert pa.team_code("Philadelphia 76ers") == "PHI" and pa.team_code("Nowhere") is None
    assert pa.report_date("10/22/2024") == "2024-10-22" == pa.report_date("10/22/24")
    assert pa.report_date("x") is None
    assert pa.team_code("Oklahoma City") == "OKC" and pa.team_code("Thunder") == "OKC"
    assert pa.team_code("Los Angeles") == "LAL"   # the report says "LA Clippers"
    assert pa.team_code("LA") == "LAC" and pa.team_code("New") is None   # New York / Orleans
    assert pa.team_code("Los Angeles Lakers") == "LAL"


def test_slot_names_and_latest_before(tmp_path):
    assert pa.slot_names(datetime(2025, 1, 5, 17, 0)) == ["05PM", "05_00PM"]
    assert pa.slot_names(datetime(2025, 1, 5, 13, 30)) == ["01_30PM"]
    seen = []

    def fetch(url):
        seen.append(url)
        return b"%PDF-1.4 fake" if url.endswith("2025-01-05_05PM.pdf") else None
    arc = pa.ReportArchive(cache_dir=str(tmp_path), fetch=fetch, sleep=0)
    dt, path = arc.latest_before(datetime(2025, 1, 5, 17, 40))
    assert dt == datetime(2025, 1, 5, 17, 0) and path.endswith("2025-01-05_05PM.pdf")
    n = len(seen)
    arc.save()
    arc2 = pa.ReportArchive(cache_dir=str(tmp_path), fetch=fetch, sleep=0)
    assert arc2.latest_before(datetime(2025, 1, 5, 17, 40))[0] == dt
    assert len(seen) == n                         # cached hit, misses remembered


def test_present_map_arms():
    st = pd.DataFrame(dict(game_id=["g"] * 5, team=["T"] * 5,
                           player_id=list("abcde"),
                           status=["Out", "Doubtful", "Questionable", "Probable",
                                   "Available"], played=[False] * 5, report=[""] * 5))
    od = pa.present_map(st, "od")
    assert [od[("g", p)] for p in "abcde"] == [0, 0, 1, 1, 1]
    q = pa.present_map(st, "q", {"Doubtful": 0.2, "Questionable": 0.6, "Probable": 0.9})
    assert [q[("g", p)] for p in "abcde"] == [0, 0.2, 0.6, 0.9, 1]
    assert pa.play_rates(st.assign(played=[False, False, True, True, True])) == {
        "Available": 1.0, "Doubtful": 0.0, "Out": 0.0, "Probable": 1.0,
        "Questionable": 1.0}


def test_pregame_mode_uses_report_and_previous_roster():
    def rows(k_minutes):
        out = []
        for k, mins in enumerate(k_minutes):
            for pid, m in mins.items():
                out.append(dict(game_id=f"G{k:02d}", date=pd.Timestamp("2025-11-01")
                                + pd.Timedelta(days=2 * k), team="T", opp="O",
                                home=True, margin=0, player_id=pid, name=pid,
                                minutes=m, pm=0.0))
        return pd.DataFrame(out)
    games = [{"star": 36, "traded": 30, "bench": 12}] * 10 + \
            [{"star": 36, "bench": 12}] + [{"star": 0, "bench": 12}]
    df = rows(games)
    val = {"star": 10.0, "traded": 5.0, "bench": 0.0}
    # G11: star listed Out; 'traded' left after G09 (absent from G10's box) -> 0
    got = av.team_availability(df, value=val, present={("G11", "star"): 0.0})
    hind = av.team_availability(df, value=val)
    assert abs(got["G11"][2] - hind["G11"][2]) < 1e-9     # report says what happened
    # star listed Questionable at 0.6 instead
    q = av.team_availability(df, value=val, present={("G11", "star"): 0.6})
    assert q["G11"][2] > got["G11"][2]
    # not listed and on G10's box -> assumed to play (1), unlike hindsight
    none = av.team_availability(df, value=val, present={})
    assert none["G11"][2] > hind["G11"][2]


def test_merged_header_words_and_status_merged_with_reason():
    header = [W("GameDate", 10, 50), W("GameTime", 80, 50), W("Matchup", 150, 50),
              W("Team", 220, 50), W("PlayerName", 330, 50),
              W("CurrentStatus", 450, 50), W("Reason", 540, 50)]
    page = header + [
        W("10/22/2024", 10, 70), W("07:30(ET)", 80, 70), W("NYK@BOS", 150, 70),
        W("BostonCeltics", 220, 70), W("Porzingis,Kristaps", 330, 70),
        W("OutInjury/Illness", 450, 70)]
    rows = pa.rows_from_words([page])
    assert rows == [dict(game_date="10/22/2024", matchup="NYK@BOS",
                         team="BostonCeltics", player="Porzingis,Kristaps",
                         status="Out")]
    assert pa.team_code("BostonCeltics") == "BOS"
    assert pa.report_name_to_first_last("Porzingis,Kristaps") == "Kristaps Porzingis"
    assert pa.status_of("Questionable") == "Questionable" and pa.status_of("GLeague") == ""


def test_text_fallback_spaced_and_unspaced():
    text = "\n".join([
        "Injury Report: 10/22/24 05:30 PM",
        "Game Date Game Time Matchup Team Player Name Current Status Reason",
        "10/22/2024 07:30 (ET) NYK@BOS Boston Celtics Porzingis, Kristaps Out Injury/Illness - Left Leg",
        "Brown, Jaylen Questionable Injury/Illness - Hip",
        "New York Knicks NOT YET SUBMITTED",
        "10/22/2024 10:00(ET) MIN@LAL LosAngelesLakers Vincent,Gabe DoubtfulInjury/Illness-Knee",
        "Porter Jr., Michael Available",
    ])
    rows = pa.rows_from_text([text])
    got = [(pa.team_code(r["team"]), r["player"], r["status"], r["game_date"]) for r in rows]
    assert got == [("BOS", "Porzingis, Kristaps", "Out", "10/22/2024"),
                   ("BOS", "Brown, Jaylen", "Questionable", "10/22/2024"),
                   ("LAL", "Vincent,Gabe", "Doubtful", "10/22/2024"),
                   ("LAL", "Porter Jr., Michael", "Available", "10/22/2024")]


def test_status_word_left_of_its_header_still_reads():
    # 'Current Status' header at x=450; long status words start at 430 (centred
    # column), so they fall in the player cell; 'Out' starts inside the column.
    page = HEADER + [
        W("10/22/2024", 10, 70), W("NYK@BOS", 150, 70), W("Boston", 220, 70),
        W("Celtics", 260, 70), W("Brown,", 330, 70), W("Jaylen", 370, 70),
        W("Questionable", 430, 70), W("Injury/Illness", 540, 70),
        W("Porzingis,", 330, 85), W("Kristaps", 380, 85), W("Out", 462, 85),
        W("Holiday,", 330, 100), W("Jrue", 372, 100), W("Doubtful", 438, 100)]
    rows = pa.rows_from_words([page])
    assert [(r["player"], r["status"]) for r in rows] == [
        ("Brown, Jaylen", "Questionable"), ("Porzingis, Kristaps", "Out"),
        ("Holiday, Jrue", "Doubtful")]
