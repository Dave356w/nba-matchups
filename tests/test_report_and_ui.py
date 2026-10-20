"""The sibling-aligned reporting: same-row favourite baseline, the plain-text
ledger_report.txt, the Model page and headline tiles, and the build's
commit-after-failed-render rule."""
from pathlib import Path

import numpy as np
import pandas as pd

import analysis
import build_site
import ledger
import market
import report
from test_analysis_and_pages import synth

ROOT = Path(__file__).resolve().parents[1]


def test_fav_baseline_is_the_market_favourite_on_the_same_rows():
    g = ledger.graded(synth(300, 31))
    p = analysis.picks(g, "close")
    q = pd.to_numeric(g["close_q_home"]).to_numpy()
    fav_home = q >= 0.5
    ml = np.where(fav_home, g["close_home_ml"], g["close_away_ml"])
    won = np.where(fav_home, g["home_won"], 1 - g["home_won"])
    want = dict(zip(g["game_id"], (market.unit_profit(m, w) for m, w in zip(ml, won))))
    for rule in ("lean", "value"):
        d = p[p["rule"] == rule]
        assert np.allclose(d["fav_units"], d["game_id"].map(want))
        r = analysis.roi_row("x", d)
        assert abs(r["fav_roi"] - d["fav_units"].mean()) < 1e-12
        assert abs(r["fav_units"] - d["fav_units"].sum()) < 1e-9


def test_fav_baseline_ties_go_home():
    df = synth(4, 32)
    df[["close_home_ml", "close_away_ml"]] = -110
    df["close_q_home"] = 0.5
    df["home_won"] = [1, 0, 1, 0]
    p = analysis.picks(ledger.graded(df), "close")
    lean = p[p["rule"] == "lean"]
    assert list(lean["fav_units"].round(4)) == [0.9091, -1.0, 0.9091, -1.0]


def test_season_rows_split_book_and_season_never_pooled():
    a, b = synth(120, 33, "reconstructed"), synth(80, 34, "reconstructed", book="espnbet")
    b["game_id"] = "e" + b["game_id"]
    b["season"] = 2026
    rows = analysis.season_rows(ledger.graded(pd.concat([a, b], ignore_index=True)))
    keys = {(r["book"], r["season"]): r["n"] for r in rows}
    assert keys == {("dk", 2027): 120, ("espnbet", 2026): 80}
    for r in rows:
        assert r["lean"]["n"] == r["n"] and r["scoring"]["n"] == r["n"]


def test_report_prints_null_and_fav_beside_every_roi_and_keeps_bases_apart():
    nat, rec = synth(150, 35, "native"), synth(200, 36, "reconstructed")
    txt = report.report_text(nat, rec, ledger.empty(), ["tag_v6"])
    for head in ("== NATIVE", "== PRE-REGISTERED HYPOTHESES", "== RECONSTRUCTED",
                 "== PRESEASON"):
        assert txt.count(head) == 1
    assert txt.index("== NATIVE") < txt.index("== RECONSTRUCTED")
    roi_lines = [ln for ln in txt.splitlines() if " ROI " in ln and "n=" in ln]
    assert roi_lines
    for ln in roi_lines:
        assert "(null " in ln and "| fav " in ln
    assert "pregame snapshot" in txt and "HINDSIGHT" in txt
    assert "Model tags: tag_v6" in txt
    # one number, two surfaces: the report's lean ROI is analysis.roi_summary's
    h = analysis.with_close(ledger.graded(rec))
    top = dict(analysis.roi_summary(h, "close"))[analysis.PICK_RULES[0][1]][0]
    assert report.roi_line("All picks", top) in txt


def test_report_is_stamped_from_data_not_the_clock():
    nat = synth(30, 37, "native")
    nat["snapshot_utc"] = "2026-11-19T23:00:00Z"
    a = report.report_text(nat, ledger.empty())
    b = report.report_text(nat, ledger.empty())
    assert a == b and "2026-11-19T23:00:00Z" in a.splitlines()[0]


def test_report_with_no_rows():
    txt = report.report_text(ledger.empty(), ledger.empty(), None)
    assert "No graded native rows" in txt and "No reconstructed rows" in txt
    assert "no bets" in txt                      # hypotheses still listed


def test_pages_publish_model_page_report_and_headline_tiles(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    text = build_site.write_pages(synth(100, 38, "native"),
                                  synth(200, 39, "reconstructed"),
                                  "2026-11-19", model_ok=True)
    assert (tmp_path / "ledger_report.txt").read_text() == text
    model = (tmp_path / "model.html").read_text()
    assert "Flat 1u ROI by season" in model and "Fav ROI" in model
    assert "Routing" in model
    idx = (tmp_path / "index.html").read_text()
    assert "Flat 1u ROI · native lean" in idx and "Flat 1u ROI · rebuilt history" in idx
    assert "market fav" in idx
    grades = (tmp_path / "grades.html").read_text()
    assert "Fav ROI" in grades
    for name in ("index.html", "grades.html", "model.html", "preseason.html",
                 "market-calibration.html"):
        html = (tmp_path / name).read_text()
        assert "class='brand'" in html and "toggleTheme" in html
        assert "href='model.html'" in html and "href='ledger_report.txt'" in html
        assert html.count("aria-current='page'") == 1


def test_headline_tiles_without_native_rows_say_so():
    t = build_site.headline_tiles(ledger.empty(), synth(100, 40, "reconstructed"))
    assert "no graded bets yet" in t and "rebuilt history" in t


def test_build_commits_data_even_after_a_failed_render():
    wf = (ROOT / ".github/workflows/build.yml").read_text()
    step = wf[wf.index("- name: Validate and commit ledger"):]
    step = step[:step.index("- uses: actions/upload-pages-artifact")]
    assert "if: ${{ !cancelled() }}" in step
    # validation gates the commit inside the same step
    assert step.index("python validate_data_files.py") < step.index("git commit")
    assert "cancel-in-progress: false" in wf


def _slate(n=3, seed=41):
    """Pending native rows on one slate (synthetic rows, results cleared)."""
    d = synth(n, seed, "native").copy()
    d["slate_date"] = "2026-11-19"
    for c in ("home_won", "home_pts", "away_pts"):
        d[c] = np.nan
    return d


def test_index_renders_one_card_per_game_with_ev_beside_its_null(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    d = _slate()
    build_site.write_pages(d, synth(100, 42, "reconstructed"), "2026-11-19",
                           model_ok=True)
    idx = (tmp_path / "index.html").read_text()
    assert idx.count("<details class='card' data-gid=") == len(d)
    for _, r in d.iterrows():
        assert f"{r['away']} @ {r['home']}" in idx
        assert f"Lean {r['lean']}" in idx
        assert f"data-gid='{r['game_id']}'" in idx           # live scores match on it
    # scores are fetched in the reader's browser for the slate's date only
    assert "data-slate='2026-11-19'" in idx and "site.api.espn.com" in idx
    assert idx.count("data-scores-asof hidden") == 1
    assert "Model EV" in idx and "if the market is right" in idx    # EV beside its null
    assert "Model − market" in idx and "How these numbers are made" in idx
    assert "Flat 1u ROI · native lean" in idx                    # record box kept


def test_index_card_abstains_without_a_probability():
    r = _slate(1).iloc[0].copy()
    r["p_home"] = np.nan
    card = build_site.game_card(r)
    assert "Abstain" in card and "Lean " not in card and "probbar" not in card


def test_tip_times_are_pacific_across_dst_and_openers_have_no_games_yet():
    # 7:30 pm ET on 2026-10-21 (PDT, UTC-7) and 2026-11-19 (PST, UTC-8)
    assert build_site.tip_pt("2026-10-21T23:30Z") == "4:30 PM PT"
    assert build_site.tip_pt("2026-11-20T00:30Z") == "4:30 PM PT"
    assert build_site.tip_pt("") == "—"
    r = _slate(1).iloc[0].copy()
    r["tip_utc"] = "2026-11-20T00:30Z"
    r[["p_home", "gp_home", "gp_away"]] = np.nan     # a team not in the logs yet
    card = build_site.game_card(r)
    assert "4:30 PM PT" in card and " ET" not in card
    assert "Abstain · no games yet" in card


def test_rebuilt_history_is_folded_and_forward_rows_open(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(synth(100, 43, "native"), synth(100, 44, "reconstructed"),
                           "2026-11-19", model_ok=True)
    grades = (tmp_path / "grades.html").read_text()
    nat = grades[grades.index("id='forward'"):grades.index("id='rebuilt'")]
    rec = grades[grades.index("id='rebuilt'"):]
    assert "<details class='fold' id='native-" in nat and "' open>" in nat
    assert "<details class='fold' id='reconstructed-" in rec and "' open>" not in rec
    assert "Hindsight, not a track record" in rec
    # the summary line carries ROI with its null and the favourite baseline
    assert "· null " in rec and "· fav " in rec
    assert (tmp_path / ".nojekyll").exists()


def test_nav_drops_report_but_every_footer_links_it(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(ledger.empty(), ledger.empty(), "2026-11-19", model_ok=True)
    for name in ("index.html", "grades.html", "model.html", "preseason.html",
                 "market-calibration.html"):
        html = (tmp_path / name).read_text()
        nav = html[html.index("<nav>"):html.index("</nav>")]
        assert "ledger_report.txt" not in nav
        foot = html[html.index("class='foot'"):]
        assert "href='ledger_report.txt'" in foot


# ------------------------------------------- H5 / H6 picks on the Today cards
def _ats_pending(gp, first_p=0.75, line=-5.5, home_odds=-110):
    """One pending row with a first snapshot (P, sportsbook spread)."""
    r = _slate(1).iloc[0].copy()
    r["gp_home"] = r["gp_away"] = gp
    r["first_snapshot_utc"] = "2026-11-19T15:07:00Z"     # 7:07 AM PST
    r["first_p_home"], r["first_book"] = first_p, "dk"
    r["first_spread"], r["first_home_spread_odds"] = line, home_odds
    r["first_away_spread_odds"] = -110
    r["pre_spread"] = line
    return r


def test_card_shows_the_h6_pick_as_registered():
    # 13.5 * Phi^-1(0.75) = home by 9.1 vs home -5.5: 3.6 pts, home side
    card = build_site.game_card(_ats_pending(gp=3))
    assert "badge hyp'>H6 ATS HOM -5.5<" in card
    assert "<b>H6 pick</b>" in card and "<b>HOM -5.5</b> at -110" in card
    assert "model line HOM -9.1 vs market -5.5: 3.6 pts" in card
    assert "7:07 AM PT" in card and "DraftKings" in card and "H5" not in card
    # games 10+ is H5's, with the same reading
    assert "badge hyp'>H5 ATS HOM -5.5<" in build_site.game_card(_ats_pending(gp=12))


def test_card_says_why_there_is_no_ats_bet():
    # P 0.66 -> home by 5.6 vs -5.5: 0.1 pts, under the 1-pt threshold
    card = build_site.game_card(_ats_pending(gp=3, first_p=0.66))
    assert "badge hyp" not in card
    assert "<b>H6</b> (pre-registered, against the spread): no bet" in card
    assert "0.1 pts; the rule needs 1" in card
    # an alternate-line price is no main-line bet: no H5/H6 line at all
    card = build_site.game_card(_ats_pending(gp=3, home_odds=320))
    assert "H6" not in card
    # no first snapshot yet: nothing to show
    r = _ats_pending(gp=3)
    r[["first_snapshot_utc", "first_p_home", "first_spread"]] = np.nan
    assert "H6" not in build_site.game_card(r)


def test_card_pick_is_the_side_the_ledger_grades():
    for p, line in ((0.75, -5.5), (0.40, -2.5), (0.55, 6.5)):
        r = _ats_pending(gp=3, first_p=p, line=line)
        a = analysis.ats_card(r)
        g = pd.DataFrame([r]).assign(home_pts=110, away_pts=100, home_won=1)
        d = analysis.hypothesis_picks(g, a["hyp"], "first")
        assert a["bet"] == (len(d) == 1)
        if a["bet"]:
            assert d["side"].iloc[0] == a["side"]


def test_card_names_the_spread_book_where_kalshi_is_the_market(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    r = _ats_pending(gp=3)
    r["kalshi_pre_home_ml"], r["kalshi_pre_away_ml"] = -150, 130
    r["kalshi_pre_q_home"] = 0.58
    build_site.write_pages(pd.DataFrame([r]), synth(50, 43, "reconstructed"),
                           "2026-11-19", model_ok=True)
    idx = (tmp_path / "index.html").read_text()
    assert "<b>HOM -5.5</b> at -110 <span class='basis'>DraftKings</span>" in idx
