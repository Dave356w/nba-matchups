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
    # the public record: the lean's ROI beside its null and the market favourite
    assert "<div class='l'>Record</div>" in idx and "<div class='l'>ROI</div>" in idx
    assert "null " in idx and "market favourite" in idx
    grades = (tmp_path / "grades.html").read_text()
    assert "Fav ROI" in grades
    for name, _ in build_site.PAGES:
        html = (tmp_path / name).read_text()
        assert "class='brand'" in html and "toggleTheme" in html
        assert "href='model.html'" in html and "href='ledger_report.txt'" in html
        nav = html[html.index("<nav class='nav'"):]
        nav = nav[:nav.index("</nav>")]
        assert nav.count("aria-current='page'") == 1      # one main tab is current
        research = name in {h for h, _ in build_site.RESEARCH_PAGES}
        assert ("aria-current='page'>Research<" in nav) == research
        assert ("class='subnav'" in html) == research


def test_record_tiles_without_native_rows_say_so():
    assert build_site.public_scope(ledger.empty()) is None
    t = build_site.record_tiles(None)
    assert "No graded picks yet" in t and "null" in t


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
    assert idx.count("<details class='gcard") == idx.count(" data-gid='") == len(d)
    for _, r in d.iterrows():
        assert f"{r['away']} @ {r['home']}" in idx
        assert f"<span class='pk'>{r['lean']} " in idx      # the lean and its price
        assert f"data-gid='{r['game_id']}'" in idx           # live scores match on it
    # scores are fetched in the reader's browser for the slate's date only
    assert "data-slate='2026-11-19'" in idx and "site.api.espn.com" in idx
    assert idx.count("data-scores-asof hidden") == 1
    assert "<div class='l'>EV</div>" in idx and "if the market is right" in idx  # EV beside its null
    assert "<div class='l'>Edge</div>" in idx and "How these numbers are made" in idx
    assert "<div class='l'>Record</div>" in idx                  # record tiles kept
    assert "data-filter" in idx and "data-state='upcoming'" in idx


def test_index_card_abstains_without_a_probability():
    r = _slate(1).iloc[0].copy()
    r["p_home"] = np.nan
    card = build_site.game_card(r)
    assert "No probability" in card and "Lean " not in card and "probbar" not in card


def _injuries(r, home_rows, away_rows, snap="2026-11-19T19:05:00Z"):
    rows = [dict(team=r["home"], **x) for x in home_rows] + \
        [dict(team=r["away"], **x) for x in away_rows]
    return pd.DataFrame([{**{c: "" for c in ledger.INJURY_COLUMNS}, **x,
                          "game_id": str(r["game_id"]), "tip_utc": r["tip_utc"],
                          "snapshot_utc": snap} for x in rows],
                        columns=ledger.INJURY_COLUMNS)


def test_card_opens_to_each_teams_injury_list_as_recorded():
    r = _slate(1).iloc[0].copy()
    inj = _injuries(r, [dict(player_id="1", name="Zed <Ant>", status="Day-To-Day",
                             detail="Ankle Sprain"),
                        dict(player_id="2", name="Al Out", status="Out", detail="Knee"),
                        dict(player_id="3", name="Bo Q", status="Questionable", detail="")],
                    [dict(player_id="", name="", status="NONE", detail="")])
    card = build_site.game_card(r, inj)
    det = card.split("<div class='g-detail'>")[1]
    assert "<h3>Injuries</h3>" in det                      # in the expanded part only
    home = det.split(f"<small>{r['home']}</small>")[1]
    # most serious status first; the reason under the name; names escaped
    assert home.index("Al Out") < home.index("Bo Q") < home.index("Zed &lt;Ant&gt;")
    assert "Al Out<small>Knee</small>" in home and "<span class='badge'>Out</span>" in home
    assert "Bo Q</span>" in home                            # no empty reason line
    away = det.split(f"<small>{r['away']}</small>")[1].split("</ul>")[0]
    assert "None listed" in away
    assert det.index(f"<small>{r['away']}</small>") < det.index(f"<small>{r['home']}</small>")
    assert "as of 11:05 AM PT" in det and "The model does not read it" in det
    # another game's list is not this card's; no frame, no section
    other = inj.assign(game_id="999")
    assert "No injury list was captured" in build_site.game_card(r, other)
    assert "Injuries" not in build_site.game_card(r)
    # a team missing from the snapshot says so, rather than "None listed"
    only_home = inj[inj["team"] == r["home"]]
    assert "Not captured" in build_site.game_card(r, only_home)


def test_index_cards_read_the_injury_file(tmp_path, monkeypatch):
    d = _slate(2)
    inj = _injuries(d.iloc[1], [dict(player_id="7", name="Cy Hurt", status="Out",
                                     detail="Hamstring")], [])
    html = build_site.render_index(d, "2026-11-19", "built", True, injuries=inj)
    assert html.count("<h3>Injuries</h3>") == 2 and html.count("Cy Hurt") == 1
    assert html.count("No injury list was captured") == 1
    # write_pages reads data/nba_injuries.csv through ledger.load_injuries
    seen = []
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    monkeypatch.setattr(build_site.ledger, "load_injuries", lambda: seen.append(1) or inj)
    build_site.write_pages(d, synth(100, 42, "reconstructed"), "2026-11-19", model_ok=True)
    assert seen and "Cy Hurt" in (tmp_path / "index.html").read_text()


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
    assert "No pick: no model probability" in card


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
    for name, _ in build_site.PAGES:
        html = (tmp_path / name).read_text()
        nav = html[html.index("<nav class='nav'"):]
        nav = nav[:nav.index("</nav>")]
        assert "ledger_report.txt" not in nav
        foot = html[html.index("class='foot'"):]
        assert "href='ledger_report.txt'" in foot


# ------------------------------------------- H5 / H5·E picks on the Today cards
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
    assert "Spread pick · H5·E</span><span class='pk'>HOM -5.5</span>" in card
    assert "<b>H5·E pick</b>" in card and "<b>HOM -5.5</b> at -110" in card
    assert "model line HOM -9.1 vs market -5.5: 3.6 pts" in card
    assert "<span>Model HOM -9.1</span>" in card and "Carryover" not in card
    assert "7:07 AM PT" in card and "DraftKings" in card and "H6" not in card
    # games 10+ is H5's, with the same reading, on the model's own P
    card = build_site.game_card(_ats_pending(gp=12))
    assert "Spread pick · H5</span><span class='pk'>HOM -5.5</span>" in card
    assert "model line HOM -9.1 vs market -5.5: 3.6 pts" in card
    assert "<span>Model HOM -9.1</span>" in card and "Carryover" not in card


def test_card_says_why_there_is_no_ats_bet():
    # P 0.66 -> home by 5.6 vs -5.5: 0.1 pts, under the 1-pt threshold
    card = build_site.game_card(_ats_pending(gp=3, first_p=0.66))
    assert "Spread pick" not in card
    assert "<b>H5·E</b> (pre-registered, against the spread): no bet" in card
    assert "0.1 pts; the rule needs 1" in card
    # an alternate-line price is no main-line bet: no H5/H5·E line at all
    card = build_site.game_card(_ats_pending(gp=3, home_odds=320))
    assert "H5·E" not in card
    # no first snapshot yet: nothing to show
    r = _ats_pending(gp=3)
    r[["first_snapshot_utc", "first_p_home", "first_spread"]] = np.nan
    assert "H5·E" not in build_site.game_card(r)


def _with_pre_spread(r, p, line, home_odds=-110, away_odds=-110):
    r = r.copy()
    r["p_home"], r["pre_spread"] = p, line
    r["pre_home_spread_odds"], r["pre_away_spread_odds"] = home_odds, away_odds
    r["lean"] = r["home"] if p >= 0.5 else r["away"]
    return r


def test_every_card_with_a_probability_has_one_spread_line():
    # no rule bets (0.1 pts at the first snapshot): the model's side against
    # the latest spread, shown and labelled as not graded
    r = _with_pre_spread(_ats_pending(gp=3, first_p=0.66), 0.66, -7.5)   # home by 5.6
    card = build_site.game_card(r)
    assert card.count("<div class='pick-lines'><span class='kind'>Lean · spread") == 1
    assert "<span class='pk'>AWY +7.5</span><span class='edge'>1.9 pts vs the line" in card
    assert "Model AWY +5.6</span> · <span>Market AWY +7.5</span>" in card
    assert "shown, not graded" in card and "rule below" in card
    assert "<b>H5·E</b> (pre-registered, against the spread): no bet" in card
    assert "Spread pick" not in card
    # the registered pick when its rule bets, and no lean line beside it
    card = build_site.game_card(_with_pre_spread(_ats_pending(gp=3), 0.75, -5.5))
    assert "Spread pick · H5·E</span>" in card and "Lean · spread" not in card
    # game 0: no rule covers it, the card says so
    r = _with_pre_spread(_ats_pending(gp=3), 0.675, -7.5)
    r["gp_away"] = 0
    card = build_site.game_card(r)
    assert "Lean · spread" in card and "AWY +7.5" in card and "Spread pick" not in card
    assert "neither covers a team's first game" in card
    # no model probability, or an alternate-line price: no spread line
    r = _with_pre_spread(_ats_pending(gp=3), np.nan, -7.5)
    r[["first_p_home", "p_lean", "lean"]] = np.nan
    assert "Lean · spread" not in build_site.game_card(r)
    assert "Lean · spread" not in build_site.game_card(
        _with_pre_spread(_ats_pending(gp=3, first_p=0.66), 0.66, -7.5, home_odds=320))


def test_card_pick_is_the_side_the_ledger_grades():
    for p, line in ((0.75, -5.5), (0.40, -2.5), (0.55, 6.5)):
        r = _ats_pending(gp=3, first_p=p, line=line)
        a = analysis.ats_card(r)
        g = pd.DataFrame([r]).assign(home_pts=110, away_pts=100, home_won=1)
        d = analysis.hypothesis_picks(g, a["hyp"], "first")
        assert a["bet"] == (len(d) == 1)
        if a["bet"]:
            assert d["side"].iloc[0] == a["side"]


def test_card_reads_one_sportsbook_and_no_kalshi(tmp_path, monkeypatch):
    # the card's moneyline and spread are the sportsbook's (owner's decision,
    # 2026-10-20), and no Kalshi price shows even where the row has one
    # (2026-10-21)
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    r = _ats_pending(gp=3)
    r["kalshi_pre_home_ml"], r["kalshi_pre_away_ml"] = -150, 130
    r["kalshi_pre_q_home"] = 0.58
    build_site.write_pages(pd.DataFrame([r]), synth(50, 43, "reconstructed"),
                           "2026-11-19", model_ok=True)
    idx = (tmp_path / "index.html").read_text()
    away, home = build_site.ml_txt(r["pre_away_ml"]), build_site.ml_txt(r["pre_home_ml"])
    assert f"<span class='px'>{home}<small>-5.5</small></span>" in idx
    assert f"<span class='px'>{away}<small>+5.5</small></span>" in idx
    assert "<span class='px'>-150" not in idx and "<span class='px'>+130" not in idx
    assert "DraftKings moneyline and spread" in idx
    q = float(r["pre_q_home"])
    assert f"<span>Market {build_site.pct(q)}</span>" in idx           # the book's no-vig
    assert "<b>HOM -5.5</b> at -110 <span class='basis'>DraftKings</span>" in idx
    # no Kalshi row in the detail, and no Kalshi anywhere on the page
    assert "Kalshi" not in idx and "58.0%" not in idx


# ----------------------------------------- public Results / Performance pages
def test_results_and_performance_show_the_native_lean_record(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    native = synth(120, 52, "native")
    build_site.write_pages(native, synth(200, 53, "reconstructed"), "2026-11-29",
                           model_ok=True)
    scope = build_site.public_scope(native)
    r = scope["row"]
    units = f"{r['units']:+.2f}u"
    res = (tmp_path / "results.html").read_text()
    assert "<h1 class='gr-h1'>Results</h1>" in res and units in res
    assert "Game by game" in res and "<span class='badge w'>W</span>" in res
    assert "Spread picks" in res and "No graded bets yet" in res     # no H5/H5·E rows
    perf = (tmp_path / "performance.html").read_text()
    assert "<div class='uchart'>" in perf and f"<span class='v'>{units}</span>" in perf
    for t in ("Model vs simple baselines", "Market favourite", "Home side", "By week"):
        assert t in perf
    assert f"null {100 * r['roi_null']:+.1f}%" in perf               # ROI beside its null


def test_performance_tracks_model_and_market_forecasts(tmp_path, monkeypatch):
    """Performance scores and calibrates the model and the market on the
    record's games (native, one book x season), never the rebuilt rows."""
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    native = synth(120, 52, "native")
    build_site.write_pages(native, synth(200, 53, "reconstructed"), "2026-11-29",
                           model_ok=True)
    perf = (tmp_path / "performance.html").read_text()
    h = build_site.public_scope(native)["sec"]["h"]
    sc = analysis.scoring(h)
    assert "Forecasts: model vs market" in perf and "Forecasts by week" in perf
    assert (f"{sc['model']['logloss']:.3f} / {sc['market']['logloss']:.3f}" in perf)
    assert (f"{sc['model']['brier']:.3f} / {sc['market']['brier']:.3f}" in perf)
    assert f"{sc['d_logloss']:+.4f}" in perf                 # model − market, ± 1 SE
    for t in ("Calibration slope · model", "Calibration slope · market",
              "Model lean: won − stated", "Market favourite: won − implied",
              "Model vs market calibration", "market-calibration.html",
              "Lean won vs stated", "Favourites won vs implied"):
        assert t in perf, t
    assert "the table below" not in perf
    sl = analysis.calibration_slope(h["p_home"], h["home_won"])
    assert f"{sl[0]:.2f}" in perf
    lc = analysis.lean_calibration(h)
    assert f"won {100 * lc['actual']:.1f}% vs {100 * lc['predicted']:.1f}% stated" in perf


def test_public_pages_never_show_the_rebuilt_seasons(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(ledger.empty(), synth(200, 54, "reconstructed"),
                           "2026-11-29", model_ok=True)
    assert "No graded picks yet" in (tmp_path / "results.html").read_text()
    perf = (tmp_path / "performance.html").read_text()
    assert "starts with the first graded pick" in perf
    assert "Starts with the first graded game with a closing line." in perf
    assert "No graded picks yet" in (tmp_path / "index.html").read_text()


def test_a_graded_h6_bet_reaches_the_spread_tile_and_results(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    r = _ats_pending(gp=3)                       # H5·E: HOM -5.5 at -110
    r["home_pts"], r["away_pts"], r["home_won"] = 110, 100, 1       # covers by 4.5
    build_site.write_pages(pd.DataFrame([r]), synth(50, 55, "reconstructed"),
                           "2026-11-20", model_ok=True)
    res = (tmp_path / "results.html").read_text()
    assert "Spread · H5·E · games 1–9" in res and "<div class='v '>1–0</div>" in res
    assert "H5·E HOM -5.5 <span class='badge w'>W</span> +0.91u" in res


# ------------------------------------------- prices on cards, spread charts
def test_card_rows_show_each_sides_moneyline_and_spread():
    r = _ats_pending(gp=3)                      # home -5.5; DraftKings prices
    card = build_site.game_card(r)
    away, home = build_site.ml_txt(r["pre_away_ml"]), build_site.ml_txt(r["pre_home_ml"])
    assert f"<span class='px'>{away}<small>+5.5</small></span>" in card
    assert f"<span class='px'>{home}<small>-5.5</small></span>" in card
    assert "<span>Model</span><span>Price</span>" in card
    assert "DraftKings moneyline and spread" in card
    assert "Kalshi" not in card                 # no Kalshi price, no Kalshi row
    r["pre_spread"] = np.nan
    assert "DraftKings moneyline · refreshed" in build_site.game_card(r)
    r["pre_spread"] = -5.5
    # an abstaining game still shows its prices
    r[["p_home", "gp_home", "gp_away", "first_p_home"]] = np.nan
    card = build_site.game_card(r)
    assert "No probability" in card and f"<span class='px'>{home}<small>-5.5</small></span>" in card


def test_performance_charts_each_spread_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    r = _ats_pending(gp=3)                       # H5·E bet that covers: +0.91u
    r["home_pts"], r["away_pts"], r["home_won"] = 110, 100, 1
    build_site.write_pages(pd.DataFrame([r]), synth(50, 56, "reconstructed"),
                           "2026-11-20", model_ok=True)
    perf = (tmp_path / "performance.html").read_text()
    assert "Running units · H5·E spread · games 1–9" in perf
    assert "<span class='v'>+0.91u</span>" in perf
    assert "Starts with the first graded H5 bet." in perf


def test_card_edge_and_ev_count_only_the_shrunk_share():
    """The card keeps the model's probability as recorded; its edge and EV
    count only analysis.SHRINK_W of the disagreement with the price."""
    r = pd.Series(dict(
        game_id="401", away="BOS", home="NYK", tip_utc="2026-11-19T00:30Z",
        p_home=0.60, p_lean=0.60, lean="NYK", pre_q_home=0.50,
        pre_home_ml=-110, pre_away_ml=-110, pre_book="dk", delta=5.0,
        gp_home=20, gp_away=20, home_b2b=0, away_b2b=0, model_tag="t",
        home_won=np.nan, home_pts=np.nan, away_pts=np.nan))
    html = build_site.game_card(r)
    assert "Model 60.0%" in html and "Market 50.0%" in html   # P as recorded
    ps = float(analysis.shrink_p(0.60, 0.50))
    assert abs(ps - 0.5399) < 1e-3                           # 40% of the logit gap
    assert "+4.0 pp edge" in html and "+10.0 pp" not in html
    ev = ps * market.decimal_payout(-110) - 1
    assert f"EV {100 * ev:+.1f}%" in html and "EV +14.5%" not in html


def test_shrink_p_keeps_the_side_and_the_ends():
    p, q = np.array([0.7, 0.3, 0.55]), np.array([0.6, 0.4, 0.6])
    s = analysis.shrink_p(p, q)
    assert (np.sign(s - q) == np.sign(p - q)).all() and (abs(s - q) < abs(p - q)).all()
    assert np.allclose(analysis.shrink_p(p, q, 1.0), p)
    assert np.allclose(analysis.shrink_p(p, q, 0.0), q)
    assert np.isnan(analysis.shrink_p(0.6, np.nan))


# ------------------------------------------------------- market performance
def _market_rows():
    """Three graded DraftKings games (the opening night's shape)."""
    base = dict(season=2027, basis="native", close_book="dk", game_id="", slate_date="2026-10-20",
                tip_utc="2026-10-20T23:30Z", model_tag="t", p_home=np.nan, lean=np.nan,
                p_lean=np.nan, gp_home=0, gp_away=0)
    rows = [  # home, away, home ML, away ML, home line, home odds, away odds, pts
        ("DET", "BOS", 105, -125, 1.5, -108, -112, 121, 113),
        ("NYK", "PHI", -175, 145, -4.5, -108, -112, 100, 106),
        ("SAS", "OKC", -102, -118, 1.5, -112, -108, 104, 116)]
    out = []
    for i, (h, a, hml, aml, hs, hso, aso, hp, ap) in enumerate(rows):
        out.append({**base, "game_id": str(401 + i), "home": h, "away": a,
                    "close_home_ml": hml, "close_away_ml": aml,
                    "close_q_home": market.devig(hml, aml), "close_spread": hs,
                    "close_home_spread_odds": hso, "close_away_spread_odds": aso,
                    "home_pts": hp, "away_pts": ap, "home_won": int(hp > ap)})
    e = ledger.empty()
    return pd.concat([e, pd.DataFrame(out)], ignore_index=True)[list(e.columns)]


def test_market_games_reads_prices_and_results_only():
    m = analysis.market_games(_market_rows())
    assert len(m) == 3
    # DET +1.5 vs BOS (-125): favourite BOS lost by 8; home covered by 9.5
    r = m.iloc[0]
    assert not r["fav_home"] and r["fav_won"] == 0 and r["fav_units"] == -1
    assert r["fav_line"] == -1.5 and r["fav_cover"] == 0 and r["home_cover"] == 1
    assert r["line_error"] == 9.5 and r["fav_margin_vs_line"] == -9.5
    # SAS -102 / OKC -118: OKC favourite won by 12 at -118, covered -1.5 at -108
    r = m.iloc[2]
    assert not r["fav_home"] and r["fav_won"] == 1
    assert np.isclose(r["fav_units"], 100 / 118) and np.isclose(r["ats_units"], 100 / 108)
    assert np.isclose(r["fav_null"], r["fav_q"] * market.decimal_payout(-118) - 1)
    s = analysis.market_summary(m)
    assert (s["su"]["w"], s["su"]["l"]) == (1, 2)
    assert (s["ats"]["w"], s["ats"]["l"], s["ats"]["push"]) == (1, 2, 0)
    assert (s["home_ats"]["w"], s["home_ats"]["l"]) == (1, 2)
    assert np.isclose(s["line"]["mae"], (9.5 + 10.5 + 10.5) / 3)
    assert np.isclose(s["line"]["bias"], (9.5 - 10.5 - 10.5) / 3)
    assert [lab for lab, _ in analysis.market_ml_bands(m)] == ["-249 to -175", "-129 to -100"]
    assert [lab for lab, _ in analysis.market_ats_bands(m)] == ["Fav −3 to −5.5",
                                                                "Pick'em −2.5 to +2.5"]
    sides = analysis.market_by_side(m)
    assert [lab for lab, _, _ in sides] == ["Home favourites", "Road favourites"]
    assert sides[1][1]["n"] == 2


def test_a_push_counts_neither_way_and_returns_the_stake():
    g = _market_rows().iloc[[1]].copy()
    g["home_pts"], g["away_pts"], g["close_spread"] = 104, 100, -4.0     # wins by 4 on -4
    m = analysis.market_games(g)
    assert m.iloc[0]["fav_cover"] == 0.5 and m.iloc[0]["ats_units"] == 0.0
    a = analysis.market_summary(m)["ats"]
    assert (a["w"], a["l"], a["push"]) == (0, 0, 1) and np.isnan(a["rate"])


def test_performance_shows_the_market_on_its_own(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(_market_rows(), synth(50, 57, "reconstructed"),
                           "2026-10-21", model_ok=True)
    perf = (tmp_path / "performance.html").read_text()
    for t in ("Market performance", "Favourites straight up", "Favourites · moneyline ROI",
              "Favourites against the spread", "Home sides against the spread",
              "Closing line vs final margin", "Moneyline by the favourite's closing price",
              "Against the spread by the favourite's closing line",
              "Home and road favourites", "DraftKings 2026-27 close"):
        assert t in perf, t
    assert "10.2 pts" in perf                       # mean miss of the closing line
    # games the model abstained on still count for the market
    assert "every graded game this season (3 games)" in perf


def test_market_view_waits_for_a_graded_close(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(ledger.empty(), synth(50, 58, "reconstructed"), "2026-10-21",
                           model_ok=True)
    perf = (tmp_path / "performance.html").read_text()
    assert "Starts with the first graded game with a DraftKings close." in perf
    assert "Favourites straight up" not in perf     # never the rebuilt rows


def _with_spreads(d, seed):
    rng = np.random.default_rng(seed)
    d = d.copy()
    d["close_spread"] = rng.choice([-9.5, -6.5, -3.5, -1.5, 2.5, 4.5, 7.5, 10.5], len(d))
    d["close_home_spread_odds"] = d["close_away_spread_odds"] = -110
    d["spread_book"] = d["close_book"]
    d["pre_spread"] = d["close_spread"]
    d["pre_home_spread_odds"] = d["pre_away_spread_odds"] = -110
    d["home_pts"] = 100 + rng.integers(-15, 16, len(d))
    d["away_pts"] = 100
    d["home_won"] = (d["home_pts"] > d["away_pts"]).astype(int)
    return d[d["home_pts"] != d["away_pts"]].reset_index(drop=True)


def test_card_context_counts_live_and_rebuilt_apart():
    native = ledger.graded(_with_spreads(synth(80, 3, "native"), 1))
    recon = ledger.graded(_with_spreads(synth(400, 4, "reconstructed"), 2))
    b = analysis.card_bands(native, recon)
    for kind in ("ml", "ats"):
        pool = b[kind]
        assert pool["n_market_games"] == (len(native), len(recon))
        ml = kind == "ml"
        want = (lambda g, px: len(analysis.picks(g, px).query("rule == 'lean'")) if ml
                else len(analysis.ats_side_picks(g, px)))
        assert len(pool["live"][0]) == want(native, "pre")            # never pooled
        assert len(pool["rebuilt"][0]) == want(recon, "close")
        assert len(pool["market"][0]) <= 2 * (len(native) + len(recon))   # both sides
        key = 0.6 if ml else -3.5
        w = analysis.card_band(pool, key, kind)
        assert w["lo"] < key < w["hi"] and abs(w["lo"] + w["hi"] - 2 * key) < 1e-12
        mk = pool["market"][0]
        assert w["market"]["n"] == int(((mk >= w["lo"]) & (mk <= w["hi"])).sum())
        assert analysis.card_band(pool, np.nan, kind) is None


def test_card_window_is_centred_on_the_price_and_widens_only_when_thin():
    # 200 sides at break-even 0.55, a handful far away: a -122 card sees only
    # its neighbours, never the long underdogs an equal-count band took in
    k = np.r_[np.full(200, 0.55), np.full(40, 0.20)]
    won = np.r_[np.ones(110), np.zeros(90), np.zeros(40)]
    q = k.copy()
    pool = dict(market=(k, won, q), live=(np.array([]),) * 3,
                rebuilt=(np.array([0.55, 0.21]), np.array([1.0, 0.0]), np.array([0.55, 0.21])))
    w = analysis.card_band(pool, 0.55, "ml")
    assert not w["widened"] and w["width"] == analysis.CARD_WINDOW["ml"]
    assert w["market"]["n"] == 200 and abs(w["market"]["implied"] - 0.55) < 1e-12
    assert w["rebuilt"]["n"] == 1 and w["live"] is None
    # a thin price widens, but never past CARD_WINDOW_MAX
    far = analysis.card_band(pool, 0.30, "ml")
    assert far is None or far["width"] <= analysis.CARD_WINDOW_MAX["ml"] + 1e-12
    thin = analysis.card_band(pool, 0.27, "ml")
    assert thin is not None and thin["widened"] and thin["market"]["n"] == 40


def test_opened_card_shows_both_context_panels():
    recon = ledger.graded(_with_spreads(synth(400, 4, "reconstructed"), 2))
    bands = analysis.card_bands(ledger.graded(ledger.empty()), recon)
    r = _with_pre_spread(_ats_pending(gp=3, first_p=0.66), 0.66, -7.5)
    card = build_site.game_card(r, bands=bands)
    assert "Moneyline · games priced like this" in card
    assert "Spread · games lined like this" in card
    assert "break-even" in card and "band " not in card
    assert card.count("<small>Model · live</small><strong>—</strong>") == 2  # no live picks yet
    assert "hindsight" in card and "never pooled" in card
    assert "historical" not in build_site.game_card(r)                    # no history: no panel
