from statistics import NormalDist

import numpy as np
import pytest
import pandas as pd

import analysis
import build_site
import ledger
import market


def synth(n=400, seed=0, basis="native", book="dk"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        q = rng.uniform(0.2, 0.8)
        p = np.clip(q + rng.normal(0, 0.05), 0.05, 0.95)
        won = int(rng.random() < q)
        vig = 0.022

        def ml(prob):
            prob = prob + vig
            return int(round(-100 * prob / (1 - prob))) if prob >= 0.5 \
                else int(round(100 * (1 - prob) / prob))
        hml, aml = ml(q), ml(1 - q)
        lean_home = p >= 0.5
        rows.append(dict(
            game_id=f"{basis}{i}", slate_date=f"2026-11-{1 + i % 28:02d}",
            season=2027, tip_utc="2026-11-20T00:30Z", model_tag="t",
            basis=basis, home="HOM", away="AWY", delta=10 * (p - 0.5),
            p_home=p, lean="HOM" if lean_home else "AWY",
            p_lean=p if lean_home else 1 - p, pre_book=book, pre_home_ml=hml,
            pre_away_ml=aml, pre_q_home=market.devig(hml, aml),
            close_home_ml=hml, close_away_ml=aml,
            close_q_home=market.devig(hml, aml), close_book=book,
            home_pts=100 + won,
            away_pts=100 + (1 - won), home_won=won))
    return pd.DataFrame(rows, columns=ledger.COLUMNS)


def test_ev_minus_null_equals_excess():
    h = analysis.with_close(ledger.graded(synth()))
    bands, pooled = analysis.lean_by_price(h)
    for b in bands + [pooled]:
        assert abs((b["ev"] - b["ev_null"]) - b["excess"]) < 1e-9
        assert b["ev_null"] < 0
    assert sum(b["n"] for b in bands) == pooled["n"] == len(h)


def test_market_calibration_has_no_both_sides_total():
    h = analysis.with_close(ledger.graded(synth()))
    rows, totals = analysis.market_calibration(h)
    assert set(totals) == {"home", "favourite"}
    assert sum(r["all"]["n"] for r in rows) == 2 * len(h)
    assert totals["favourite"]["n"] <= len(h)


def test_scoring_on_identical_rows():
    h = analysis.with_close(ledger.graded(synth()))
    s = analysis.scoring(h)
    assert s["n"] == len(h)
    assert abs(s["d_brier"] - (s["model"]["brier"] - s["market"]["brier"])) < 1e-12
    cal = analysis.model_calibration(h)
    assert sum(c["n"] for c in cal) == len(h)


def test_rows_without_close_are_excluded_not_imputed():
    df = synth(50)
    df.loc[:9, ["close_home_ml", "close_away_ml", "close_q_home"]] = np.nan
    h = analysis.with_close(ledger.graded(df))
    assert len(h) == 40


def test_value_and_lean_picks_are_graded_at_the_right_price():
    df = synth(300, 6)
    g = ledger.graded(df)
    p = analysis.picks(g, price="close")
    lean, val = p[p["rule"] == "lean"], p[p["rule"] == "value"]
    assert len(lean) == len(g) and (lean["model_p"] >= 0.5).all()
    assert (val["model_p"] > val["q"]).all() and (val["edge"] > 0).all()
    for _, r in val.head(20).iterrows():
        assert r["units"] == market.unit_profit(r["ml"], r["won"])
    s = dict(analysis.roi_summary(g, "close"))
    assert analysis.PICK_RULES[1][1] not in s      # value side retired from display
    top = s[analysis.PICK_RULES[0][1]][0]
    assert top["label"] == "All picks" and top["n"] == len(lean)
    assert abs(top["roi"] - lean["units"].mean()) < 1e-12
    assert top["roi_null"] < 0             # the hold: market-correct ROI is negative


def test_roi_splits_early_games_when_present():
    df = synth(200, 7)
    df.loc[:49, ["gp_home", "gp_away"]] = 5
    df.loc[50:, ["gp_home", "gp_away"]] = 30
    labels = [r["label"] for r in dict(analysis.roi_summary(
        ledger.graded(df), "close"))[analysis.PICK_RULES[0][1]]]
    assert labels[:3] == ["All picks", "Games 10+", "Games 1–9 (carryover)"]


def test_pages_render_bases_separately_with_ev_null(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(synth(200, 1, "native"),
                           synth(300, 2, "reconstructed"),
                           "2026-11-19", model_ok=True)
    cal = (tmp_path / "market-calibration.html").read_text()
    assert "Native (pregame-locked, forward)" in cal
    assert "Reconstructed (leave-one-season-out, hindsight)" in cal
    assert "500 games (200 native, 300 reconstructed;" in cal
    assert "Actual win rate" in cal and "<svg" in cal      # reliability charts
    assert "Bet grading by price band" in cal and "Null (pp)" not in cal
    grades = (tmp_path / "grades.html").read_text()
    assert "Closing-line value" in grades
    assert "ROI — one unit on every pick" in grades and "vs null (± 1 SE)" in grades
    assert "Probabilities vs the close" in grades        # verdict strip
    assert "pregame snapshot price" in grades        # native bettable price
    assert "Lean · P/L (1u)" in grades and "Value pick" not in grades
    assert "Value bets" not in grades and "Value ATS" not in grades
    assert "Retired 2026-10-06" in grades
    idx = (tmp_path / "index.html").read_text()
    assert "Model − market" in idx


def test_pages_render_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(ledger.empty(), ledger.empty(), "2026-10-01",
                           model_ok=False)
    assert "Fit model" in (tmp_path / "index.html").read_text()
    assert "No graded" in (tmp_path / "market-calibration.html").read_text()


def _two_books():
    return pd.concat([synth(120, 3, "reconstructed", book="espnbet"),
                      synth(80, 4, "reconstructed", book="dk")
                      .assign(game_id=lambda d: "dk" + d["game_id"])],
                     ignore_index=True)


def test_books_get_separate_sections_never_pooled(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    monkeypatch.setattr(build_site, "HIDDEN_BOOKS", ())     # every book shown
    recon = _two_books()
    build_site.write_pages(ledger.empty(), recon, "2026-11-19", model_ok=True)
    cal = (tmp_path / "market-calibration.html").read_text()
    grades = (tmp_path / "grades.html").read_text()
    for page_ in (cal, grades):
        assert "Reconstructed (leave-one-season-out, hindsight) · DraftKings close · 2026-27" in page_
        assert "Reconstructed (leave-one-season-out, hindsight) · ESPN BET close · 2026-27" in page_
    assert "120 games (0 native, 120 reconstructed;" in cal
    assert "80 games (0 native, 80 reconstructed;" in cal
    assert "200 games" not in cal
    parts = dict(analysis.book_split(analysis.with_close(ledger.graded(recon))))
    assert len(parts["espnbet"]) == 120 and len(parts["dk"]) == 80


def test_research_pages_leave_out_espn_bet_and_the_report_keeps_it(tmp_path, monkeypatch):
    # ESPN BET is no longer offered (owner's decision, 2026-10-21)
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    txt = build_site.write_pages(ledger.empty(), _two_books(), "2026-11-19",
                                 model_ok=True)
    pages = {f: (tmp_path / f).read_text() for f in
             ("grades.html", "market-calibration.html", "model.html")}
    for f, html in pages.items():
        assert "ESPN BET close" not in html and "Recon ESPN BET" not in html, f
        assert "Market · ESPN BET" not in html, f
        assert "DraftKings" in html and "no longer offered" in html, f
    cal = pages["market-calibration.html"]
    assert "80 games (0 native, 80 reconstructed;" in cal and "120 games" not in cal
    model = pages["model.html"]
    assert "DraftKings 2026-27" in model and "ESPN BET 2026-27" not in model
    assert "ESPN BET 2026-27" in txt                      # the report keeps it


def test_clv_only_within_one_book():
    g = ledger.graded(synth(40, 5))
    g["pre_q_home"] = g["close_q_home"] - 0.02
    g.loc[g.index[:10], "close_book"] = "espnbet"      # pre dk, close espnbet
    c = analysis.clv(g)
    assert c["n"] == 30


def test_ats_picks_grade_covers_pushes_and_null():
    df = synth(200, 3)
    rng = np.random.default_rng(3)
    df["close_spread"] = np.round(rng.normal(0, 6, len(df)) * 2) / 2
    df["close_home_spread_odds"] = -110
    df["close_away_spread_odds"] = -110
    df["home_pts"] = 100 + rng.integers(-15, 16, len(df))
    df["away_pts"] = 100
    df.loc[0, ["close_spread", "home_pts"]] = [-5.0, 105]   # a push
    df.loc[1, "close_spread"] = np.nan                       # no spread: skipped
    g = ledger.graded(df)
    p = analysis.ats_picks(g)
    lean, val = p[p["rule"] == "lean"], p[p["rule"] == "value"]
    assert len(lean) == len(g) - 1
    assert (val["model_p"] > 0.5).all()
    push = lean[lean["game_id"] == g.iloc[0]["game_id"]].iloc[0]
    assert push["result"] == 0.5 and push["units"] == 0.0
    assert np.allclose(lean["q"], 0.5)
    r = analysis.ats_row("All", lean)
    assert r["w"] + r["l"] + r["push"] == r["n"] and r["push"] >= 1
    assert abs(r["roi_null"] - (0.5 * (1 + 100 / 110) - 1)) < 1e-9
    assert abs(r["breakeven"] - 110 / 210) < 1e-9
    assert analysis.ats_summary(synth(20)) == []   # no spread columns filled


def test_roi_by_band_partitions_picks_with_ev_null():
    g = ledger.graded(synth(400, 8))
    p = analysis.picks(g, "close")
    bands = dict(analysis.roi_by_band(g, "close"))
    assert set(bands) == {label for rule, label in analysis.PICK_RULES
                          if rule in analysis.SHOWN_RULES}
    for rule, label in analysis.PICK_RULES:
        if rule not in analysis.SHOWN_RULES:
            continue
        rows = bands[label]
        d = p[p["rule"] == rule]
        assert sum(r["n"] for r in rows) == len(d)
        assert abs(sum(r["units"] for r in rows) - d["units"].sum()) < 1e-9
        for r in rows:
            s = d[d["ml"].map(market.ladder_rung) == r["label"]]
            assert abs(r["ev"] - (s["won"].mean() - s["breakeven"].mean())) < 1e-12
            assert abs(r["ev_null"] - (s["q"].mean() - s["breakeven"].mean())) < 1e-12
            assert r["ev_null"] < 0                 # the hold, not zero


def test_roi_by_band_filters_model_tag_and_page_shows_current(tmp_path, monkeypatch):
    df = synth(300, 9)
    df.loc[:149, "model_tag"] = build_site.MODEL_TAG_V4_AVAIL
    g = ledger.graded(df)
    v3 = analysis.roi_by_band(g, "pre", tags=[build_site.MODEL_TAG_V4_AVAIL])
    assert sum(r["n"] for r in dict(v3)[analysis.PICK_RULES[0][1]]) == 150
    assert analysis.roi_by_band(g, "pre", tags=["nope"]) == []
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    monkeypatch.setattr(build_site, "ACTIVE_TAGS", [build_site.MODEL_TAG_V4,
                                                    build_site.MODEL_TAG_V4_AVAIL])
    build_site.write_pages(df, synth(200, 10, "reconstructed"), "2026-11-19",
                           model_ok=True)
    grades = (tmp_path / "grades.html").read_text()
    assert "ROI by price band" in grades and "EV null (pp)" in grades
    assert "Current model rows only" in grades and "<th class=''>z</th>" in grades


def test_single_bet_band_shows_a_dash_not_nan(tmp_path, monkeypatch):
    df = synth(200, 11)
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(df, ledger.empty(), "2026-11-19", model_ok=True)
    grades = (tmp_path / "grades.html").read_text()
    assert ">nan<" not in grades and "v1 model" not in grades
    assert build_site.se_txt(float("nan")) == "—" and build_site.se_txt(0.023) == "2.3"


def test_seasons_get_separate_sections_never_pooled(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    recon = pd.concat([synth(120, 3, "reconstructed").assign(season=2025),
                       synth(50, 4, "reconstructed")
                       .assign(season=2026, game_id=lambda d: "s" + d["game_id"])],
                      ignore_index=True)
    build_site.write_pages(ledger.empty(), recon, "2026-11-19", model_ok=True)
    grades = (tmp_path / "grades.html").read_text()
    assert "DraftKings close · 2024-25" in grades and "120 graded games" in grades
    assert "DraftKings close · 2025-26" in grades and "50 graded games" in grades
    assert "170 graded games" not in grades


def test_one_lean_definition_everywhere(tmp_path, monkeypatch):
    """A row at exactly p_home = 0.5 is graded as the side it recorded, so the
    lean record agrees across the tiles, ROI table, accuracy and ATS."""
    df = synth(100, 12, "reconstructed")
    df.loc[0, ["p_home", "p_lean", "lean"]] = [0.5, 0.5, "AWY"]
    h = analysis.with_close(ledger.graded(df))
    _, pooled = analysis.lean_by_price(h)
    top = dict(analysis.roi_summary(h, "close"))[analysis.PICK_RULES[0][1]][0]
    assert (top["w"], top["l"]) == (pooled["w"], pooled["l"])
    assert abs(analysis.scoring(h)["model"]["acc"] - pooled["win"]) < 1e-12
    lean = analysis.picks(h)
    assert lean[(lean["rule"] == "lean") & (lean["game_id"] == df.loc[0, "game_id"])
                ]["side"].iloc[0] == "AWY"


def test_open_price_is_devigged_from_the_open_pair():
    df = synth(60, 13)
    df["open_home_ml"], df["open_away_ml"] = -150, 130
    p = analysis.picks(ledger.graded(df), price="open")
    lean = p[p["rule"] == "lean"]
    q = market.devig(-150, 130)
    assert np.allclose(np.where(lean["ml"] == -150, lean["q"], 1 - lean["q"]), q)


def test_hypotheses_use_frozen_rules_and_render_native_beside_hindsight(tmp_path,
                                                                        monkeypatch):
    df = synth(300, 14)
    df.loc[:99, ["gp_home", "gp_away"]] = 4
    df.loc[100:, ["gp_home", "gp_away"]] = 30
    g = ledger.graded(df)
    h1, h2, h3, h4, a1, h5, h6 = analysis.HYPOTHESES
    assert h4["native"] == "first" and h4["min_edge"] == h3["min_edge"]
    # amendment A1: H2's frozen rule, graded at the first snapshot
    assert a1["native"] == "first" and {k: v for k, v in a1.items()
                                        if k not in ("key", "native")} == \
        {k: v for k, v in h2.items() if k not in ("key", "native")}
    assert [h["native"] for h in (h1, h2, h3)] == ["pre"] * 3
    d1 = analysis.hypothesis_picks(g, h1, "pre")
    assert d1["early"].all() and (d1["edge"] >= 0.08).all()
    d2 = analysis.hypothesis_picks(g, h2, "pre")
    assert (~d2["early"]).all() and (d2["q"] > 0.5).all() and (d2["edge"] > 0).all()
    d3 = analysis.hypothesis_picks(g, h3, "pre")
    assert (d3["edge"] >= 0.12).all()
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(df, synth(200, 15, "reconstructed"), "2026-11-19",
                           model_ok=True)
    grades = (tmp_path / "grades.html").read_text()
    assert "Pre-registered hypotheses" in grades
    for hyp in analysis.HYPOTHESES:
        assert f"<b>{hyp['key']}</b>" in grades
    assert "left one hypothesis to test forward" not in grades   # retired band
    assert "retired" in grades


def test_colour_marks_only_two_se_gaps():
    assert build_site.sig_cls(1.99) == "" and build_site.sig_cls(-1.5) == ""
    assert build_site.sig_cls(2.0) == "pos" and build_site.sig_cls(-2.4) == "neg"
    assert build_site.sig_cls(float("nan")) == ""
    r = dict(roi=-0.026, roi_null=-0.041, roi_se=0.023)   # negative ROI above null
    cells = build_site._roi_cells(r)
    assert "class=''" in cells[0] and "-2.6%" in cells[0]
    assert build_site.season_txt(2026) == "2025-26"


def test_native_empty_state_counts_pending_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    nat = synth(5, 16)
    nat[["home_won", "home_pts", "away_pts", "close_home_ml", "close_away_ml",
         "close_q_home"]] = np.nan
    build_site.write_pages(nat, synth(50, 17, "reconstructed"), "2026-11-19",
                           model_ok=True)
    assert "5 pregame rows recorded, 5 waiting" in (tmp_path / "grades.html").read_text()


def test_one_graded_game_renders_dashes_not_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(synth(1, 18), synth(1, 19, "reconstructed"),
                           "2026-11-19", model_ok=True)
    for f in ("grades.html", "market-calibration.html"):
        s = (tmp_path / f).read_text()
        assert ">nan" not in s and "± nan" not in s and "nan (1 SE)" not in s


def with_first(df, p_shift=0.0, ml_shift=0):
    """First-snapshot columns: an earlier price and model P than the latest."""
    df = df.copy()
    df["first_snapshot_utc"] = "2026-11-19T10:00:00Z"
    df["first_model_tag"] = "t"
    df["first_book"] = df["pre_book"]
    df["first_p_home"] = np.clip(df["p_home"] + p_shift, 0.05, 0.95)
    df["first_home_ml"] = df["pre_home_ml"] + ml_shift
    df["first_away_ml"] = df["pre_away_ml"]
    df["first_q_home"] = [market.devig(h, a) for h, a in
                          zip(df["first_home_ml"], df["first_away_ml"])]
    return df


def test_first_price_grades_the_early_p_at_the_early_price():
    df = with_first(synth(200, 20), p_shift=0.1, ml_shift=-15)
    g = ledger.graded(df)
    p = analysis.picks(g, price="first")
    val = p[p["rule"] == "value"].set_index("game_id")
    for gid, r in val.head(20).iterrows():
        src = df.set_index("game_id").loc[gid]
        home = r["side"] == "HOM"
        assert r["model_p"] == pytest.approx(src["first_p_home"] if home
                                             else 1 - src["first_p_home"])
        assert r["ml"] == (src["first_home_ml"] if home else src["first_away_ml"])
    lean = p[p["rule"] == "lean"]
    assert (lean["model_p"] >= 0.5).all()          # lean = the early P's side
    assert analysis.picks(ledger.graded(synth(30, 21)), price="first").empty


def test_h4_graded_at_the_first_snapshot_on_the_page(tmp_path, monkeypatch):
    df = synth(300, 22)
    df[["gp_home", "gp_away"]] = 30
    df = with_first(df, p_shift=0.15)               # early P disagrees more
    g = ledger.graded(df)
    h3, h4 = analysis.HYPOTHESES[2], analysis.HYPOTHESES[3]
    d3 = analysis.hypothesis_picks(g, h3, "pre")
    d4 = analysis.hypothesis_picks(g, h4, "first")
    assert len(d4) > len(d3)                        # different P, different bets
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(df, synth(200, 23, "reconstructed"), "2026-11-19",
                           model_ok=True)
    grades = (tmp_path / "grades.html").read_text()
    assert "<b>H4</b>" in grades and "at the first snapshot" in grades
    assert f"{len(d4)}<br><span class='mut'>at the first snapshot</span>" in grades
    assert "At the <b>first snapshot</b>" in grades


# ------------------------------------------------ H5: against the spread
def _ats_rows(n=6):
    """Graded games 10+ with a first-snapshot and an opening spread."""
    d = synth(n, 21)
    d[["gp_home", "gp_away"]] = 30
    d["home_pts"], d["away_pts"] = 110, 100            # home wins by 10
    d["home_won"] = 1
    for k in ("first", "open", "close"):
        d[f"{k}_spread"] = -5.5
        d[f"{k}_home_spread_odds"] = -110
        d[f"{k}_away_spread_odds"] = -110
    d["first_p_home"] = d["p_home"]
    return d


def _hyp(key):
    return next(h for h in analysis.HYPOTHESES if h["key"] == key)


def test_h5_is_frozen_ats_at_the_first_snapshot():
    h5 = _hyp("H5")
    assert h5["market"] == "ats"
    assert (h5["sigma"], h5["min_edge"], h5["early"]) == (13.5, 1.0, False)
    assert (h5["native"], h5["hindsight"]) == ("first", "open")
    assert analysis.ATS_MAIN_LINE == (-135, 115)


def test_h6_is_h5s_rule_on_games_1_to_9():
    h5, h6 = _hyp("H5"), _hyp("H6")
    assert h6["early"] is True and h5["early"] is False
    same = ("market", "favourite", "min_edge", "sigma", "hindsight", "native")
    assert {k: h6[k] for k in same} == {k: h5[k] for k in same}
    g = ledger.graded(_ats_rows(4))
    g["first_p_home"] = 0.75                         # edge 3.6 pts, home
    early = g.assign(gp_home=3, gp_away=12)          # the fewer of the two < 10
    assert len(analysis.hypothesis_picks(early, h6, "first")) == 4
    assert analysis.hypothesis_picks(early, h5, "first").empty
    assert analysis.hypothesis_picks(g, h6, "first").empty


def test_h5_reads_the_first_snapshot_p_and_line_and_its_edge_in_points():
    h5 = _hyp("H5")
    g = ledger.graded(_ats_rows(4))
    # margin 13.5 * Phi^-1(P) against home -5.5: P 0.75 -> +9.1 (edge 3.6,
    # home); 0.60 -> +3.4 (edge 2.1, away); 0.66 -> +5.6 (edge 0.1: no bet)
    g["first_p_home"] = [0.75, 0.60, 0.66, 0.75]
    g["p_home"] = 0.10                                  # pregame P: not read
    g.loc[g.index[3], "first_home_spread_odds"] = 320  # alternate line: no bet
    d = analysis.hypothesis_picks(g, h5, "first")
    assert list(d["side"]) == ["HOM", "AWY"]
    assert abs(d["edge"].iloc[0] - (13.5 * NormalDist().inv_cdf(0.75) - 5.5)) < 1e-9
    assert list(d["result"]) == [1.0, 0.0]              # home covered -5.5 by 4.5
    r = analysis.ats_row("x", d)
    assert r["n"] == 2 and abs(r["roi_null"] - (0.5 * 210 / 110 - 1)) < 1e-9
    assert "fav_roi" in r                               # same-row baseline
    early = g.assign(gp_home=4, gp_away=4)
    assert analysis.hypothesis_picks(early, h5, "first").empty


def test_h5_hindsight_reads_the_open_spread_cache(tmp_path, monkeypatch):
    nat = ledger.graded(_ats_rows(3))
    rec = _ats_rows(3).assign(basis="reconstructed")
    rec = rec.drop(columns=["open_spread", "open_home_spread_odds",
                            "open_away_spread_odds"])
    cache = tmp_path / "open.csv"
    pd.DataFrame(dict(game_id=rec["game_id"], open_spread=-1.5,
                      open_home_spread_odds=-110, open_away_spread_odds=-110)
                 ).to_csv(cache, index=False)
    monkeypatch.setattr(analysis, "OPEN_SPREADS", str(cache))
    rows = {h["key"]: (hind, n) for h, hind, n in analysis.hypothesis_rows(nat, rec)}
    hind, native = rows["H5"]
    assert hind and hind["n"] == 3 and native["n"] >= 1
    # without the cache the hindsight cell is empty, never the close
    monkeypatch.setattr(analysis, "OPEN_SPREADS", str(tmp_path / "missing.csv"))
    rows = {h["key"]: hind for h, hind, _ in analysis.hypothesis_rows(nat, rec)}
    assert rows["H5"] is None


def test_h5_h6_by_season_partition_the_pooled_scoreboard(tmp_path, monkeypatch):
    rec = _ats_rows(12).assign(basis="reconstructed")
    rec["season"] = [2025] * 6 + [2026] * 6
    rec["close_book"] = ["espnbet"] * 6 + ["dk"] * 3 + ["espnbet"] * 3
    rec.loc[rec.index[:4], ["gp_home", "gp_away"]] = 3     # four games 1-9 (H6)
    rec["p_home"] = np.linspace(0.62, 0.9, 12)             # all clear 1 pt vs -5.5
    rec.loc[rec.index[::2], "home_pts"] = 101              # mixed covers
    cache = tmp_path / "open.csv"
    rec[["game_id", "open_spread", "open_home_spread_odds",
         "open_away_spread_odds"]].to_csv(cache, index=False)
    rec = rec.drop(columns=["open_spread", "open_home_spread_odds",
                            "open_away_spread_odds"])
    monkeypatch.setattr(analysis, "OPEN_SPREADS", str(cache))
    nat = ledger.empty()
    pooled = {h["key"]: hind for h, hind, _ in analysis.hypothesis_rows(nat, rec)}
    by = {h["key"]: parts for h, parts in analysis.hypothesis_by_season(nat, rec)}
    assert list(by) == ["H5", "H6"]
    for key, parts in by.items():
        assert parts[0]["basis"] == "native" and parts[0]["row"] is None
        rp = [p for p in parts if p["basis"] == "reconstructed"]
        assert sum(p["row"]["n"] for p in rp) == pooled[key]["n"]
        assert abs(sum(p["row"]["units"] for p in rp) - pooled[key]["units"]) < 1e-9
        assert [p["season"] for p in rp] == sorted((p["season"] for p in rp),
                                                   reverse=True)
    h5 = [(p["book"], p["season"]) for p in by["H5"] if p["row"]]
    assert h5 == [("dk", 2026), ("espnbet", 2026), ("espnbet", 2025)]
    # on one book: the DraftKings bets alone, the same as the pooled DK part
    dk = {h["key"]: hind for h, hind, _ in
          analysis.hypothesis_rows(nat, rec, book="dk")}
    by_dk = {h["key"]: parts for h, parts in
             analysis.hypothesis_by_season(nat, rec, book="dk")}
    dk_part = next(p["row"] for p in by["H5"] if p["book"] == "dk")
    assert dk["H5"]["n"] == dk_part["n"] == 3 and dk["H6"] is None
    assert [(p["book"], p["season"]) for p in by_dk["H5"] if p["row"]] == [("dk", 2026)]
    # the page and the report show DraftKings only (ESPN BET is not offered)
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(nat, rec, "2026-11-19", model_ok=True)
    grades = (tmp_path / "grades.html").read_text()
    assert "H5 / H6 by season — never pooled" in grades
    assert "DraftKings 2025-26" in grades and "ESPN BET 2024-25" not in grades
    assert "ESPN BET 2025-26" not in grades
    assert "no graded bets yet" in grades and "DraftKings backtest" in grades
    txt = (tmp_path / "ledger_report.txt").read_text()
    assert "By season, never pooled" in txt and "H5   DraftKings 2025-26 @ open" in txt
    assert "ESPN BET 2024-25 @ open" not in txt and "DraftKings backtest" in txt


def test_dk_backtest_reads_one_book_and_reprices_the_open_bets(tmp_path, monkeypatch):
    rec = _ats_rows(8).assign(basis="reconstructed", season=2026)
    rec["close_book"] = ["dk"] * 5 + ["espnbet"] * 3
    rec["slate_date"] = ["2026-01-10"] * 2 + ["2026-03-10"] * 6
    rec["p_home"] = 0.8                                # margin +11.4 vs -5.5
    rec["close_spread"] = -6.5                         # one point toward home
    cache = tmp_path / "open.csv"
    rec[["game_id", "open_spread", "open_home_spread_odds",
         "open_away_spread_odds"]].to_csv(cache, index=False)
    monkeypatch.setattr(analysis, "OPEN_SPREADS", str(cache))
    rec = rec.drop(columns=["open_spread", "open_home_spread_odds",
                            "open_away_spread_odds"])
    bt = {b["label"]: b for b in analysis.hypothesis_backtest(rec, "dk")}
    assert list(bt) == ["H1", "H2 / H2·F", "H3 / H4", "H5", "H6"]
    h5 = bt["H5"]
    assert h5["open"]["n"] == 5 and h5["close"]["n"] == 5      # DraftKings only
    assert h5["open_at_close"]["n"] == 5
    assert h5["move"]["mean"] == 1.0 and h5["move"]["unit"] == "pts"
    assert "fav_roi" not in h5["open_at_close"]                 # priced the open
    assert [(name, r["n"]) for name, r in h5["halves"]] == [("to 31 Jan", 2),
                                                             ("from 1 Feb", 3)]
    assert bt["H6"]["open"] is None and bt["H6"]["close"] is None


def test_ledger_ats_fold_reads_draftkings_spreads_only():
    h = analysis.with_close(ledger.graded(_ats_rows(6)))
    h["spread_book"] = ["dk"] * 4 + ["espnbet"] * 2     # as market_view keeps it
    body = []
    build_site._ats_section(body, h)
    html = "".join(body)
    assert "<td class=''>4</td>" in html and "DraftKings closing spread" in html
