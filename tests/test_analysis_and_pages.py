import numpy as np
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
    top = s[analysis.PICK_RULES[1][1]][0]
    assert top["label"] == "All picks" and top["n"] == len(val)
    assert abs(top["roi"] - val["units"].mean()) < 1e-12
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
    assert "Value pick · P/L (1u)" in grades
    idx = (tmp_path / "index.html").read_text()
    assert "Model − market" in idx


def test_pages_render_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    build_site.write_pages(ledger.empty(), ledger.empty(), "2026-10-01",
                           model_ok=False)
    assert "Fit model" in (tmp_path / "index.html").read_text()
    assert "No graded" in (tmp_path / "market-calibration.html").read_text()


def test_books_get_separate_sections_never_pooled(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "OUT_DIR", tmp_path)
    recon = pd.concat([synth(120, 3, "reconstructed", book="espnbet"),
                       synth(80, 4, "reconstructed", book="dk")
                       .assign(game_id=lambda d: "dk" + d["game_id"])],
                      ignore_index=True)
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
    for rule, label in analysis.PICK_RULES:
        rows = dict(analysis.roi_by_band(g, "close"))[label]
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
    recon = pd.concat([synth(120, 3, "reconstructed", book="espnbet").assign(season=2025),
                       synth(50, 4, "reconstructed", book="espnbet")
                       .assign(season=2026, game_id=lambda d: "s" + d["game_id"])],
                      ignore_index=True)
    build_site.write_pages(ledger.empty(), recon, "2026-11-19", model_ok=True)
    grades = (tmp_path / "grades.html").read_text()
    assert "ESPN BET close · 2024-25" in grades and "120 graded games" in grades
    assert "ESPN BET close · 2025-26" in grades and "50 graded games" in grades
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
    h1, h2, h3 = analysis.HYPOTHESES
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
