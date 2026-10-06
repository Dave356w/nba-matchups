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
