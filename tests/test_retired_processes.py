"""Production ignores retired markets/pages while preserving their history."""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import analysis
import build_site as bs
import kalshi
import ledger
import market
import report
from test_kalshi_benchmark import BEFORE, row
from test_report_and_ui import synth


def test_historical_kalshi_columns_survive_refresh_and_grade():
    old = synth(1, 7)
    incoming = row(game_id=str(old.iloc[0].game_id))
    old["home_won"] = float("nan")
    old.loc[:, "tip_utc"] = incoming["tip_utc"]
    old.loc[:, "kalshi_pre_home_ml"] = -135
    old.loc[:, "kalshi_first_home_ml"] = -130
    old.loc[:, "kalshi_close_home_ml"] = -140
    historical = old[ledger.KALSHI_COLUMNS].copy()
    refreshed, _, _ = ledger.upsert_pregame(old, [incoming], now=BEFORE)
    pd.testing.assert_frame_equal(refreshed[ledger.KALSHI_COLUMNS], historical)
    ledger.apply_result(refreshed, str(old.iloc[0].game_id),
                        dict(completed=True, home_pts=110, away_pts=100),
                        dict(book="dk", close_home_ml=-150, close_away_ml=130))
    pd.testing.assert_frame_equal(refreshed[ledger.KALSHI_COLUMNS], historical)
    assert not set(ledger.KALSHI_COLUMNS) & set(ledger.GRADE_COLUMNS)
    fresh, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    assert fresh[ledger.KALSHI_COLUMNS].isna().all().all()


def test_exhibition_build_does_not_read_or_write_retired_files(tmp_path, monkeypatch,
                                                             weights, model):
    monkeypatch.chdir(tmp_path)
    history = Path(ledger.PRESEASON_PATH)
    history.parent.mkdir()
    history.write_bytes(b"historical preseason bytes must stay untouched\n")
    monkeypatch.setattr(bs, "OUT_DIR", tmp_path / "public")
    monkeypatch.setattr(bs, "REPORT_PATH", tmp_path / "report.txt")
    monkeypatch.setattr(bs, "load_model", lambda: (weights, model))
    monkeypatch.setattr(bs, "load_avail", lambda: None)
    monkeypatch.setattr(market, "scoreboard", lambda date: [dict(
        game_id="pre", season_type=market.PRESEASON, home="BOS", away="LAL",
        state="pre", tip_utc="2099-10-23T23:00Z")])
    def forbidden(*args, **kw):
        raise AssertionError("retired network/source read")
    monkeypatch.setattr(kalshi, "open_games", forbidden)
    monkeypatch.setattr(kalshi, "grade_odds", forbidden)
    monkeypatch.setattr(bs, "fresh_logs", forbidden)
    bs.OUT_DIR.mkdir()
    (bs.OUT_DIR / "preseason.html").write_text("stale retired page")
    assert bs.main(["--date", "2026-10-23"]) == 0
    assert history.read_bytes() == b"historical preseason bytes must stay untouched\n"
    assert not (bs.OUT_DIR / "preseason.html").exists()
    assert not len(ledger.load(ledger.NATIVE_PATH))
    assert all("preseason.html" not in (bs.OUT_DIR / page).read_text()
               for page, _ in bs.PAGES)
    assert "== PRESEASON" not in bs.REPORT_PATH.read_text()


def test_exactly_the_nine_registered_workflows_remain():
    root = Path(bs.__file__).parent
    assert {p.stem for p in (root / ".github/workflows").glob("*.yml")} == {
        "build", "watch", "tests", "fit-model", "backfill", "rebuild", "box-history",
        "history-backtest", "margin-model"}
