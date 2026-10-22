"""Model v7: roster talent on games 1-9, with H1 and H6 kept on the
carryover model's P (carry_p_home / first_carry_p_home)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import analysis
import backfill_history as bf
import build_site
import ledger

TIP = "2026-11-20T00:30Z"
BEFORE = datetime(2026, 11, 19, 18, 0, tzinfo=timezone.utc)
LATER_BEFORE = datetime(2026, 11, 19, 23, 0, tzinfo=timezone.utc)
AFTER = datetime(2026, 11, 20, 1, 0, tzinfo=timezone.utc)


def row(**kw):
    r = dict(game_id="401", slate_date="2026-11-19", season=2027, tip_utc=TIP,
             model_tag=build_site.MODEL_TAG_V7_EARLY, home="GSW", away="UTA",
             gp_home=3, gp_away=3, delta=5.0, p_home=0.45, carry_p_home=0.62,
             lean="UTA", p_lean=0.55, pre_book="dk", pre_home_ml=-110,
             pre_away_ml=-110, pre_q_home=0.5)
    r.update(kw)
    return r


def test_carry_p_refreshed_before_tip_first_written_once_never_graded():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    r = led.iloc[0]
    assert (r["carry_p_home"], r["first_carry_p_home"]) == (0.62, 0.62)
    led, acc, _ = ledger.upsert_pregame(led, [row(carry_p_home=0.66, p_home=0.48)],
                                        now=LATER_BEFORE)
    r = led.iloc[0]
    assert acc and r["carry_p_home"] == 0.66 and r["first_carry_p_home"] == 0.62
    led, acc, _ = ledger.upsert_pregame(led, [row(carry_p_home=0.9)], now=AFTER)
    assert not acc and led.iloc[0]["carry_p_home"] == 0.66
    assert "carry_p_home" not in ledger.GRADE_COLUMNS
    assert "first_carry_p_home" in ledger.WRITE_ONCE_COLUMNS


def test_pre_v7_file_loads_and_validation_bounds_carry(tmp_path):
    import validate_data_files as v
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()], now=BEFORE)
    p = tmp_path / "old.csv"
    led[ledger.PRE_CARRY_COLUMNS].to_csv(p, index=False)
    back = ledger.load(str(p))
    assert list(back.columns) == ledger.COLUMNS and back["carry_p_home"].isna().all()
    assert v.check(str(p)) == []
    ledger.save(led, str(p))
    assert v.check(str(p)) == []
    bad = led.copy()
    bad.loc[:, ["gp_home", "gp_away"]] = 12              # a games-10+ row
    ledger.save(bad, str(p))
    assert any("outside games 1-9" in e for e in v.check(str(p)))


def graded(**kw):
    r = {c: np.nan for c in ledger.COLUMNS}
    r.update(row(), basis="native", home_won=1, close_book="dk",
             close_home_ml=-110, close_away_ml=-110, close_q_home=0.5)
    r.update(kw)
    return pd.DataFrame([r], columns=ledger.COLUMNS)


def test_h1_bets_on_the_carryover_p_not_the_v7_p():
    h1 = next(h for h in analysis.HYPOTHESES if h["key"] == "H1")
    g = graded()
    # v7 P 0.45 vs q 0.5 would be the away side at 0.05 (no H1 bet); the
    # carryover P 0.62 is the home side at 0.12 (an H1 bet)
    d = analysis.hypothesis_picks(g, h1, "pre")
    assert len(d) == 1 and d.iloc[0]["side"] == "GSW"
    assert abs(d.iloc[0]["model_p"] - 0.62) < 1e-9
    # a row from before v7 (no carry recorded) reads p_home as before
    old = graded(carry_p_home=np.nan, p_home=0.62)
    assert analysis.hypothesis_picks(old, h1, "pre").iloc[0]["side"] == "GSW"
    # the games-10+ rules never read it
    v = analysis.carry_view(g)
    assert v.iloc[0]["p_home"] == 0.62 and g.iloc[0]["p_home"] == 0.45


def test_rescore_writes_the_carryover_p():
    recon = pd.DataFrame([dict(
        game_id=str(i), slate_date="2026-01-0%d" % (i + 1), season=2026,
        tip_utc="2026-01-01T00:00Z", model_tag=build_site.MODEL_TAG_V6,
        basis="reconstructed", home="HOM", away="AWY", delta=1.0, p_home=0.6,
        lean="HOM", p_lean=0.6, home_won=1) for i in range(2)], columns=ledger.COLUMNS)
    test = pd.DataFrame(dict(
        date=pd.to_datetime(["2026-01-01", "2026-01-02"]), home="HOM", away="AWY",
        delta=[2.0, 7.0], p_home=[0.55, 0.7], carry_p_home=[0.6, np.nan],
        gp_home=[3, 11], gp_away=[4, 12],
        model_tag=[build_site.MODEL_TAG_V7_EARLY, build_site.MODEL_TAG_V6]))
    out, n = bf.rescore(recon, test)
    assert n == 2 and out.iloc[0]["carry_p_home"] == 0.6 and pd.isna(out.iloc[1]["carry_p_home"])
    assert list(out["model_tag"]) == [build_site.MODEL_TAG_V7_EARLY, build_site.MODEL_TAG_V6]
