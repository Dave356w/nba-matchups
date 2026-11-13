"""Blend shadow record (blend_shrink40_v1): the registered estimate,
population and scoring, never a later price in place of a lock value."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))

import analysis  # noqa: E402
import ledger    # noqa: E402
import market    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "blend_forward", os.path.join(_here, "..", "research", "blend_forward.py"))
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)


def _rows(n, slate="2026-11-14", seed=3, **over):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: np.nan for c in ledger.COLUMNS}, index=range(n))
    for c in ledger.TEXT_COLUMNS:
        d[c] = d[c].astype(object)
    tip = pd.Timestamp(f"{slate}T00:00Z") + pd.Timedelta(days=1)
    q = rng.uniform(.25, .8, n)
    hml = np.where(q >= .5, -100 * q / (1 - q) - 10, 100 * (1 - q) / q - 10).round()
    aml = np.where(q >= .5, 100 * q / (1 - q) - 10, -100 * (1 - q) / q - 10).round()
    d = d.assign(
        game_id=[str(700000 + i) for i in range(n)], basis="native", season=2027,
        slate_date=slate, tip_utc=tip.isoformat(), home="AAA", away="BBB",
        gp_home=12, gp_away=12, lock_p_home=np.clip(q + rng.normal(0, .08, n), .05, .95),
        lock_model_tag="x_margin_avail_v11", lock_q_home=q,
        lock_home_ml=hml, lock_away_ml=aml,
        lock_snapshot_utc=(tip - pd.Timedelta(hours=4)).isoformat(),
        close_q_home=np.clip(q + .02, .01, .99), close_home_ml=hml, close_away_ml=aml,
        home_won=(rng.random(n) < q).astype(float))
    for k, v in over.items():
        d[k] = v
    return d


def test_estimate_is_shrink_p_at_the_registered_fixed_weight():
    assert bf.W == 0.40
    p, q = np.array([.7, .3, .55]), np.array([.6, .45, .5])
    assert np.allclose(bf.blend(p, q), analysis.shrink_p(p, q, 0.40))
    # between the model and the market, on the model's side of q
    b = bf.blend(p, q)
    assert np.all((b - q) * (p - q) >= 0) and np.all(np.abs(b - q) <= np.abs(p - q))
    assert np.isclose(bf.blend(np.array([.6]), np.array([.6]))[0], .6)


def test_population_cuts_start_voids_family_stage_and_unpriced_locks():
    d = _rows(30)
    before = _rows(3, slate="2026-11-12", seed=4)
    before["game_id"] = ["800001", "800002", "800003"]
    d.loc[0, "game_id"] = next(iter(analysis.VOID_LOCKS))           # void
    d.loc[1, "lock_model_tag"] = "x_margin_v10"                      # off-family
    d.loc[2, ["gp_home", "gp_away"]] = 0                              # opener
    d.loc[3, "lock_away_ml"] = np.nan                                 # one price
    d.loc[4, "lock_q_home"] = np.nan                                  # no q
    d.loc[5, "home_won"] = np.nan                                     # pending
    d.loc[6:9, ["gp_home", "gp_away"]] = 5                            # games 1-9
    pop = bf.population(pd.concat([before, d], ignore_index=True))
    assert len(pop) == 30 - 5                     # pending stays in the population
    assert not pop["slate_date"].lt(bf.START_SLATE).any()
    assert len(bf.graded(pop)) == 24
    assert set(pop["prediction_stage"]) == {"early", "established"}
    assert np.allclose(pop["blend_p_home"],
                       analysis.shrink_p(pop["lock_p_home"], pop["lock_q_home"], 0.40))


def test_lock_before_registration_never_enters():
    d = _rows(5, slate="2026-11-13")
    d["lock_snapshot_utc"] = "2026-11-13T15:00:00+00:00"
    assert len(bf.population(d)) == 0


def test_later_prices_never_replace_lock_values():
    d = _rows(200)
    g = bf.graded(bf.population(d))
    s1 = bf.scores(g)
    d2 = d.copy()
    d2["close_q_home"] = 1 - d2["close_q_home"]
    d2["p_home"] = 0.99
    s2 = bf.scores(bf.graded(bf.population(d2)))
    for k in ("blend", "lock price", "model"):
        assert s1["rows"][k] == s2["rows"][k]
    assert s1["diffs"] == s2["diffs"]


def test_scores_pair_identical_games_and_read_the_primary_cell():
    g = bf.graded(bf.population(_rows(400)))
    s = bf.scores(g)
    assert s["n"] == len(g)
    d = s["diffs"]["blend − lock price"]
    assert np.isclose(d["ll"], s["rows"]["blend"]["ll"] - s["rows"]["lock price"]["ll"])
    assert bf.reading(s).split(" ")[0] in ("supports", "contradicts", "inconclusive")
    assert bf.reading(None) == "no graded games yet"


def test_paper_bets_one_per_game_only_above_break_even():
    g = bf.graded(bf.population(_rows(300)))
    b = bf.paper_bets(g)
    assert b["game_id"].is_unique
    assert (b["model_p"] > b["breakeven"]).all() and (b["ev"] > 0).all()
    for _, r in b.iterrows():
        assert np.isclose(r["breakeven"], market.implied(r["ml"]))


def test_render_keeps_stages_and_seasons_apart():
    d = pd.concat([_rows(40), _rows(20, seed=9, gp_home=4, gp_away=4)], ignore_index=True)
    d.loc[40:, "game_id"] = [str(900000 + i) for i in range(20)]
    text = bf.render(bf.population(d))
    assert "## 2026-27 · Games 1–9" in text and "## 2026-27 · Games 10+" in text
    assert "Primary, games 10+ 2026-27" in text
    assert "no graded games yet" in bf.render(bf.population(d.iloc[:0]))


def test_report_block_shows_the_record_at_one_se_beside_the_registered_reading():
    import report
    d = pd.concat([_rows(60), _rows(20, seed=9, gp_home=4, gp_away=4)], ignore_index=True)
    d.loc[60:, "game_id"] = [str(900000 + i) for i in range(20)]
    lines = report.blend_block(d)
    text = "\n".join(lines)
    assert lines[0].startswith("== BLEND SHADOW (blend_shrink40_v1")
    assert "+/- is 1 SE" in text and "95%, fixed at registration" in text
    assert "2026-27 Games 1–9" in text and "2026-27 Games 10+" in text
    g = bf.graded(bf.population(d))
    s = bf.scores(g[g["prediction_stage"] == "established"])
    assert f"{s['diffs']['blend − lock price']['ll']:+.4f}" in text
    assert bf.reading(s) in text
    full = report.report_text(d, ledger.empty())
    assert full.index("== BLEND SHADOW") < full.index("== RECONSTRUCTED")


def test_report_block_failure_never_stops_the_report(monkeypatch):
    import report

    def boom(native):
        raise RuntimeError("broken shadow")
    monkeypatch.setattr(bf, "population", boom)
    monkeypatch.setitem(sys.modules, "research.blend_forward", bf)
    import research
    monkeypatch.setattr(research, "blend_forward", bf, raising=False)
    assert report.blend_block(_rows(5)) == ["== BLEND SHADOW: unavailable (RuntimeError: broken shadow)"]
    assert "== BLEND SHADOW: unavailable" in report.report_text(_rows(5), ledger.empty())
