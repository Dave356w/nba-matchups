"""O/U model ou2 (CLAUDE.md amendment, registered 2026-11-03T23:05:00Z)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import analysis
import build_site
import ou_rescore
import player_availability as pav
import totals
import validate_data_files as v
from public_site import PublicSite, rule_key
from test_ou_floors import _ou_ledger

REPORT = dict(coef=dict(intercept=0.3, av_o=0.9, av_d=-1.2, av_m=0.3),
              sigma_early=18.0, sigma_late=18.5, n_games=1000, lead=300, seasons=[2023])


def _model(report=True):
    m = dict(tag=totals.TAG_OU2 if report else totals.TAG_NEUTRAL_OU2,
             coef=dict(intercept=0.0, rel=1.0, form=0.0, b2b=0.0, phase=0.0),
             sigma_early=19.0, sigma_late=18.4)
    if report:
        m["report"] = REPORT
    return m


F = dict(rel=2.0, form=0.0, b2b=0.0, phase=0.0, lg_total=224.0, gp_home=12, gp_away=15)


# ------------------------------------------------------------------ model --
def test_forecast_adds_the_report_stage_or_stays_neutral():
    T, s, tag = totals.forecast(_model(), F, dict(av_o=-2.0, av_d=1.0, av_m=0.5))
    assert tag == totals.TAG_OU2 and s == 18.5
    assert abs(T - (226.0 + 0.3 - 1.8 - 1.2 + 0.15)) < 1e-9
    T0, s0, tag0 = totals.forecast(_model(), F, None)          # no report terms
    assert (T0, s0, tag0) == (226.0, 18.4, totals.TAG_NEUTRAL_OU2)
    assert totals.forecast(_model(), F, dict(av_o=np.nan, av_d=0, av_m=0))[2] == totals.TAG_NEUTRAL_OU2
    assert totals.forecast(_model(report=False), F, dict(av_o=1, av_d=1, av_m=1))[2] == totals.TAG_NEUTRAL_OU2
    early = dict(F, gp_home=4)
    assert totals.forecast(_model(), early, dict(av_o=0, av_d=0, av_m=0))[1] == 18.0
    assert totals.TAG_OU2.endswith(analysis.OU2_TAG_SUFFIX) and totals.TAG_NEUTRAL_OU2.endswith("_ou2")


def test_adjust_starts_from_a_recorded_base_total():
    T, s, tag = totals.adjust(_model(), 230.0, 5, dict(av_o=1.0, av_d=0.0, av_m=0.0))
    assert (round(T, 6), s, tag) == (231.2, 18.0, totals.TAG_OU2)
    assert totals.adjust(_model(), 230.0, 12, None) == (230.0, 18.4, totals.TAG_NEUTRAL_OU2)


def test_fit_report_recovers_the_stage_on_the_base_residual():
    rng = np.random.default_rng(5)
    n = 6000
    rows = pd.DataFrame(dict(rel=rng.normal(0, 8, n), form=0.0, b2b=0.0, phase=0.0, lg_total=225.0,
                             gp_home=20, gp_away=20, season=2024,
                             av_o=rng.normal(-1, 3, n), av_d=rng.normal(0, 2, n), av_m=rng.normal(0, 1, n)))
    rows["total"] = 225 + rows["rel"] + 0.5 + 1.0 * rows["av_o"] - 1.3 * rows["av_d"] \
        + rng.normal(0, 17, n)
    m = totals.fit_report(_model(report=False), rows)
    c = m["report"]["coef"]
    assert m["tag"] == totals.TAG_OU2 and abs(c["av_o"] - 1.0) < 0.15 and abs(c["av_d"] + 1.3) < 0.2
    assert abs(m["report"]["sigma_late"] - 17) < 0.6 and m["report"]["lead"] == 300


def test_add_totals_records_the_route_tag():
    r = dict(game_id="g", home="AAA", away="BBB", slate_date="2026-11-04", pre_ou_line=226.0)

    class S:
        def features(self, h, a, d):
            return dict(F)
    build_site.add_totals(r, _model(), S(), dict(av_o=0.0, av_d=0.0, av_m=0.0))
    assert r["ou_tag"] == totals.TAG_OU2 and abs(r["ou_total"] - 226.3) < 1e-9
    build_site.add_totals(r, _model(), S(), None)
    assert r["ou_tag"] == totals.TAG_NEUTRAL_OU2 and r["ou_total"] == 226.0


def test_totals_terms_need_a_covering_report():
    box = pd.DataFrame(columns=pav.BOX_COLUMNS)
    rep = pd.DataFrame(columns=["game_date", "matchup", "team", "player", "status"])
    assert pav.totals_terms(None, None, box, "g", "AAA", "BBB", "2026-11-04", None, {}, None) is None
    assert pav.totals_terms(rep, datetime(2026, 11, 4, 12), box, "g", "AAA", "BBB", "2026-11-04",
                            None, {"OBPM": {}, "DBPM": {}}, None) is None


# ---------------------------------------------------------------- records --
def _rescore(g, p):
    return pd.DataFrame(dict(game_id=g["game_id"].astype(str), slate_date=g["slate_date"].astype(str),
                             lock_ou_snapshot_utc=g["lock_ou_snapshot_utc"], rescore_ou_tag=totals.TAG_OU2,
                             rescore_ou_total=[230.0] * len(g), rescore_ou_sigma=18.0,
                             rescore_ou_p_over=p, rescore_ou_report_utc="2026-10-29T20:45:00Z",
                             rescored_utc="2026-11-04T00:00:00Z"))[ou_rescore.COLUMNS]


def test_with_ou2_reads_the_rescore_for_ou1_locks_and_never_changes_native_columns():
    g = _ou_ledger()                                         # ou1 locks, P 0.60 / 0.40 / 0.52
    before = g.copy()
    rs = _rescore(g.iloc[:2], [0.30, 0.70])
    w = analysis.with_ou2(g, rs)
    pd.testing.assert_frame_equal(w[before.columns], before)
    assert list(w["ou2_p_over"].iloc[:2]) == [0.30, 0.70] and pd.isna(w["ou2_p_over"].iloc[2])
    assert list(w["ou2_line"].iloc[:2]) == [220.5, 220.5] and pd.isna(w["ou2_line"].iloc[2])
    assert set(w["ou2_tag"].iloc[:2]) == {totals.TAG_OU2} and list(w["ou2_total"].iloc[:2]) == [230.0, 230.0]
    native = analysis.with_ou2(g.assign(lock_ou_tag=totals.TAG_OU2), rs)   # native ou2: as recorded
    assert list(native["ou2_p_over"]) == [0.60, 0.40, 0.52]


def test_ou2_rows_read_the_rescore_and_the_ou1_rows_close_at_activation(monkeypatch, tmp_path):
    g = _ou_ledger()
    path = tmp_path / "rescore.csv"
    _rescore(g.iloc[:2], [0.30, 0.70]).to_csv(path, index=False)
    monkeypatch.setattr(analysis, "OU_RESCORE_PATH", path)
    ou2, ou1 = analysis.OU2_TESTS[0], analysis.OU_TESTS[0]
    sig = analysis.ou_signals(g, ou2)
    assert list(sig["side"]) == ["under", "over"]            # the rescored P flips both sides
    d = analysis.ou_picks(g, ou2)
    assert list(d["result"]) == [0.0, 0.0] and list(d["line"]) == [220.5, 220.5]
    assert set(analysis.ou_signals(g, ou1)["game_id"]) == {"o1", "o2"}   # ou1 before activation
    # a later slate locks natively under ou2: activation; ou1 closes before it
    later = g.iloc[[2]].assign(game_id="o4", slate_date="2026-11-05", lock_ou_tag=totals.TAG_OU2,
                               lock_ou_p_over=0.7)
    h = pd.concat([g, later], ignore_index=True)
    assert analysis.ou2_activation_slate(h) == "2026-11-05"
    assert set(analysis.ou_signals(h, ou2)["game_id"]) == {"o1", "o2", "o4"}
    assert set(analysis.ou_signals(h, ou1)["game_id"]) == {"o1", "o2"}
    assert analysis.ou_signals(h.assign(slate_date="2026-11-06"), ou1).empty


def test_public_pages_show_only_the_ou2_rows_and_the_dated_line(monkeypatch, tmp_path):
    from conftest import pin_ou2_model
    pin_ou2_model(monkeypatch)                      # before ou3's activation
    g = _ou_ledger()
    g["pre_book"], g["pre_ou_line"], g["pre_ou_over"], g["pre_ou_under"] = "dk", 220.5, -110, -110
    path = tmp_path / "rescore.csv"
    _rescore(g.iloc[:2], [0.30, 0.70]).to_csv(path, index=False)
    monkeypatch.setattr(analysis, "OU_RESCORE_PATH", path)
    site = PublicSite(build_site, g, "2026-10-30", "built")
    assert rule_key("OU", True) == "OU·E·L·ou2" and rule_key("OU", False) == "OU·L·ou2"
    assert "OU·L" not in site.hypotheses and "OU·L·ou2" in site.hypotheses
    ou = [r for r in site.records if r["hyp"].get("market") == "ou"]
    assert {r["row"].game_id for r in ou} == {"o1", "o2"} and {r["signal"].side for r in ou} == {"under", "over"}
    note = "Picks through October 29, 2026 were scored with this model"
    assert note in site.ou2_clause() and note not in site.guide_page()
    assert note not in site.results_page() and note not in site.performance_page()   # owner: guide only
    monkeypatch.setattr(analysis, "OU_RESCORE_PATH", tmp_path / "none.csv")
    assert "Picks through" not in PublicSite(build_site, g, "2026-10-30", "built").guide_page()


# ---------------------------------------------------------------- rescore --
class _Archive:
    def __init__(self):
        self.cutoffs = []

    def latest_before(self, cutoff):
        self.cutoffs.append(cutoff)
        return cutoff.replace(minute=0, second=0), "report.pdf"

    def save(self):
        pass


def test_rescore_reads_only_lock_time_information(monkeypatch):
    seen = {}

    def terms(report, rdt, box, gid, home, away, date, tip, values, q):
        seen.update(box_max=box["date"].max(), rdt=rdt)
        return dict(av_o=-1.0, av_d=0.0, av_m=0.0, report="x")
    monkeypatch.setattr(pav, "parse_report", lambda path: pd.DataFrame())
    monkeypatch.setattr(pav, "totals_terms", terms)
    row = pd.Series(dict(game_id="o1", slate_date="2026-10-30", home="AAA", away="BBB",
                         tip_utc="2026-10-30T23:30:00Z", lock_ou_snapshot_utc="2026-10-30T18:32:40Z",
                         lock_ou_total=224.0, lock_ou_line=222.5, gp_home=3, gp_away=4, season=2027))
    box = pd.DataFrame(dict(date=pd.to_datetime(["2026-10-28", "2026-10-30", "2026-10-31"])))
    arch = _Archive()
    out = ou_rescore.rescore_row(row, _model(), box, {}, None, arch)
    assert arch.cutoffs == [datetime(2026, 10, 30, 14, 32, 40)]          # the lock, in ET
    assert seen["box_max"] == pd.Timestamp("2026-10-28")                 # strictly before the slate
    assert out["rescore_ou_tag"] == totals.TAG_OU2 and abs(out["rescore_ou_total"] - 223.4) < 1e-9
    assert out["rescore_ou_report_utc"] == "2026-10-30T18:00:00Z"
    assert pd.Timestamp(out["rescore_ou_report_utc"]) <= pd.Timestamp(out["lock_ou_snapshot_utc"])


def test_rescore_is_written_once_and_never_raises(monkeypatch, tmp_path):
    g = _ou_ledger().assign(season=2027)
    path = tmp_path / "r.csv"
    first = _rescore(g.iloc[:1], [0.25])
    first.to_csv(path, index=False)
    assert list(ou_rescore.pending(g, ou_rescore.load(path))["game_id"]) == ["o2", "o3"]
    monkeypatch.setattr(pav, "fetch_box", lambda y: pd.DataFrame(dict(date=pd.to_datetime([]))))
    monkeypatch.setattr(pav, "stat_values", lambda box, y: {})
    monkeypatch.setattr(ou_rescore, "rescore_row",
                        lambda row, *a: dict(_rescore(g[g.game_id == row["game_id"]], [0.9]).iloc[0]))
    assert ou_rescore.update(g, log=lambda m: None, path=path, model=_model(), archive=_Archive()) == 2
    out = ou_rescore.load(path)
    assert list(out["game_id"]) == ["o1", "o2", "o3"] and out.iloc[0]["rescore_ou_p_over"] == 0.25
    assert ou_rescore.update(g, log=lambda m: None, path=path, model=_model()) == 0      # nothing pending
    assert ou_rescore.update(g.assign(game_id=["x1", "x2", "x3"]), log=lambda m: None,
                             path=tmp_path / "bad" / "r.csv", model=dict(tag="t")) == 0   # no report stage
    assert ou_rescore.update("not a frame", log=lambda m: None, path=path) == 0              # never raises
    native_ou2 = g.assign(lock_ou_tag=totals.TAG_OU2)
    assert ou_rescore.pending(native_ou2, ou_rescore.load(tmp_path / "none.csv")).empty


def test_validation_of_the_rescore_file(tmp_path):
    g = _ou_ledger()
    led = tmp_path / "ledger.csv"
    g.to_csv(led, index=False)
    path = tmp_path / "r.csv"
    good = _rescore(g.iloc[:2], [0.3, 0.7])
    good.to_csv(path, index=False)
    assert v.check_ou_rescore(str(path), str(led)) == []
    for bad, msg in ((good.assign(rescore_ou_report_utc="2026-10-29T22:00:00Z"), "later than its lock"),
                     (good.assign(rescore_ou_p_over=1.5), "outside [0,1]"),
                     (good.assign(rescore_ou_tag="x_ou1"), "not ending _ou2"),
                     (good.assign(lock_ou_snapshot_utc="2026-10-29T20:00:00Z"), "native ou1 lock"),
                     (pd.concat([good, good.iloc[:1]]), "duplicate")):
        bad.to_csv(path, index=False)
        assert any(msg in e for e in v.check_ou_rescore(str(path), str(led))), msg
