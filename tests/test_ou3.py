"""O/U model ou3 (CLAUDE.md amendment, registered 2026-11-06T21:10:26Z): ou2
plus both teams' days of rest; rows restart with a lock-time rescore."""
from datetime import datetime

import numpy as np
import pandas as pd

import analysis
import build_site
import ou3_rescore
import player_availability as pav
import totals
import validate_data_files as v
from public_site import PublicSite, rule_key
from test_ou_floors import _ou_ledger
from test_totals_od import _season

REPORT = dict(coef=dict(intercept=0.3, av_o=0.9, av_d=-1.2, av_m=0.3),
              sigma_early=18.0, sigma_late=18.5, n_games=1000, lead=300, seasons=[2023])


def _ou3(rest=0.75):
    return dict(tag=totals.TAG, report=REPORT, sigma_early=19.0, sigma_late=18.4,
                coef=dict(intercept=0.0, rel=1.0, form=0.0, b2b=0.0, phase=0.0, rest_tot=rest))


def _ou2():
    return dict(tag=totals.TAG_OU2, report=REPORT, sigma_early=19.0, sigma_late=18.4,
                coef=dict(intercept=0.0, rel=1.0, form=0.0, b2b=0.0, phase=0.0))


# ------------------------------------------------------------------ model --
def test_rest_days_and_the_feature():
    assert totals.rest_days(None, "2026-11-06") == 3                     # first game
    assert totals.rest_days("2026-11-05", "2026-11-06") == 0             # back-to-back
    assert totals.rest_days("2026-11-04", "2026-11-06") == 1
    assert totals.rest_days("2026-10-20", "2026-11-06") == 3             # capped
    logs = {2025: _season(2025, 1), 2026: _season(2026, 2)}
    s = totals.Season(logs[2026], totals.Season(logs[2025]).end())
    t = logs[2026]["AAA"]
    d, opp = t["date"].iloc[5], t["opp"].iloc[5]
    f = s.features("AAA", opp, d)
    o = logs[2026][opp]
    last_o = o["date"][o["date"] < d].max()                  # games strictly before the slate
    assert f["rest_tot"] == totals.rest_days(t["date"].iloc[4], d) + totals.rest_days(last_o, d)


def test_the_ou2_file_scores_and_tags_exactly_as_before():
    f = dict(rel=2.0, form=0.0, b2b=0.0, phase=0.0, rest_tot=4.0, lg_total=224.0,
             gp_home=12, gp_away=15)
    T, s = totals.predict(_ou2(), f)
    assert T == 226.0                                         # rest_tot is not in its terms
    assert totals.forecast(_ou2(), f, dict(av_o=0, av_d=0, av_m=0))[2] == totals.TAG_OU2
    assert totals.forecast(_ou2(), f, None)[2] == totals.TAG_NEUTRAL_OU2
    T3, _ = totals.predict(_ou3(), f)
    assert T3 == 226.0 + 0.75 * 4
    assert totals.forecast(_ou3(), f, dict(av_o=0, av_d=0, av_m=0))[2] == totals.TAG
    assert totals.forecast(_ou3(), f, None)[2] == totals.TAG_NEUTRAL
    assert totals.TAG.endswith(analysis.OU3_TAG_SUFFIX) and totals.TAG_NEUTRAL.endswith("_ou3")


def test_fit_adds_rest_and_takes_ou3_tags():
    rng = np.random.default_rng(5)
    n = 4000
    rows = pd.DataFrame(dict(rel=rng.normal(0, 8, n), form=rng.normal(0, 3, n),
                             b2b=rng.integers(0, 3, n).astype(float), phase=rng.uniform(0, 1, n),
                             rest_tot=rng.integers(0, 7, n).astype(float),
                             lg_total=225.0, gp_home=20, gp_away=20, season=2025))
    rows["total"] = rows["lg_total"] + 0.9 * rows["rel"] + 0.8 * rows["rest_tot"] + rng.normal(0, 18, n)
    m = totals.fit(rows)
    assert set(m["coef"]) == {"intercept", *totals.FEATURES} and m["tag"] == totals.TAG_NEUTRAL
    assert abs(m["coef"]["rest_tot"] - 0.8) < 0.3
    rows[list(totals.REPORT_TERMS)] = 0.0
    assert totals.fit_report(m, rows)["tag"] == totals.TAG
    assert totals.fit(rows, totals.FEATURES_OU2)["tag"] == totals.TAG_NEUTRAL_OU2


# ---------------------------------------------------------------- records --
def _rescore(g, p):
    return pd.DataFrame(dict(game_id=g["game_id"].astype(str), slate_date=g["slate_date"].astype(str),
                             lock_ou_snapshot_utc=g["lock_ou_snapshot_utc"], rescore_ou_tag=totals.TAG,
                             rescore_ou_total=[230.0] * len(g), rescore_ou_sigma=18.0,
                             rescore_ou_p_over=p, rescore_ou_rest_tot=2.0,
                             rescore_ou_report_utc="2026-10-29T20:45:00Z",
                             rescored_utc="2026-11-07T00:00:00Z"))[ou3_rescore.COLUMNS]


def test_with_ou3_reads_the_rescore_for_ou1_and_ou2_locks():
    g = _ou_ledger()
    g.loc[g.index[1], "lock_ou_tag"] = totals.TAG_OU2          # o1 ou1, o2 ou2, o3 ou1
    before = g.copy()
    rs = _rescore(g.iloc[:2], [0.30, 0.70])
    w = analysis.with_ou3(g, rs)
    pd.testing.assert_frame_equal(w[before.columns], before)    # native columns untouched
    assert list(w["ou3_p_over"].iloc[:2]) == [0.30, 0.70] and pd.isna(w["ou3_p_over"].iloc[2])
    assert set(w["ou3_tag"].iloc[:2]) == {totals.TAG} and list(w["ou3_line"].iloc[:2]) == [220.5, 220.5]
    native = analysis.with_ou3(g.assign(lock_ou_tag=totals.TAG), rs)   # native ou3: as recorded
    assert list(native["ou3_p_over"]) == [0.60, 0.40, 0.52]


def test_ou3_rows_restart_and_the_ou2_rows_close_at_activation(monkeypatch, tmp_path):
    g = _ou_ledger().assign(lock_ou_tag=totals.TAG_OU2)
    path = tmp_path / "rescore3.csv"
    _rescore(g.iloc[:2], [0.30, 0.70]).to_csv(path, index=False)
    monkeypatch.setattr(analysis, "OU3_RESCORE_PATH", path)
    monkeypatch.setattr(analysis, "OU_RESCORE_PATH", tmp_path / "none.csv")
    ou3, ou2 = analysis.OU3_TESTS[0], analysis.OU2_TESTS[0]
    assert list(analysis.ou_signals(g, ou3)["side"]) == ["under", "over"]   # rescored P flips both
    assert list(analysis.ou_picks(g, ou3)["result"]) == [0.0, 0.0]
    assert set(analysis.ou_signals(g, ou2)["game_id"]) == {"o1", "o2"}      # ou2 open before activation
    later = g.iloc[[2]].assign(game_id="o4", slate_date="2026-11-07", lock_ou_tag=totals.TAG,
                               lock_ou_p_over=0.7)
    h = pd.concat([g, later], ignore_index=True)
    assert analysis.ou3_activation_slate(h) == "2026-11-07"
    assert set(analysis.ou_signals(h, ou3)["game_id"]) == {"o1", "o2", "o4"}
    assert set(analysis.ou_signals(h, ou2)["game_id"]) == {"o1", "o2"}
    assert analysis.ou_signals(h.assign(slate_date="2026-11-08"), ou2).empty   # closed


def test_void_locks_never_enter_the_ou3_rows(monkeypatch, tmp_path):
    g = _ou_ledger().assign(lock_ou_tag=totals.TAG)
    void = next(iter(analysis.VOID_LOCKS))
    g["game_id"] = [void, "o2", "o3"]
    assert void not in set(analysis.ou_signals(g, analysis.OU3_TESTS[0])["game_id"])
    assert void not in set(ou3_rescore.pending(g.assign(lock_ou_tag=totals.TAG_OU2),
                                               ou3_rescore.load(tmp_path / "none.csv"))["game_id"])


def test_public_rows_follow_the_live_totals_model(monkeypatch, tmp_path):
    g = _ou_ledger().assign(lock_ou_tag=totals.TAG_OU2)
    g["pre_book"], g["pre_ou_line"], g["pre_ou_over"], g["pre_ou_under"] = "dk", 220.5, -110, -110
    path = tmp_path / "rescore3.csv"
    _rescore(g.iloc[:2], [0.30, 0.70]).to_csv(path, index=False)
    monkeypatch.setattr(analysis, "OU3_RESCORE_PATH", path)
    monkeypatch.setattr(totals, "load", lambda path=None: _ou2())
    site = PublicSite(build_site, g, "2026-10-30", "built")
    assert site.ou == "ou2" and "OU·L·ou2" in site.hypotheses and "OU·L·ou3" not in site.hypotheses
    assert "days of rest" not in site.guide_page()
    monkeypatch.setattr(totals, "load", lambda path=None: _ou3())        # Fit model wrote ou3
    site = PublicSite(build_site, g, "2026-10-30", "built")
    assert site.ou == "ou3" and "OU·L·ou3" in site.hypotheses and "OU·L·ou2" not in site.hypotheses
    assert rule_key("OU", True, "ou3") == "OU·E·L·ou3"
    ou = [r for r in site.records if r["hyp"].get("market") == "ou"]
    assert {r["row"].game_id for r in ou} == {"o1", "o2"} and {r["signal"].side for r in ou} == {"under", "over"}
    guide = site.guide_page()
    assert "both teams’ days of rest" not in guide
    note = "Picks through October 29, 2026 were scored with this model"
    assert note in site.ou2_clause()
    assert all(note not in page for page in (guide, site.results_page(), site.performance_page()))


# ---------------------------------------------------------------- rescore --
class _Archive:
    def __init__(self):
        self.cutoffs = []

    def latest_before(self, cutoff):
        self.cutoffs.append(cutoff)
        return cutoff.replace(minute=0, second=0), "report.pdf"

    def save(self):
        pass


def test_rescore_rebuilds_t_star_from_logs_before_the_slate(monkeypatch):
    seen = {}

    def terms(report, rdt, box, gid, home, away, date, tip, values, q):
        seen.update(box_max=box["date"].max())
        return dict(av_o=0.0, av_d=0.0, av_m=0.0, report="x")
    monkeypatch.setattr(pav, "parse_report", lambda path: pd.DataFrame())
    monkeypatch.setattr(pav, "totals_terms", terms)
    logs = {2025: _season(2025, 1), 2026: _season(2026, 2)}
    s = totals.Season(logs[2026], totals.Season(logs[2025]).end())
    t = logs[2026]["AAA"]
    i = int(np.where(t["home"])[0][4])
    slate = pd.Timestamp(t["date"].iloc[i]).strftime("%Y-%m-%d")
    row = pd.Series(dict(game_id="o1", slate_date=slate, home="AAA", away=t["opp"].iloc[i],
                         tip_utc=f"{slate}T23:30:00Z", lock_ou_snapshot_utc=f"{slate}T18:32:40Z",
                         lock_ou_total=999.0, lock_ou_line=222.5, gp_home=3, gp_away=4, season=2026))
    box = pd.DataFrame(dict(date=pd.to_datetime([pd.Timestamp(slate) - pd.Timedelta(days=1),
                                                  pd.Timestamp(slate)])))
    arch = _Archive()
    out = ou3_rescore.rescore_row(row, _ou3(), s, box, {}, None, arch)
    f = s.features("AAA", row["away"], slate)
    want = totals.forecast(_ou3(), f, dict(av_o=0.0, av_d=0.0, av_m=0.0))[0]
    assert abs(out["rescore_ou_total"] - round(want, 2)) < 1e-9          # not the recorded 999
    assert out["rescore_ou_rest_tot"] == f["rest_tot"] and out["rescore_ou_tag"] == totals.TAG
    assert seen["box_max"] < pd.Timestamp(slate)                          # strictly before the slate
    assert arch.cutoffs == [datetime(*pd.Timestamp(slate).timetuple()[:3], 14, 32, 40)]


def test_rescore_waits_for_an_ou3_model_and_never_raises(tmp_path):
    g = _ou_ledger().assign(season=2027, lock_ou_tag=totals.TAG_OU2)
    path = tmp_path / "r.csv"
    assert len(ou3_rescore.pending(g, ou3_rescore.load(path))) == 3
    assert ou3_rescore.update(g, log=lambda m: None, path=path, model=_ou2()) == 0   # not ou3 yet
    assert not path.exists()
    assert ou3_rescore.update("not a frame", log=lambda m: None, path=path) == 0      # never raises
    assert ou3_rescore.pending(g.assign(lock_ou_tag=totals.TAG), ou3_rescore.load(path)).empty


def test_validation_of_the_ou3_rescore_file(tmp_path):
    g = _ou_ledger()
    led = tmp_path / "ledger.csv"
    g.to_csv(led, index=False)
    path = tmp_path / "r.csv"
    good = _rescore(g.iloc[:2], [0.3, 0.7])
    good.to_csv(path, index=False)
    assert v.check_ou3_rescore(str(path), str(led)) == []
    for bad, msg in ((good.assign(rescore_ou_report_utc="2026-10-29T22:00:00Z"), "later than its lock"),
                     (good.assign(rescore_ou_p_over=1.5), "outside [0,1]"),
                     (good.assign(rescore_ou_tag="x_ou2"), "not ending _ou3"),
                     (good.assign(rescore_ou_rest_tot=7), "days of rest"),
                     (good.assign(lock_ou_snapshot_utc="2026-10-29T20:00:00Z"), "native ou1 / ou2 lock"),
                     (pd.concat([good, good.iloc[:1]]), "duplicate")):
        bad.to_csv(path, index=False)
        assert any(msg in e for e in v.check_ou3_rescore(str(path), str(led))), msg
