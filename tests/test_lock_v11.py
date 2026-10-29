"""Five-hour lock and model v11 (CLAUDE.md amendments of 2026-10-28)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import analysis
import backfill_history as bf
import build_site
import ledger
import player_availability as pav
from public_site import PublicSite
from test_analysis_and_pages import _ats_rows

TIP = "2026-11-20T00:30Z"
TIP_DT = datetime(2026, 11, 20, 0, 30, tzinfo=timezone.utc)


def row(**kw):
    r = dict(game_id="401", slate_date="2026-11-19", season=2027, tip_utc=TIP,
             model_tag="t", home="GSW", away="UTA", delta=5.0, p_home=0.62,
             lean="GSW", p_lean=0.62, pre_book="dk", pre_home_ml=-150,
             pre_away_ml=130, pre_q_home=0.585, pre_spread=-3.5,
             pre_home_spread_odds=-110, pre_away_spread_odds=-110)
    r.update(kw)
    return r


# ---------------------------------------------------------------- ledger --
def test_lock_is_the_first_priced_snapshot_within_five_hours_written_once():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()],
                                      now=TIP_DT - timedelta(hours=18))
    assert led.iloc[0]["first_p_home"] == 0.62
    assert led[ledger.LOCK_COLUMNS].isna().all(axis=None)
    # 5 h 1 min before tip: still outside the window
    led, _, _ = ledger.upsert_pregame(led, [row(p_home=0.55)],
                                      now=TIP_DT - timedelta(hours=5, minutes=1))
    assert pd.isna(led.iloc[0]["lock_snapshot_utc"])
    # no price inside the window: no lock yet
    led, _, _ = ledger.upsert_pregame(led, [row(p_home=0.57, pre_q_home=np.nan)],
                                      now=TIP_DT - timedelta(hours=4, minutes=30))
    assert pd.isna(led.iloc[0]["lock_snapshot_utc"])
    led, _, _ = ledger.upsert_pregame(led, [row(p_home=0.66, pre_spread=-4.5, route="avail",
                                                report_utc="2026-11-19T19:30:00Z")],
                                      now=TIP_DT - timedelta(hours=4))
    r = led.iloc[0]
    assert (r["lock_p_home"], r["lock_spread"], r["lock_route"], r["lock_book"]) == \
        (0.66, -4.5, "avail", "dk")
    assert r["lock_snapshot_utc"] == "2026-11-19T20:30:00Z"
    assert r["first_p_home"] == 0.62                     # the first snapshot is kept
    lock = led[ledger.LOCK_COLUMNS].copy()
    led, _, _ = ledger.upsert_pregame(led, [row(p_home=0.30, pre_spread=2.5)],
                                      now=TIP_DT - timedelta(hours=1))
    pd.testing.assert_frame_equal(led[ledger.LOCK_COLUMNS], lock)
    assert led.iloc[0]["p_home"] == 0.30                 # the forecast still refreshes
    # grading never writes a lock column
    assert not set(ledger.LOCK_COLUMNS) & set(ledger.GRADE_COLUMNS)


def test_a_row_first_seen_inside_the_window_gets_first_and_lock_together():
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()],
                                      now=TIP_DT - timedelta(hours=2))
    r = led.iloc[0]
    assert r["first_snapshot_utc"] == r["lock_snapshot_utc"] == r["seen_utc"]


def test_validator_flags_a_lock_outside_its_window(tmp_path):
    import validate_data_files as v
    led, _, _ = ledger.upsert_pregame(ledger.empty(), [row()],
                                      now=TIP_DT - timedelta(hours=8))
    led, _, _ = ledger.upsert_pregame(led, [row()], now=TIP_DT - timedelta(hours=3))
    p = tmp_path / "ok.csv"
    ledger.save(led, str(p))
    assert v.check(str(p)) == []
    led.loc[0, "lock_snapshot_utc"] = "2026-11-19T12:00:00Z"     # 12.5 h before tip
    ledger.save(led, str(p))
    assert any("outside its window" in e for e in v.check(str(p)))
    led.loc[0, "lock_snapshot_utc"] = "2026-11-19T22:00:00Z"
    led.loc[0, "lock_book"] = "nobook"
    ledger.save(led, str(p))
    assert any("lock_book" in e for e in v.check(str(p)))


# -------------------------------------------------------------- analysis --
def _two_slates():
    """Graded games 10+: three on a first-snapshot slate, three on the lock's
    first slate (lock P on the other side of the first P)."""
    g = _ats_rows(6)
    g["slate_date"] = ["2026-11-18"] * 3 + ["2026-11-19"] * 3
    g["first_book"] = g["pre_book"] = "dk"
    g["first_p_home"], g["p_home"] = 0.8, 0.8
    g["first_home_ml"], g["first_away_ml"], g["first_q_home"] = -150, 130, 0.58
    late = g.index[3:]
    for c in ledger.TEXT_COLUMNS:
        g[c] = g[c].astype(object)
    g.loc[late, "lock_snapshot_utc"] = "2026-11-19T21:00:00Z"
    g.loc[late, "lock_model_tag"] = build_site.MODEL_TAG_V10
    g.loc[late, "lock_p_home"] = 0.2
    g.loc[late, "lock_book"] = "dk"
    g.loc[late, ["lock_home_ml", "lock_away_ml", "lock_q_home"]] = [-150, 130, 0.58]
    g.loc[late, ["lock_spread", "lock_home_spread_odds", "lock_away_spread_odds"]] = \
        [-5.5, -110, -110]
    return g


def test_first_rows_close_at_the_locks_first_slate_and_lock_rows_read_the_lock():
    g = _two_slates()
    assert analysis.lock_activation_slate(g) == "2026-11-19"
    assert analysis.lock_activation_slate(g.drop(columns="lock_snapshot_utc")) is None
    h5, h5l = (next(h for h in analysis.HYPOTHESES if h["key"] == k) for k in ("H5", "H5·L"))
    assert (h5l["native"], h5l["sigma"], h5l["min_edge"], h5l["hindsight"]) == \
        ("lock", 13.5, 1.0, "open")
    first = analysis.hypothesis_picks(g, h5, "first")
    lock = analysis.hypothesis_picks(g, h5l, "lock")
    assert set(first.game_id) == set(g.game_id[:3])       # closed at the lock's slate
    assert set(lock.game_id) == set(g.game_id[3:])
    assert (first["side"] == "HOM").all() and (lock["side"] == "AWY").all()
    # signals and grades agree, and an ungraded slate cannot reopen the first rows
    sig = analysis.hypothesis_signals(g.assign(home_won=np.nan), h5, "first")
    assert set(sig.game_id) == set(first.game_id)
    # before any lock exists the first-snapshot rows stay open
    open_ = analysis.hypothesis_picks(g.drop(columns=ledger.LOCK_COLUMNS), h5, "first")
    assert set(open_.game_id) == set(g.game_id)
    # every active test has its lock twin with identical rules and stages
    active = {h["key"]: h for h in analysis.active_hypotheses()}
    for key, h in active.items():
        if h.get("lock"):
            parent = active[h["parent"]]
            for f in ("market", "early", "favourite", "min_edge", "sigma", "requires_v9"):
                assert h.get(f) == parent.get(f)
    assert analysis.ats_card(g.iloc[4])["hyp"]["key"] == "H5·L"
    # hindsight groupings are not duplicated by the lock rows
    assert all("·L" not in label for label, _ in analysis.hypothesis_groups())


def test_lock_rows_clv_and_status():
    g = _two_slates()
    g["close_spread"] = -7.5                     # the line moved toward the home side
    g["close_home_spread_odds"] = g["close_away_spread_odds"] = -110
    h5, h5l = (next(h for h in analysis.HYPOTHESES if h["key"] == k) for k in ("H5", "H5·L"))
    first, lock = analysis.hypothesis_clv(g, h5), analysis.hypothesis_clv(g, h5l)
    assert first["n"] == lock["n"] == 3 and first["unit"] == "pts"
    assert first["mean"] == 2.0 and lock["mean"] == -2.0  # home bets gained, away bets lost
    st = analysis.lock_status(g, now="2026-11-21T00:00Z")
    assert (st["activation"], st["locked"], st["missed"]) == ("2026-11-19", 3, 0)
    assert abs(st["lead_median_h"] - 3.5) < 1e-9


# ---------------------------------------------------------------- pages --
def _page(g, today):
    return PublicSite(build_site, g, today, "built")


def test_today_marks_locked_picks_and_says_when_the_rest_lock():
    g = _two_slates()
    g["home_won"] = np.nan
    g.loc[g.index[3:], ["home_pts", "away_pts"]] = np.nan
    g.loc[g.index[5], ledger.LOCK_COLUMNS] = np.nan        # not yet in its window
    g.loc[g.index[5], "tip_utc"] = "2099-01-01T00:30Z"
    site = _page(g, "2026-11-19")
    locked = site.card(g.iloc[3])
    assert "Tracked picks · 5 h lock" in locked
    waiting = site.card(g.iloc[5])
    assert "Forecast only" in waiting and "lock about 5 hours before tip" in waiting
    # a closed first-snapshot slate keeps its picks, labelled as such
    assert "Tracked picks · first snapshot" in site.card(g.iloc[0])
    assert all(r["lock"] for r in site.records if r["row"].slate_date == "2026-11-19")


def test_results_and_performance_keep_the_two_records_apart():
    g = _two_slates()
    site = _page(g, "2026-11-20")
    res = site.results_page()
    assert "data-basis='lock'" in res and "data-basis='first'" in res
    assert "First snapshot · closed" in res
    perf = site.performance_page()
    assert "DraftKings · 5 h lock" in perf
    assert "first snapshot · closed, slates before November 19, 2026" in perf
    guide = site.guide_page()
    assert "within 5 hours of tip" in guide and "closed on November 19, 2026" in guide
    # before any lock: the first-snapshot record is still open
    pre = _page(g.assign(**{c: np.nan for c in ledger.LOCK_COLUMNS}), "2026-11-20")
    assert "until the 5 h lock begins" in pre.performance_page()


# ------------------------------------------------------------------ v11 --
def test_q_rule_weights_doubtful_questionable_and_probable_by_lead():
    rates = {"1080": {"Doubtful": 0.05, "Questionable": 0.5, "Probable": 0.87},
             "300": {"Doubtful": 0.02, "Questionable": 0.48, "Probable": 0.91},
             "30": {"Doubtful": 0.0, "Questionable": 0.51, "Probable": 0.92}}
    assert pav.lead_rates(rates, 1000) == rates["1080"]
    assert pav.lead_rates(rates, 270) == rates["300"]
    assert pav.lead_rates(rates, 10) == rates["30"]
    r = rates["300"]
    assert [pav.q_status_p(s, r) for s in ("Out", "Doubtful", "Questionable",
                                           "Probable", "Available")] == \
        [0.0, 0.02, 0.48, 0.91, 1.0]
    box = pd.DataFrame(dict(game_id="g0", team=["GSW", "GSW"], player_id=["1", "2"],
                            name=["Stephen Curry", "Draymond Green"]))
    rep = pd.DataFrame(dict(game_date="11/19/2026", matchup="UTA@GSW", team="Golden State Warriors",
                            player=["Curry, Stephen", "Green, Draymond"],
                            status=["Questionable", "Doubtful"]))
    od = pav.od_present(rep, box, "g1", {"GSW", "UTA"}, "2026-11-19")
    q = pav.od_present(rep, box, "g1", {"GSW", "UTA"}, "2026-11-19", r)
    assert od == {("g1", "1"): 1.0, ("g1", "2"): 0.0}
    assert q == {("g1", "1"): 0.48, ("g1", "2"): 0.02}


def test_v11_tags_routes_and_early_rows():
    base = dict(version=11, features=["delta"], fade_rho=0.5, sigma=13.0, intercept=0.0,
                coef=[1.0], kind="margin")
    assert build_site.is_v11(base) == build_site.is_v10(base) or not build_site.is_v9(base)
    tags = [build_site.MODEL_TAG_V11, build_site.MODEL_TAG_V11_AVAIL,
            build_site.MODEL_TAG_V11_NEUTRAL, build_site.MODEL_TAG_V11_NEUTRAL_AVAIL]
    assert all(t.endswith("_v11") for t in tags) and len(set(tags)) == 4
    assert build_site.MODEL_TAG_V11_AVAIL in build_site.AVAIL_TAGS
    assert build_site.MODEL_TAG_V11_NEUTRAL_AVAIL in build_site.NEUTRAL_TAGS
    assert analysis.EARLY_TAG_SUFFIXES == ("_v9", "_v10", "_v11")
    g = _ats_rows(3).assign(gp_home=3, gp_away=3, first_p_home=0.8,
                            first_model_tag=[build_site.MODEL_TAG_V10,
                                             build_site.MODEL_TAG_V11_AVAIL, "x_v8"])
    h5e = next(h for h in analysis.HYPOTHESES if h["key"] == "H5·E")
    assert set(analysis.hypothesis_picks(g, h5e, "first").game_id) == set(g.game_id[:2])
    assert build_site.model_version(tags) == "v11"


def test_live_terms_use_the_q_rule_only_with_a_rate_table():
    class Live(pav.LiveAvailability):
        def __init__(self):
            self.report_time = datetime(2026, 11, 19, 14, 30)   # ET, 5 h before tip
            self.report = "r"
            self.box, self.value = None, {}
    seen = {}
    orig = (pav.covers, pav.report_coverage, pav.od_present, pav.upcoming_terms)
    try:
        pav.covers = lambda *a: True
        pav.report_coverage = lambda *a: None
        pav.od_present = lambda rep, box, gid, teams, slate, rates=None: seen.update(rates=rates) or {}
        pav.upcoming_terms = lambda *a: (0.0, 0.0, 0.0)
        table = {"1080": {"Questionable": 0.5}, "300": {"Questionable": 0.48}}
        Live().terms("g", "GSW", "UTA", "2026-11-19")
        assert seen["rates"] is None                          # v10: od
        Live().terms("g", "GSW", "UTA", "2026-11-19", tip_utc=TIP, q_rates=table)
        assert seen["rates"] == {"Questionable": 0.48}        # 300 min before tip
    finally:
        pav.covers, pav.report_coverage, pav.od_present, pav.upcoming_terms = orig


def test_backfill_refits_read_the_od_rows_while_scoring_reads_q():
    model = dict(features=["av_bpm"], intercept=0.0, coef=[1.0], sigma=12.0, kind="margin")
    train = pd.DataFrame(dict(av_bpm=[0.0, 1.0, -1.0], margin=[0.0, 2.0, -2.0]))
    te = pd.DataFrame(dict(date=pd.to_datetime(["2026-11-01", "2026-11-02"]),
                           av_bpm=[5.0, 1.0], margin=[50.0, 1.0]))
    od = te.assign(av_bpm=[0.0, 1.0], margin=[0.0, 1.0])
    seen = []
    orig = pav.refit
    try:
        pav.refit = lambda m, extra, tr: seen.append(extra["av_bpm"].tolist()) or m
        bf.slate_refits(model, train, te, te[["av_bpm"]])
        assert seen[-1] == [5.0]                 # v10: the scored rows refit
        bf.slate_refits(model, train, te, te[["av_bpm"]], od)
        assert seen[-1] == [0.0]                 # v11: the od rows refit
    finally:
        pav.refit = orig


def test_how_it_works_describes_v11_with_its_dated_line(monkeypatch):
    native = pd.DataFrame({"basis": ["native"] * 2,
                           "first_model_tag": [build_site.MODEL_TAG_V10, build_site.MODEL_TAG_V11],
                           "first_snapshot_utc": ["2026-10-27T18:00Z", "2026-10-30T18:00Z"]})
    site = PublicSite.__new__(PublicSite)
    site.ui, site.native, site.today = build_site, native, "2026-10-31"
    monkeypatch.setattr(build_site, "ACTIVE_TAGS", [build_site.MODEL_TAG_V11])
    note = site.model_note()
    assert "questionable, doubtful or probable" in note and "October 30, 2026" in note
    monkeypatch.setattr(build_site, "ACTIVE_TAGS", [build_site.MODEL_TAG_V10])
    assert "questionable" not in site.model_note()


def test_only_a_missing_object_is_remembered_as_a_miss(tmp_path, monkeypatch):
    import io
    import urllib.error

    def http_error(code, ctype):
        return urllib.error.HTTPError("u", code, "x", {"Content-Type": ctype}, io.BytesIO(b""))
    calls = []

    def fake(answers):
        def urlopen(req, timeout=30):
            calls.append(req.full_url)
            a = answers.pop(0) if len(answers) > 1 else answers[0]
            raise a
        return urlopen
    monkeypatch.setattr(pav.time, "sleep", lambda s: None)
    arch = pav.ReportArchive(str(tmp_path))
    # a 404 or the CDN's XML AccessDenied: never published
    for err in (http_error(404, "text/html"), http_error(403, "application/xml")):
        monkeypatch.setattr(pav.urllib.request, "urlopen", fake([err]))
        assert arch._http("https://x/u.pdf") is None
    # throttling (an HTML 403) or a dropped connection: retried, then raised
    monkeypatch.setattr(pav.urllib.request, "urlopen", fake([http_error(403, "text/html")]))
    calls.clear()
    try:
        arch._http("https://x/u.pdf", tries=3)
        raise AssertionError("no TransientFetch")
    except pav.TransientFetch:
        assert len(calls) == 3
    dt = datetime(2024, 1, 5, 17, 0)
    try:
        arch.get(dt)
    except pav.TransientFetch:
        pass
    assert not arch.misses                       # nothing remembered as missing


def test_q_rates_refuse_an_incomplete_archive(monkeypatch):
    monkeypatch.setattr(pav, "fetch_box", lambda t, cache_dir=None: pd.DataFrame())
    monkeypatch.setattr(pav, "fetch_tips", lambda t, cache_dir=None: {})
    monkeypatch.setattr(pav, "game_statuses", lambda box, tips, arch, lead: (
        pd.DataFrame(columns=["status", "played"]), dict(games=100, with_report=90)))

    class Arch:
        def prefetch(self, cut): return 0
        def save(self): pass
    try:
        pav.q_rates([2024], leads=(300,), archive=Arch())
        raise AssertionError("no refusal")
    except SystemExit as e:
        assert "only 90 of 100" in str(e)


def test_q_rate_file_round_trips_and_refuses_other_seasons(tmp_path):
    table = {str(x): {"Doubtful": 0.03, "Questionable": 0.49, "Probable": 0.9}
             for x in pav.Q_LEADS}
    path = str(tmp_path / "q_rates.json")
    pav.save_q_rates(table, [2023, 2024, 2025, 2026], path)
    assert pav.load_q_rates(path, [2023, 2024, 2025, 2026]) == table
    try:
        pav.load_q_rates(path, [2024, 2025, 2026])
        raise AssertionError("no refusal")
    except SystemExit as e:
        assert "rerun the q-rates command" in str(e)


def test_committed_q_rate_table_covers_the_fit_seasons_and_leads():
    rates = pav.load_q_rates(pav.Q_RATES_FILE, [2023, 2024, 2025, 2026])
    for lead in pav.Q_LEADS:
        r = rates[str(lead)]
        assert 0.4 < r["Questionable"] < 0.6 and r["Doubtful"] < 0.1 < 0.8 < r["Probable"]
