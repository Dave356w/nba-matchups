"""Model-family reset (CLAUDE.md amendment, registered 2026-10-29T13:58:40Z)."""
import numpy as np
import pandas as pd

import analysis
import build_site
from public_site import PublicSite
from test_lock_v11 import _two_slates

V10, V11 = build_site.MODEL_TAG_V10, build_site.MODEL_TAG_V11


def _hyp(key):
    return next(h for h in analysis.HYPOTHESES if h["key"] == key)


def _across_reset(tags=(V10, V11, V11)):
    """Locked games 10+: three on the 2026-10-28 slate (v10, the ·L rows) and
    three on the reset slate with the given lock tags."""
    old = _two_slates("2026-10-27", "2026-10-28", V10)
    new = _two_slates("2026-10-28", "2026-10-29", V11).iloc[3:].copy()
    new["game_id"] = [f"new{i}" for i in range(3)]
    new["lock_model_tag"] = list(tags)
    return pd.concat([old.iloc[3:], new], ignore_index=True)


def test_registration_and_rules_are_unchanged():
    assert analysis.FAMILY_REGISTRATION_UTC == "2026-10-29T13:58:40Z"
    assert analysis.FAMILY_SLATE == "2026-10-29" and analysis.FAMILY_TAGS == ("_v11",)
    fam = {h["key"]: h for h in analysis.family_hypotheses() if h["registered_utc"] ==
           analysis.FAMILY_REGISTRATION_UTC}
    assert set(fam) == {k + "·L·v11" for k in ("H5", "H5·E", "H2·F", "H2·F·E", "H4", "H4·E")}
    for key, h in fam.items():
        parent = _hyp(h["parent"])
        assert parent["key"] + "·v11" == key and parent["until_slate"] == "2026-10-29"
        for f in ("market", "early", "favourite", "min_edge", "sigma", "native",
                  "hindsight", "requires_v9"):
            assert h.get(f) == parent.get(f), (key, f)
    assert fam["H5·L·v11"]["sigma"] == 13.5 and fam["H5·L·v11"]["min_edge"] == 1
    assert fam["H4·L·v11"]["min_edge"] == 0.12 and fam["H2·F·L·v11"]["favourite"]
    # no hindsight grouping is duplicated by the family rows
    assert all("v11" not in label for label, _ in analysis.hypothesis_groups())


def test_lock_rows_close_and_family_rows_start_at_the_reset_slate():
    g = _across_reset()
    h5l, fam = _hyp("H5·L"), _hyp("H5·L·v11")
    old = set(g.game_id[:3])
    assert set(analysis.hypothesis_picks(g, h5l, "lock").game_id) == old
    # the v10 lock on the reset slate enters neither row; the v11 locks the family
    assert set(analysis.hypothesis_picks(g, fam, "lock").game_id) == {"new1", "new2"}
    sig = analysis.hypothesis_signals(g.assign(home_won=np.nan), fam, "lock")
    assert set(sig.game_id) == {"new1", "new2"}
    st = analysis.lock_status(g, now="2026-11-21T00:00Z")
    assert (st["family_locked"], st["off_family"]) == (2, 1)
    # hindsight prices are not cut by the native scope: same bets as the parent
    for price in ("first", "pre", "close"):
        a, b = (analysis.hypothesis_picks(g, h, price) for h in (fam, h5l))
        assert (set(a.game_id) if len(a) else set()) == (set(b.game_id) if len(b) else set())


def test_both_stages_need_a_v11_lock_tag():
    g = _across_reset((V11, V10, build_site.MODEL_TAG_V11_NEUTRAL)).assign(gp_home=3, gp_away=3)
    for key in ("H5·E·L·v11", "H4·E·L·v11"):
        assert set(analysis.hypothesis_picks(g, _hyp(key), "lock").game_id) == {"new0", "new2"}
    # the closed early lock row keeps only its own slate
    assert set(analysis.hypothesis_picks(g, _hyp("H5·E·L"), "lock").game_id) == set(g.game_id[:3])
    # the public floored rule first (edge floors amendment, 2026-10-29)
    assert analysis.ats_card(g.iloc[3])["hyp"]["key"] == "H5·3·E·L·v11"
    assert analysis.ats_card(g.iloc[0])["hyp"]["key"] == "H5·E·L"


def test_public_pages_show_only_the_family_record():
    g = _across_reset()
    site = PublicSite(build_site, g, "2026-10-30", "built")
    assert {r["hyp"]["key"] for r in site.records} <= {h["key"] for h in analysis.family_hypotheses()}
    assert {str(r["row"].game_id) for r in site.records} == {"new1", "new2"}
    res, perf = site.results_page(), site.performance_page()
    for page in (res, perf):
        assert "restarted with the October 29, 2026 games" in page
        assert "kept separately and are not included here" in page
        assert "first snapshot" not in page.lower().replace("first priced snapshot", "")
    assert "data-basis" not in res and "AWY @ HOM<small>2026-10-28" not in res
    assert res.count("data-result-row ") == 2 * 2           # H5 and H4 on two games
    # an off-family lock is forecast only on Today, with its reason
    card = PublicSite(build_site, g, "2026-10-29", "built").card(g.iloc[3])
    assert "Forecast only" in card and "earlier model" in card
    guide = site.guide_page()
    assert "the current record, its minimum gaps and the total began with the October 29, 2026 games" in guide
    assert "closed on" not in guide


def test_empty_family_record_starts_here():
    g = _two_slates()                       # only the closed 2026-10-28 locks
    site = PublicSite(build_site, g, "2026-10-29", "built")
    assert not site.records
    res = site.results_page()
    assert "The forward record starts here" in res and "restarted" in res
    assert "No graded picks yet" in site.performance_page()
