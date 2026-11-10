"""H4 (large probability gap) retirement (CLAUDE.md, registered
2026-11-10T05:55:00Z): its v11-family rows close with the 2026-11-09 slate
and leave the public pages; closed bets stay graded internally."""
import analysis
import build_site
import report
from public_site import RULES, PublicSite
from test_public_site import CLOSED, rows

H4_ROWS = ("H4·L·v11", "H4·E·L·v11")


def hyp(key):
    return next(h for h in analysis.HYPOTHESES if h["key"] == key)


def test_registration_and_closed_rows():
    assert analysis.GAP_RETIREMENT_UTC == "2026-11-10T05:55:00Z"
    assert analysis.GAP_CLOSE_SLATE == "2026-11-10"
    for key in H4_ROWS:
        h = hyp(key)
        assert h["until_slate"] == analysis.GAP_CLOSE_SLATE and h["from_slate"] == analysis.FAMILY_SLATE
        assert h["min_edge"] == 0.12 and not h["favourite"] and h["native"] == "lock"
        assert not h.get("public") and not h.get("retired")     # closed, still graded internally
    assert not {h["key"] for h in analysis.public_hypotheses()} & set(H4_ROWS)
    # no other row's scope moved
    for h in analysis.family_hypotheses():
        if not h["key"].startswith("H4"):
            assert h["until_slate"] is None


def test_bets_through_the_last_slate_stay_and_later_slates_make_none():
    g = rows()                                   # every eligible game has a 20-point gap
    for key in H4_ROWS:
        h = hyp(key)
        last = g.assign(slate_date="2026-11-09")
        after = g.assign(slate_date=analysis.GAP_CLOSE_SLATE)
        kept = analysis.hypothesis_picks(last, h, "lock")
        assert len(kept) and len(analysis.hypothesis_signals(last, h, "lock")) == len(kept)
        assert not len(analysis.hypothesis_picks(after, h, "lock"))
        assert not len(analysis.hypothesis_signals(after, h, "lock"))
        # the other public tests still bet the later slate
    other = hyp("H5·3·L·v11")
    assert len(analysis.hypothesis_picks(g.assign(slate_date=analysis.GAP_CLOSE_SLATE), other, "lock"))


def test_public_pages_drop_the_test_and_carry_one_dated_line():
    assert [k for k, _, _ in RULES] == ["H5·3", "H2·F·5", "OU"]
    g = rows().assign(slate_date="2026-11-09")   # a slate H4 still bet internally
    site = PublicSite(build_site, g, CLOSED, "built")
    assert not [r for r in site.records if r["hyp"]["key"].startswith("H4")]
    line = "A fourth test, large probability gap, was retired after the November 9, 2026 games."
    pages = site.render()
    for name in ("results.html", "performance.html", "how-it-works.html"):
        assert line not in pages[name]
        assert "large probability gap" not in pages[name]
    for name, html in pages.items():
        assert "Large probability gap" not in html and "#gap" not in html and "id='gap'" not in html
        assert "12 percentage points" not in html and "at least 12 points" not in html
        assert "Four registered tests" not in html and "Four fixed tracking rules" not in html
    assert line not in pages["index.html"] and line not in pages["dfs.html"]
    guide = pages["how-it-works.html"]
    assert "Paper tracking" not in guide
    assert "Total (over/under)" not in guide
    assert "rule-number'>04" not in guide


def test_internal_report_keeps_the_closed_rows_and_names_the_retirement():
    g = rows().assign(slate_date="2026-11-09")
    text = "\n".join(report.hypotheses_block(g, None))
    assert f"H4 RETIRED (registered {analysis.GAP_RETIREMENT_UTC}" in text
    assert "H4·E·L·v11" in text and "H4·L·v11" in text
