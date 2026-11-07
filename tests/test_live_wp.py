"""Live win chance (owner request, 2026-11-07): display only, never data."""
import json
import math
import shutil
import subprocess

import numpy as np
import pytest

import build_site as ui
import live_wp
from public_site import PublicSite
from test_public_site import CLOSED, rows


FIT = live_wp.load()


def test_deployed_fit_is_valid_and_rewards_the_lead_and_the_line():
    assert FIT is not None, 'model/live_wp.json is missing or invalid'
    assert FIT['knots'][0] == 0 and FIT['knots'][-1] == 1
    assert all(a > 0 for a in FIT['a']) and all(b > 0 for b in FIT['b'])
    for t in (0.9, 0.6, 0.3, 0.1, 0.01):
        p = [live_wp.win_prob(lead, 0, t, FIT) for lead in (-6, -1, 0, 1, 6)]
        assert p == sorted(p) and p[2] == 0.5
        q = [live_wp.win_prob(0, mu, t, FIT) for mu in (-8, -2, 2, 8)]
        assert q == sorted(q)
        # home and away are the same game seen from either side
        assert math.isclose(live_wp.win_prob(4, -3, t, FIT), 1 - live_wp.win_prob(-4, 3, t, FIT))


def test_clock_and_end_of_game():
    assert live_wp.tau(1, 720) == 1 and live_wp.tau(2, 0) == .5
    assert live_wp.tau(4, 60) == 60 / 2880 and live_wp.tau(5, 300) == 300 / 2880
    assert live_wp.tau(6, 0) == 0
    # no time left: the leader has won; a tie goes to overtime's start
    assert live_wp.win_prob(1, -10, 0, FIT) == 1 and live_wp.win_prob(-1, 10, 0, FIT) == 0
    assert live_wp.win_prob(0, 4, 0, FIT) == live_wp.win_prob(0, 4, 300 / 2880, FIT)
    # a lead counts for more as time runs out
    assert live_wp.win_prob(3, 0, .05, FIT) > live_wp.win_prob(3, 0, .5, FIT)


def test_missing_or_broken_fit_shows_nothing(tmp_path):
    assert live_wp.load(tmp_path / 'none.json') is None
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(dict(knots=[0, 1], a=[.05], b=[.08, .08])))
    assert live_wp.load(bad) is None
    assert live_wp.script(None) == ''


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_browser_copy_matches_python():
    states = [(lead, mu, per, clock) for lead in (-12, -3, 0, 2, 9) for mu in (-7.5, 0, 4)
              for per, clock in ((1, 720), (1, 301.5), (2, 0), (3, 45.2), (4, 600), (4, 59.9), (4, 3), (4, 0), (5, 300), (6, 12))]
    js = live_wp.script(FIT).removeprefix('<script>').removesuffix('</script>')
    prog = ('var window={};' + js + ';var S=' + json.dumps(states) +
            ';console.log(JSON.stringify(S.map(function(s){return window.nbaLiveWP(s[0],s[1],window.nbaLiveTau(s[2],s[3]));})));')
    out = json.loads(subprocess.run(['node', '-e', prog], capture_output=True, text=True, check=True).stdout)
    want = [live_wp.win_prob(lead, mu, live_wp.tau(per, clock), FIT) for lead, mu, per, clock in states]
    assert np.allclose(out, want, atol=1e-6)


def test_card_carries_the_frozen_spread_and_an_empty_hidden_readout():
    g = rows().assign(pre_spread=-5.5)
    site = PublicSite(ui, g, CLOSED, 'built')
    card = site.card(g.iloc[1])
    assert "data-live-spread='-5.5'" in card
    assert "<div class='live-wp' data-live-wp hidden>" in card
    assert "<em data-wp='away'>—</em>" in card and "<em data-wp='home'>—</em>" in card
    assert 'live win chance' in card
    today = PublicSite(ui, g, '2026-11-19', 'built').today_page()
    assert 'window.nbaLive=function' in today
    assert today.index('window.nbaLive=function') < today.index('scoreboard?limit=100')
    assert 'if(window.nbaLive)window.nbaLive(c,ty,st,s);' in ui.SCORES_JS


def test_no_readout_without_a_draftkings_spread_or_a_fit():
    g = rows().assign(pre_spread=np.nan)
    assert 'data-live-wp' not in PublicSite(ui, g, CLOSED, 'built').card(g.iloc[1])
    g = rows().assign(pre_spread=-3, pre_book='espnbet')
    assert 'data-live-wp' not in PublicSite(ui, g, CLOSED, 'built').card(g.iloc[1])
    site = PublicSite(ui, rows().assign(pre_spread=-3), '2026-11-19', 'built')
    site.live_fit = None
    assert 'data-live-wp' not in site.card(site.native.iloc[1])
    assert 'window.nbaLive=function' not in site.today_page()


def test_live_readout_changes_no_record_or_selection():
    g = rows().assign(pre_spread=-5.5)
    a = PublicSite(ui, g, CLOSED, 'built')
    b = PublicSite(ui, g, CLOSED, 'built')
    b.live_fit = None
    assert [a.selection(r) for r in a.records] == [b.selection(r) for r in b.records]
    assert a.results_page() == b.results_page() and a.performance_page() == b.performance_page()
