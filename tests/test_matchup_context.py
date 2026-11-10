"""Point-in-time display data and complete public bet-type coverage."""
import numpy as np
import pandas as pd

import analysis
import build_site as ui
import ledger
import nba_composite as nc
from matchup_context import MatchupContext
from public_site import PublicSite, RULES, rule_key
from conftest import make_log, pin_ou2_model
from test_public_site import rows


def game():
    return rows().iloc[1].copy()


def test_record_and_factors_use_summed_prior_slate_totals_only():
    row = game()
    log = make_log(3, start='2026-11-17', gap_days=1)
    # Only Nov 17 and 18 qualify; the current slate and future never do.
    log.loc[0, ['pts', 'opp_pts']] = [110, 100]
    log.loc[1, ['pts', 'opp_pts']] = [90, 100]
    log.loc[2, nc.COLS] = 9999
    row['gp_home'] = 2
    ctx = MatchupContext({(2027, row.home): log})
    got = ctx.team(row, 'home')
    expected = nc.features_from_totals(log.iloc[:2][nc.COLS].sum().to_numpy(), orb_pct=True)
    expected *= [1, -1, 1, 1, -1, 1, -1, -1]
    assert got['record'] == (1, 1)
    assert got['games'] == 2
    assert np.allclose(got['factors'], expected)
    # This is not a mean of per-game percentages.
    log.loc[0, 'TFGA'] = 20
    got = ctx.team(row, 'home')['factors'][0]
    mean = np.mean([(r.TFG + .5 * r.T3P) / r.TFGA * 100 for _, r in log.iloc[:2].iterrows()])
    assert not np.isclose(got, mean)


def test_incomplete_stats_never_masquerade_as_full_season_and_boxes_deduplicate_players():
    row = game()
    row['gp_home'] = 2
    log = make_log(1, start='2026-11-17')
    boxes = pd.DataFrame([
        dict(game_id='one', team=row.home, date='2026-11-17', margin=5),
        dict(game_id='one', team=row.home, date='2026-11-17', margin=5),
        dict(game_id='two', team=row.home, date='2026-11-18', margin=-2),
        dict(game_id='today', team=row.home, date=row.slate_date, margin=10)])
    ctx = MatchupContext({(2027, row.home): log}, {2027: boxes})
    got = ctx.team(row, 'home')
    assert got['record'] == (1, 1) and got['factors'] is None
    assert MatchupContext({(2026, row.home): log}).team(row, 'home')['record'] is None
    assert MatchupContext(boxes={2027: boxes.iloc[:2]}).team(row, 'home')['record'] is None
    row['gp_home'] = 0
    assert ctx.team(row, 'home')['record'] == (0, 0)
    assert ctx.team(row, 'home')['factors'] is None


def test_saved_context_is_cache_only_and_missing_data_cannot_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(nc, 'CACHE', tmp_path)
    monkeypatch.setattr(nc, 'read_gamelog', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network')))
    ctx = MatchupContext.saved(pd.DataFrame([game()]))
    assert ctx.team(game(), 'home')['factors'] is None


def test_cards_show_records_factors_and_distinct_injury_source_with_safe_missing_states():
    row = game()
    row['gp_home'] = row['gp_away'] = 2
    log = make_log(2, start='2026-11-17', gap_days=1)
    ctx = MatchupContext({(2027, t): log for t in (row.home, row.away)})
    injuries = pd.DataFrame([
        dict(game_id=row.game_id, tip_utc=row.tip_utc, snapshot_utc='2026-11-19T20:00:00Z',
             team=row.home, player_id='1', name='<Player>', status='Questionable', detail='Ankle & foot'),
        dict(game_id=row.game_id, tip_utc=row.tip_utc, snapshot_utc='2026-11-19T20:00:00Z',
             team=row.away, player_id='', name='', status='NONE', detail=''),
        dict(game_id='different', tip_utc=row.tip_utc, snapshot_utc='2026-11-19T20:00:00Z',
             team=row.home, player_id='2', name='Other game', status='Out', detail=''),
        dict(game_id=row.game_id, tip_utc=row.tip_utc, snapshot_utc='2026-11-20T02:00:00Z',
             team=row.home, player_id='3', name='After tip', status='Out', detail='')], columns=ledger.INJURY_COLUMNS)
    site = PublicSite(ui, pd.DataFrame([row]), row.slate_date, 'built', context=ctx, injuries=injuries)
    card = site.card(row)
    assert "class='team-record'" in card and 'Season W/L before this slate' in card
    assert 'Four factors · season to date' in card and "class='factor-table'" in card
    assert 'Opponent rates allowed' in card and 'without decay or prior-season blending' in card
    assert 'through Nov 18, 2026' in card
    assert '&lt;Player&gt;' in card and 'Ankle &amp; foot' in card
    assert 'Questionable' in card and 'None listed in captured report' in card
    assert 'Source: ESPN' in card and 'NBA’s official injury report separately' in card
    assert 'Other game' not in card and 'After tip' not in card
    injuries.loc[0, 'snapshot_utc'] = '2026-11-19T00:15:00Z'
    # Both the date and the clock use Pacific time across UTC midnight.
    assert 'Captured Nov 18, 2026 · 4:15 PM PT' in site.card(row)
    empty = PublicSite(ui, pd.DataFrame([row]), row.slate_date, 'built', context=MatchupContext(),
                       injuries=pd.DataFrame(columns=ledger.INJURY_COLUMNS)).card(row)
    assert 'Season record unavailable' in empty and 'Season comparison unavailable' in empty
    assert 'No pregame injury list was captured' in empty


def test_all_three_public_test_records_match_every_registered_signal_and_grade(monkeypatch):
    pin_ou2_model(monkeypatch)                      # the ou2 public rows
    g = rows().iloc[1:5].copy()
    g['lock_ou_p_over'] = .8
    g['lock_ou_total'] = 220.
    g['lock_ou_line'] = 210.
    g['lock_ou_over'] = g['lock_ou_under'] = -110.
    g['lock_ou_book'] = 'dk'
    g['lock_ou_tag'] = 'totals_hl25_form5_rep_ou2'   # public rows: ou2
    g['lock_ou_snapshot_utc'] = g.lock_snapshot_utc
    g.loc[g.index[-1], ['home_won', 'home_pts', 'away_pts']] = np.nan
    site = PublicSite(ui, g, '2026-11-19', 'built')
    for root, _, slug in RULES:
        for early in (True, False):
            key = rule_key(root, early)
            hyp = site.hypotheses[key]
            if root == 'OU':
                signals, grades = analysis.ou_signals(g, hyp), analysis.ou_picks(g, hyp)
            else:
                src = analysis.on_book(g, 'lock', 'dk')
                signals = analysis.hypothesis_signals(src, hyp, 'lock')
                grades = analysis.hypothesis_picks(src, hyp, 'lock')
            actual = [r for r in site.records if r['hyp']['key'] == key]
            assert {r['row'].game_id for r in actual} == set(signals.game_id)
            assert {r['row'].game_id for r in actual if r['grade'] is not None} == set(grades.game_id)
            assert all(r['slug'] == slug for r in actual)
    assert len(site.records) == 12  # every game qualifies for each of the three tests
    assert site.results_page().count('data-result-row ') == 12
    # Display context cannot alter the frozen record.
    fresh = PublicSite(ui, g.assign(p_home=.01, pre_home_ml=999), '2026-11-19', 'built',
                       context=MatchupContext(), injuries=pd.DataFrame(columns=ledger.INJURY_COLUMNS))
    assert [site.selection(r) for r in site.records] == [fresh.selection(r) for r in fresh.records]
