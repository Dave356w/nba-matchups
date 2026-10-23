"""Publication boundary and parity of pending selections with frozen grades."""
import numpy as np
import pandas as pd
import pytest

import analysis
import build_site as ui
import ledger
from public_site import PublicSite
from test_analysis_and_pages import _ats_rows


def rows():
    g = _ats_rows(6).assign(first_p_home=.8, p_home=.8,
        first_home_ml=-150, first_away_ml=130, first_q_home=.6,
        first_book='dk', first_snapshot_utc='2026-11-19T15:07:00Z',
        slate_date='2026-11-19', tip_utc='2026-11-20T00:30:00Z',
        pre_home_ml=-150, pre_away_ml=130, pre_q_home=.6,
        pre_book='dk', margin_home=9.5)
    g['gp_home'], g['gp_away'] = [0,1,9,10,30,3], 30
    g.loc[5,'first_model_tag'] = ui.MODEL_TAG_V8
    return g


def test_pending_signal_membership_matches_registered_grades_for_every_stage_and_price():
    g = rows()
    for hyp in analysis.active_hypotheses():
        for price in ('first','pre','close'):
            a = analysis.hypothesis_signals(g,hyp,price)
            b = analysis.hypothesis_picks(g,hyp,price)
            assert (set(a.game_id) if len(a) else set()) == (set(b.game_id) if len(b) else set())
            if len(a):
                aa=a.set_index('game_id').sort_index()
                bb=b.set_index('game_id').sort_index()
                assert list(aa.side) == list(bb.side)
                assert np.allclose(aa.ml,bb.ml)
                assert np.allclose(aa.edge,bb.edge)
    for hyp in analysis.HYPOTHESES:
        if hyp.get('retired'):
            assert analysis.hypothesis_signals(g,hyp).empty


def test_pending_selections_never_fabricate_results_and_ignore_later_refreshes():
    g = rows().assign(home_won=np.nan,home_pts=np.nan,away_pts=np.nan)
    site = PublicSite(ui,g,'2026-11-19','built')
    assert len(site.records) == 12  # four eligible games, three tests
    assert all(r['grade'] is None for r in site.records)
    changed = g.assign(p_home=.01,pre_home_ml=999,pre_away_ml=-999,pre_q_home=.01,
                       model_tag=ui.MODEL_TAG_V9_NEUTRAL)
    fresh = PublicSite(ui,changed,'2026-11-19','built')
    assert [site.selection(r) for r in site.records] == [fresh.selection(r) for r in fresh.records]
    html = fresh.results_page()
    assert html.count("class='result-badge'>Pending") == 12
    assert 'No graded picks yet' in fresh.performance_page()
    assert '>+999<' not in html and '>-999<' not in html


def test_public_forecasts_cannot_promote_a_legacy_lean_record_or_reconstructed_rows():
    g = rows().assign(first_model_tag=ui.MODEL_TAG_V8,gp_home=3)
    site = PublicSite(ui,g,'2026-11-19','built')
    assert not site.records
    assert 'The forward record starts here' in site.results_page()
    assert '100.0%' not in site.today_page()
    assert 'class=\'record\'' not in site.performance_page()
    card = site.card(g.iloc[1])
    assert 'Forecast only' in card and 'earlier model' in card
    assert ' pp edge' not in card and 'EV ' not in card
    assert 'Composite Δ' not in card and 'INJURIES' not in card
    assert 'Game details' in card and 'Predicted margin' in card
    assert not PublicSite(ui,g.assign(basis='reconstructed'),'2026-11-19','built').records


def test_results_keep_losses_pushes_pending_overlap_and_seasons_separate():
    g = rows().iloc[1:4].copy()
    g['home_pts'],g['away_pts'],g['home_won'] = [90,106,np.nan],[100,100,np.nan],[0,1,np.nan]
    g.loc[g.index[1],'first_spread']=-6
    g.loc[g.index[1],'season']=2026
    site=PublicSite(ui,g,'2026-11-19','built')
    result=site.results_page()
    assert 'Loss' in result and 'Push' in result and 'Pending' in result
    assert result.count('data-result-row ')==9
    perf=site.performance_page()
    assert '2025-26 · DraftKings' in perf and '2026-27 · DraftKings' in perf
    assert 'Early · games 1–9' in perf and 'Established · games 10+' in perf
    assert 'no combined portfolio return' in perf
    # An actual loss remains a negative return, not an empty or hidden cell.
    assert '-1.00u' in result


def test_publication_allowlist_removes_stale_research_and_preserves_backend(tmp_path,monkeypatch):
    out=tmp_path/'public'
    out.mkdir()
    (out/'grades.html').write_text('Retired')
    (out/'ledger_report.txt').write_text('private research')
    (out/'unexpected.csv').write_text('old data')
    (out/'nested').mkdir()
    (out/'nested'/'research.html').write_text('retired')
    outside=tmp_path/'keep.txt'
    outside.write_text('preserved')
    (out/'external').symlink_to(outside)
    monkeypatch.setattr(ui,'OUT_DIR',out)
    text=ui.write_pages(rows(),ledger.empty(),'2026-11-19',True)
    assert {p.name for p in out.iterdir()}==ui.PUBLIC_FILES
    assert outside.read_text()=='preserved'
    backend=tmp_path/'diagnostics'/'site'
    assert (backend/'ledger_report.txt').read_text()==text
    assert 'Retired' in (backend/'grades.html').read_text()
    for path in out.glob('*.html'):
        html=path.read_text()
        for forbidden in ('href=\'grades.html','href=\'model.html','market-calibration.html',
                          'ledger_report.txt','ESPN BET','Retired','Games priced like this','Fitted coefficients'):
            assert forbidden not in html,(path.name,forbidden)
        assert html.count("aria-current='page'")==1
    (out/'leak.txt').write_text('must fail')
    with pytest.raises(ValueError,match='approved pages'):
        ui.validate_public_output()


def test_unavailable_probability_and_market_render_without_a_false_tracked_pick():
    g=rows().iloc[:1].assign(p_home=np.nan,first_p_home=np.nan,pre_home_ml=np.nan,
                            pre_away_ml=np.nan,pre_book=np.nan)
    site=PublicSite(ui,g,'2026-11-19','built',False)
    assert 'Forecast unavailable' in site.today_page()
    assert 'Tracked picks · first snapshot' not in site.today_page()
    assert not site.records
