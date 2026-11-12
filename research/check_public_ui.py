"""CI browser checks for the public tool, using live data and labelled fixtures.

Run after installing playwright and Chromium. Screenshots are CI artifacts;
fixture predictions/results are never written to the ledger or public site.
"""
from __future__ import annotations

import functools
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_site as ui  # noqa: E402
import dfs  # noqa: E402
import ledger  # noqa: E402
import live_wp  # noqa: E402
from matchup_context import MatchupContext  # noqa: E402
import nba_composite as nc  # noqa: E402
import totals  # noqa: E402

# the O/U locks carry the deployed totals model's tag (ou2 until Fit model
# wrote ou3, ou3 since), so the fixture's picks enter the public O/U rows
_TOTALS = totals.load()
OU_TAG = totals.model_tags(_TOTALS)[0] if _TOTALS and _TOTALS.get("report") else totals.TAG_OU2
from public_site import PublicSite  # noqa: E402


def fixture():
    rows = []
    for i, (gp, won, margin) in enumerate(((3, 1, 12), (3, 0, -5), (3, 1, 6), (30, None, None), (3, 1, 2))):
        row = dict.fromkeys(ledger.COLUMNS, np.nan)
        row.update(game_id=f'fixture-{i}', basis='native', season=2027,
            slate_date='2026-11-19', tip_utc='2026-11-20T03:00:00Z',
            home='BOS', away='NYK', model_tag=ui.MODEL_TAG_V11,
            first_model_tag=ui.MODEL_TAG_V9, gp_home=gp, gp_away=30,
            p_home=.8, p_lean=.8, lean='BOS', margin_home=11.2,
            first_p_home=.8, first_snapshot_utc='2026-11-19T15:07:00Z',
            pre_book='dk', first_book='dk', close_book='dk',
            pre_home_ml=-150, pre_away_ml=130, pre_q_home=.6,
            first_home_ml=-150, first_away_ml=130, first_q_home=.6,
            first_spread=-6, first_home_spread_odds=-110,
            first_away_spread_odds=-110, pre_spread=-6,
            # the five-hour lock (4 h before tip): today's tracked picks, in
            # the v11-family rows the pages show (model-family reset)
            lock_snapshot_utc='2026-11-19T23:00:00Z', lock_model_tag=ui.MODEL_TAG_V11,
            lock_p_home=.8, lock_book='dk', lock_home_ml=-150, lock_away_ml=130,
            lock_q_home=.6, lock_spread=-6, lock_home_spread_odds=-110,
            lock_away_spread_odds=-110,
            # the O/U test (edge floors and O/U amendment): the model leans over
            pre_ou_line=205.5, pre_ou_over=-110, pre_ou_under=-110, ou_total=212.0,
            ou_sigma=18.5, ou_p_over=.62, ou_tag=OU_TAG,
            lock_ou_snapshot_utc='2026-11-19T23:00:00Z', lock_ou_tag=OU_TAG,
            lock_ou_total=212.0, lock_ou_p_over=.62, lock_ou_book='dk', lock_ou_line=205.5,
            lock_ou_over=-110, lock_ou_under=-110,
            home_won=np.nan if won is None else won,
            home_pts=np.nan if margin is None else 100+margin,
            away_pts=np.nan if margin is None else 100)
        if won is not None:
            row.update(close_home_ml=-150, close_away_ml=130, close_q_home=.6)
        if i == 4:
            # A model prediction that qualifies for no betting rule.
            row.update(lock_p_home=.5, lock_q_home=.5, lock_spread=0,
                       lock_ou_p_over=.5)
        rows.append(row)
    return pd.DataFrame(rows, columns=ledger.COLUMNS)


def dfs_fixture():
    """A synthetic Yahoo slate for 2026-11-19: three games at 4:00, 4:30 and
    7:00 PM PT, so the slate toggle offers Main, Early (4:00-4:30) and Late
    (4:30-7:00); a one-game slate is never offered."""
    games = (('NYK', 'BOS', '2026-11-20T00:00:00Z', '401'), ('ORL', 'MIA', '2026-11-20T00:30:00Z', '402'),
             ('GSW', 'LAL', '2026-11-20T03:00:00Z', '403'))
    yahoo, sleeper, native = [], [], []
    for gi, (away, home, start, gid) in enumerate(games):
        native.append(dict(game_id=gid, slate_date='2026-11-19', home=home, away=away, tip_utc=start))
        for ti, team in enumerate((away, home)):
            for k, pos in enumerate(('PG', 'SG', 'SF', 'PF', 'C') * 2):
                name, code = f'Fixture {team} {pos}{k // 5}', f'nba.p.{gi}{ti}{k}'
                sal = 12 + (7 * k + 5 * ti + 3 * gi) % 30
                yahoo.append(dict(slate_date='2026-11-19', snapshot_utc='2026-11-19T15:00:00Z', player_code=code,
                                  game_code=f'nba.g.{gid}', name=name, team=team, position=pos, salary=sal,
                                  fppg=sal, home=home, away=away, start_utc=start, cap=200))
                sleeper.append({**{c: '' for c in ledger.SLEEPER_COLUMNS}, 'game_id': gid, 'tip_utc': start,
                                'snapshot_utc': '2026-11-19T15:00:00Z', 'kind': 'pre', 'team': team,
                                'sleeper_id': code, 'name': name, 'minutes': '30', 'pts': str(sal * .7),
                                'reb': '5', 'ast': '3', 'stl': '1', 'blk': '.5', 'to': '2', 'tpm': '1'})
    as_str = lambda rows, cols: pd.DataFrame(rows, columns=cols).astype(str)  # noqa: E731
    return (as_str(yahoo, dfs.YAHOO_COLUMNS), as_str(sleeper, ledger.SLEEPER_COLUMNS),
            pd.DataFrame(native))


def matchup_fixture():
    """Explicitly synthetic prior-game context, never saved as production data."""
    logs = {}
    for team, n in (('BOS', 3), ('NYK', 30)):
        totals = dict(zip(nc.COLS, [40, 85, 12, 18, 23, 10, 34, 12,
                                   38, 84, 11, 17, 22, 9, 32, 14]))
        logs[(2027, team)] = pd.DataFrame([
            dict(date=date, pts=110 if i % 2 else 100, opp_pts=105, **totals)
            for i, date in enumerate(pd.date_range(end='2026-11-18', periods=n))])
    injuries = pd.DataFrame([
        dict(game_id='fixture-0', tip_utc='2026-11-20T03:00:00Z', snapshot_utc='2026-11-19T22:00:00Z',
             team='BOS', player_id='fixture', name='Example player', status='Questionable', detail='Ankle'),
        dict(game_id='fixture-0', tip_utc='2026-11-20T03:00:00Z', snapshot_utc='2026-11-19T22:00:00Z',
             team='NYK', player_id='', name='', status='NONE', detail='')], columns=ledger.INJURY_COLUMNS)
    return dict(context=MatchupContext(logs), injuries=injuries)


def main():
    from playwright.sync_api import sync_playwright

    _, model = ui.load_model()
    ui.ACTIVE_TAGS = ui.active_tags(model, ui.load_avail()) if model else [ui.MODEL_TAG_V9]
    out = ROOT/'diagnostics'/'public_ui'/'screenshots'
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for name, native, day in (('live',ledger.load(ledger.NATIVE_PATH),ui.et_today()),
                                  ('fixture',fixture(),'2026-11-19')):
            target = root/name
            target.mkdir()
            context = matchup_fixture() if name == 'fixture' else dict(replay=pd.concat([ledger.load(ledger.RECON_PATH), ledger.load(ledger.REBUILT_PATH)], ignore_index=True))
            pages = PublicSite(ui,native,day,'UI verification · Pacific time', **context).render()
            for filename, html in pages.items():
                if name == 'fixture':
                    if filename == 'dfs.html':
                        y, sl, nat = dfs_fixture()
                        html = dfs.page(ui, day, 'UI verification · Pacific time', native=nat, yahoo=y,
                                        sleeper=sl, css=__import__('public_site').CSS)
                    html = html.replace("<main id='main-content'>", "<main id='main-content'><div class='flag-note' role='note'>UI test fixture · synthetic data for verification</div>")
                (target/filename).write_text(html,encoding='utf-8')
        handler = functools.partial(SimpleHTTPRequestHandler,directory=str(root))
        server = ThreadingHTTPServer(('127.0.0.1',0),handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                errors = []
                page.on('pageerror',lambda e: errors.append(str(e)))
                # Stable pregame UI: score overlays are verified separately below.
                page.route('**/apis/site/v2/sports/basketball/nba/scoreboard?**',
                           lambda route: route.fulfill(json={'events':[]}))
                for device, width, height in (('desktop',1440,1000),('mobile',390,844)):
                    page.set_viewport_size({'width':width,'height':height})
                    for filename, label in ui.PUBLIC_PAGES:
                        page.goto(f'{base}/live/{filename}')
                        page.get_by_role('heading',level=1).wait_for()
                        assert page.locator('.nav [aria-current="page"]').count()==1
                        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                        assert page.locator('.nav a').evaluate_all('els => els.every(e => e.scrollWidth <= e.clientWidth)')
                        assert page.get_by_role('link',name='Research',exact=True).count()==0
                        assert page.locator('#paper-results, #paper-performance, [data-result-row], .tracked-line').count()==0
                        assert 'Paper tracking' not in page.locator('body').inner_text()
                        if filename=='index.html' and page.locator('.gcard:visible').count():
                            top=page.locator('.gcard:visible').first.bounding_box()['y']
                            assert top < (460 if device == 'mobile' else height-100),(device,'first game is too far down the page',top)
                        if filename=='dfs.html' and page.locator('#dfs-data').count():
                            # the live slate: lineups, or (every game started) a clear "no lineup"
                            page.wait_for_function("!/Building/.test(document.getElementById('dfs-msg').textContent)")
                            msg = page.locator('#dfs-msg').inner_text()
                            built = page.locator('.dfs-lineup').count()
                            assert built or msg.startswith('No complete portfolio found'), msg
                            checks.append(f'{device}: DFS lineups · the browser optimizer ran on the live slate'
                                          f" ({built} lineups{'' if built else ': every game started'})")
                        # The mixed archive has hundreds of rows; a full-page
                        # JPEG can exceed Chromium's image-height limit.
                        shot = out/f'{device}-{filename[:-5]}.jpg'
                        page.screenshot(path=str(shot),full_page=filename!='results.html',type='jpeg',quality=85)
                        assert shot.stat().st_size > 0, f'empty screenshot: {shot}'
                        checks.append(f'{device}: {label} · navigation, overflow and screenshot passed')
                    page.goto(f'{base}/fixture/index.html')
                    assert page.locator('.gcard:visible').count()==1
                    assert page.locator('.gcard:visible [data-season-calibration]').count()==1
                    assert 'No historical results in this cell yet' in page.locator('.gcard:visible').inner_text()
                    page.locator('[data-group="stage"][data-show="all"]').click()
                    assert page.locator('.gcard:visible').count()==5
                    # Stage and status compose; counts reflect the other group.
                    page.locator('[data-group="state"][data-show="final"]').click()
                    assert page.locator('.gcard:visible').count()==4
                    page.locator('[data-group="stage"][data-show="early"]').click()
                    assert page.locator('.gcard:visible').count()==4
                    page.locator('[data-group="stage"][data-show="established"]').click()
                    assert page.locator('.gcard:visible').count()==0
                    assert 'No final games with both teams at 10+ games yet.' in page.locator('[data-filter-empty]').inner_text()
                    page.get_by_role('button',name='Show all games',exact=True).click()
                    assert page.locator('.gcard:visible').count()==5
                    assert page.locator('[data-filter] [aria-pressed="true"]').count()==2
                    page.locator('[data-group="state"][data-show="live"]').click()
                    assert 'No games are live right now.' in page.locator('[data-filter-empty]').inner_text()
                    page.get_by_role('button',name='Show all games',exact=True).click()
                    assert page.get_by_text('Paper picks · 5 h lock').count()==0
                    page.locator('[data-gid="fixture-0"] > summary').click()
                    assert page.get_by_role('heading',name='Recorded before play',exact=True).is_visible()
                    card = page.locator('[data-gid="fixture-0"]')
                    assert card.locator('.team-record').all_inner_texts() == ['(15–15)', '(1–2)']
                    assert card.get_by_role('table', name='Season four-factor comparison before this slate').is_visible()
                    assert card.get_by_text('Example player', exact=True).is_visible()
                    assert card.get_by_text('Questionable', exact=True).is_visible()
                    assert 'Source: ESPN' in card.inner_text()
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    checks.append(f'{device}: matchup context · season W/L, prior-slate four factors and timestamped player injuries passed')
                    page.get_by_role('button',name='Toggle colour theme').click()
                    assert page.locator('html').get_attribute('data-theme') in ('dark','light')
                    page.screenshot(path=str(out/f'{device}-fixture-tracked.jpg'),full_page=True,type='jpeg',quality=85)
                    page.goto(f'{base}/fixture/results.html')
                    assert page.locator('[data-prediction-row]:visible').count()==1
                    assert page.locator('[data-prediction-row][data-outcome="loss"]').count()==1
                    page.get_by_label('Prediction outcome',exact=True).select_option('pending')
                    assert page.locator('[data-prediction-row]:visible').count()==1
                    page.get_by_label('Prediction stage',exact=True).select_option('early')
                    assert page.locator('[data-prediction-row]:visible').count()==0
                    assert page.locator('[data-predictions-empty]').is_visible()
                    page.get_by_label('Prediction outcome',exact=True).select_option('all')
                    assert page.locator('[data-prediction-row]:visible').count()==4
                    page.screenshot(path=str(out/f'{device}-fixture-model-results.jpg'),full_page=True,type='jpeg',quality=85)
                    assert page.locator('[data-result-row]').count()==0
                    assert page.locator('#paper-results').count()==0
                    page.screenshot(path=str(out/f'{device}-fixture-results.jpg'),full_page=True,type='jpeg',quality=85)
                    page.goto(f'{base}/fixture/performance.html')
                    assert page.locator('[data-ml-panel]:visible').count()==1
                    assert page.get_by_label('Record source',exact=True).input_value()=='live'
                    assert page.locator('[data-source-summary]:visible article').count()==2
                    page.get_by_label('Model stage',exact=True).select_option('early')
                    panel = page.locator('[data-ml-panel]:visible')
                    assert panel.locator('.ml-metrics').get_by_text('3–1',exact=True).count()==1
                    assert page.locator('[data-source-summary]:visible').get_attribute('data-stage')=='early'
                    page.get_by_label('Record source',exact=True).select_option('all')
                    assert page.locator('[data-ml-panel]:visible .ml-metrics').get_by_text('3–1',exact=True).count()==1
                    assert page.locator('[data-source-summary]:visible article').count()==2

                    assert panel.locator('[data-roi-matrix]').count()==1
                    assert panel.locator('[data-confidence-bands]').count()==0
                    counts=panel.locator('[data-roi-matrix] [data-n]').evaluate_all("els => els.map(e => Number(e.dataset.n))")
                    assert sum(counts)==4 and max(counts)<=50
                    assert panel.locator('[data-model-band="3"][data-market-band="1"]').get_by_text('+11.1%',exact=True).count()==1
                    assert panel.get_by_text('Log loss',exact=True).count()==1
                    assert panel.get_by_text('Brier score',exact=True).count()==1
                    page.get_by_label('Record source',exact=True).select_option('replay')
                    assert page.locator('[data-ml-panel]:visible').get_by_text('No predictions in this view yet',exact=True).count()==1
                    page.get_by_label('Record source',exact=True).select_option('live')
                    assert page.locator('[data-ml-panel]:visible .ml-metrics').get_by_text('3–1',exact=True).count()==1
                    assert page.locator('#paper-performance').count()==0
                    checks.append(f'{device}: ML focus · 10+ default, separate early record, W/L, matched scores, source/outcome/stage filters, empty states and exclusion of betting tests passed')
                    panel = page.locator('[data-ml-panel]:visible')
                    panel.get_by_text('Return over time',exact=True).click()
                    page.screenshot(path=str(out/f'{device}-fixture-performance.jpg'),full_page=True,type='jpeg',quality=85)
                    checks.append(f'{device}: labelled fixture · ML outcomes, filters, details, theme and return chart passed')
                    page.goto(f'{base}/fixture/dfs.html')
                    page.locator('.dfs-lineup').first.wait_for()
                    labels = page.locator('#dfs-slate option').all_inner_texts()
                    assert [l.split(' · ')[0] for l in labels] == ['Main','Early','Late'], labels
                    for key, teams in (('main',{'NYK','BOS','ORL','MIA','GSW','LAL'}),('early1',{'NYK','BOS','ORL','MIA'}),
                                       ('late1',{'ORL','MIA','GSW','LAL'})):
                        page.select_option('#dfs-slate', key)
                        page.wait_for_function("!/Building/.test(document.getElementById('dfs-msg').textContent)")
                        used = set(page.locator('.dfs-lineup li em').all_inner_texts())
                        assert used and used <= teams and len(used) >= 3, (key, used)
                        assert page.locator('.dfs-lineup').count() == 10, key
                    page.select_option('#dfs-slate', 'main')
                    page.select_option('#dfs-mode', 'gpp')
                    assert page.locator('#dfs-sims').is_visible()
                    page.wait_for_function("/independent evaluation slates/.test(document.getElementById('dfs-msg').textContent)")
                    assert page.locator('.dfs-lineup').count() == 10
                    assert page.locator('.dfs-lineup header', has_text='candidate top 5%:').count() == 10
                    # exposure by salary: at most floor(10 x min(70%, 2% per $1)) lineups per player
                    counts = """() => { const D = JSON.parse(document.getElementById('dfs-data').textContent), sal = {};
                        D.players.forEach(p => sal[p.name] = p.sal); const n = {};
                        document.querySelectorAll('.dfs-lineup li:not(.hd) span:nth-child(2)').forEach(s => {
                          const name = s.childNodes[0].textContent.trim(); n[name] = (n[name] || 0) + 1; });
                        return Object.entries(n).map(([k, v]) => [k, v, sal[k]]); }"""
                    for mode in ('cash', 'gpp'):
                        for rule, cap in (('salary', lambda s: int(10 * min(.7, .02 * s) + 1e-9)), ('flat', lambda s: 7)):
                            page.select_option('#dfs-mode', mode)
                            page.select_option('#dfs-exprule', rule)
                            page.click('#dfs-go')
                            page.wait_for_function("!document.getElementById('dfs-go').disabled && /lineup/.test(document.getElementById('dfs-msg').textContent)")
                            used = page.evaluate(counts)
                            assert page.locator('.dfs-lineup').count() == 10, (mode, rule)
                            assert all(n <= max(1, cap(s)) for _, n, s in used), (mode, rule, used)
                    assert page.locator('#dfs-k').is_hidden()       # flat: no per-$1 input
                    # min player salary: nobody under it, except a locked player
                    sal = page.evaluate("() => Object.fromEntries(JSON.parse(document.getElementById('dfs-data').textContent).players.map(p => [p.name, [p.sal, p.id]]))")
                    cheap = min((v for v in sal.items() if v[1][0] < 20), key=lambda kv: kv[1][0])
                    # Isolate the salary filter from exposure feasibility:
                    # this small fixture has scarce centres after filtering.
                    page.fill('#dfs-exp', '100')
                    page.fill('#dfs-min', '20')
                    page.locator('#dfs-min').dispatch_event('change')
                    for mode in ('cash', 'gpp'):
                        page.select_option('#dfs-mode', mode)
                        page.click('#dfs-go')
                        page.wait_for_function("!document.getElementById('dfs-go').disabled && /lineup/.test(document.getElementById('dfs-msg').textContent)")
                        names = [n for n, _, _ in page.evaluate(counts)]
                        assert names and all(sal[n][0] >= 20 for n in names), (mode, names)
                    assert page.locator('#dfs-pool tr.out', has_text='below min').count() > 0
                    page.locator(f"#dfs-pool tr[data-id='{cheap[1][1]}'] input[data-act='lock']").check()
                    page.select_option('#dfs-mode', 'cash')
                    page.click('#dfs-go')
                    page.wait_for_function("!document.getElementById('dfs-go').disabled && /lineup/.test(document.getElementById('dfs-msg').textContent)")
                    assert page.locator('.dfs-lineup', has_text=cheap[0]).count() == page.locator('.dfs-lineup').count() == 10
                    page.fill('#dfs-min', '0')
                    page.fill('#dfs-exp', '70')
                    page.locator('#dfs-min').dispatch_event('change')
                    page.click('#dfs-reset')
                    page.wait_for_function("!document.getElementById('dfs-go').disabled && /lineup/.test(document.getElementById('dfs-msg').textContent)")
                    page.select_option('#dfs-exprule', 'salary')
                    page.select_option('#dfs-mode', 'gpp')
                    page.wait_for_function("/independent evaluation slates/.test(document.getElementById('dfs-msg').textContent)")
                    # each tournament lineup shows its players' points in a central independent evaluation slate
                    assert page.locator('.dfs-lineup li.sim:not(.hd)').count() == 80
                    assert page.locator('.dfs-lineup', has_text='Median evaluation example').count() == 10
                    seed = page.locator('#dfs-msg').inner_text()
                    page.click('#dfs-reseed')
                    page.wait_for_function(f"document.getElementById('dfs-msg').textContent !== {json.dumps(seed)} && /seed/.test(document.getElementById('dfs-msg').textContent)")
                    assert page.locator('.dfs-lineup').count() == 10
                    page.click('#dfs-reset')
                    page.wait_for_function("/seed 356\\)/.test(document.getElementById('dfs-msg').textContent)")
                    page.screenshot(path=str(out/f'{device}-fixture-dfs.jpg'),full_page=True,type='jpeg',quality=85)
                    checks.append(f'{device}: DFS fixture · Main, Early and Late slates each build 10 lineups from their own games; tournament mode builds 10 simulated lineups; exposure by salary (2% per $1, max 70%) and flat caps hold in both modes; tournament lineups show median examples from 5,000 independent evaluation slates; New draws reseeds; the min player salary leaves out cheaper players except a locked one')
                assert not errors,errors
                # Live-score overlay may update status/score only, never a selection.
                page.unroute_all()
                page.route('**/apis/site/v2/sports/basketball/nba/scoreboard?**',lambda route: route.fulfill(json={'events':[{
                    'id':'fixture-3','status':{'type':{'state':'in','name':'STATUS_IN_PROGRESS'},'period':2,'displayClock':'4:00'},
                    'competitions':[{'competitors':[{'homeAway':'home','score':'42'},{'homeAway':'away','score':'38'}]}]}]}))
                page.goto(f'{base}/fixture/index.html')
                live=page.locator('[data-gid="fixture-3"]')
                live.locator('.st').filter(has_text='Q2 4:00').wait_for()
                assert live.locator('.sc[data-side="home"]').inner_text()=='42'
                assert live.locator('.tracked-line').count()==0
                assert live.locator('.tracked-box .tracked-label').inner_text().lower().startswith('locked pick:')
                assert live.locator('[data-forecast-result]').inner_text()=='Pending'
                locked_text = live.locator('.tracked-box').inner_text()
                page.route('**/apis/site/v2/sports/basketball/nba/scoreboard?**',lambda route: route.fulfill(json={'events':[{
                    'id':'fixture-3','status':{'type':{'state':'post','completed':True,'shortDetail':'Final'}},
                    'competitions':[{'competitors':[{'homeAway':'home','score':'110'},{'homeAway':'away','score':'100'}]}]}]}))
                page.reload()
                live=page.locator('[data-gid="fixture-3"]')
                live.locator('.st').filter(has_text='Final').wait_for()
                assert live.locator('[data-forecast-result]').inner_text()=='Win'
                assert live.locator('.tracked-box').inner_text()==locked_text
                # Restore the in-progress fixture for the live win-chance checks below.
                page.unroute_all()
                page.route('**/apis/site/v2/sports/basketball/nba/scoreboard?**',lambda route: route.fulfill(json={'events':[{
                    'id':'fixture-3','status':{'type':{'state':'in','name':'STATUS_IN_PROGRESS'},'period':2,'displayClock':'4:00'},
                    'competitions':[{'competitors':[{'homeAway':'home','score':'42'},{'homeAway':'away','score':'38'}]}]}]}))
                page.reload()
                live=page.locator('[data-gid="fixture-3"]')
                live.locator('.st').filter(has_text='Q2 4:00').wait_for()
                checks.append('Live-score overlay updated status, scores and actual forecast outcome without changing the frozen ML selection')
                # Live win chance (owner request, 2026-11-07): the browser's
                # figure equals live_wp.win_prob for the card's frozen spread
                fit = live_wp.load()
                ph = live_wp.win_prob(42-38, 6, live_wp.tau(2, 240), fit)
                fmt = lambda p: '>99%' if p > .995 else '<1%' if p < .005 else f'{round(100*p)}%'  # noqa: E731
                box = live.locator('[data-live-wp]')
                assert box.is_visible()
                assert box.locator('[data-wp="home"]').inner_text()==fmt(ph)
                assert box.locator('[data-wp="away"]').inner_text()==fmt(1-ph)
                assert page.locator('[data-gid="fixture-0"] [data-live-wp]').is_hidden()  # final: no live readout
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                for device, width, height in (('desktop',1440,1000),('mobile',390,844)):
                    page.set_viewport_size({'width':width,'height':height})
                    live.screenshot(path=str(out/f'{device}-fixture-live-card.png'))
                checks.append(f'Live win chance · BOS {fmt(ph)} / NYK {fmt(1-ph)} at Q2 4:00, 42–38, BOS −6 at tip, '
                              'matching live_wp.win_prob; hidden for a final game; no horizontal overflow')
                # Auto-refresh (owner request, 2026-11-07): a live game is
                # re-fetched every 30 s (fake clock), a hidden tab waits and
                # catches up when shown, polling stops once every game is final
                def event(state, name, period, clock, home, away):
                    return {'id': 'fixture-3', 'status': {'type': {'state': state, 'name': name, 'completed': state == 'post',
                            'shortDetail': 'Final' if state == 'post' else ''}, 'period': period, 'displayClock': clock},
                            'competitions': [{'competitors': [{'homeAway': 'home', 'score': str(home)},
                                                              {'homeAway': 'away', 'score': str(away)}]}]}
                feed = [event('in', 'STATUS_IN_PROGRESS', 2, '4:00', 42, 38), event('in', 'STATUS_IN_PROGRESS', 3, '7:30', 61, 63),
                        event('in', 'STATUS_IN_PROGRESS', 4, '0:45.5', 99, 99), event('post', 'STATUS_FINAL', 4, '0.0', 101, 99)]
                calls = []
                def serve(route):
                    calls.append(route.request.url)
                    route.fulfill(json={'events': [feed[min(len(calls), len(feed)) - 1]]})
                page.unroute_all()
                page.clock.install()
                page.route('**/apis/site/v2/sports/basketball/nba/scoreboard?**', serve)
                page.goto(f'{base}/fixture/index.html')
                live.locator('.st').filter(has_text='Q2 4:00').wait_for()
                assert len(calls) == 1, calls
                cards = "els => els.map(e => e.dataset.gid)"
                # card order: live first, then upcoming, then final (tip order within)
                assert page.eval_on_selector_all('.tool-grid > .gcard', cards) == \
                    ['fixture-3', 'fixture-0', 'fixture-1', 'fixture-2', 'fixture-4']
                page.clock.run_for(31_000)
                live.locator('.st').filter(has_text='Q3 7:30').wait_for()
                assert len(calls) == 2
                assert live.locator('.sc[data-side="home"]').inner_text() == '61'
                ph = live_wp.win_prob(61 - 63, 6, live_wp.tau(3, 450), fit)
                assert box.locator('[data-wp="home"]').inner_text() == fmt(ph)
                page.evaluate("Object.defineProperty(document,'hidden',{value:true,configurable:true})")
                page.clock.run_for(5 * 60_000)
                assert len(calls) == 2, 'a hidden tab fetched'
                page.evaluate("Object.defineProperty(document,'hidden',{value:false,configurable:true});"
                              "document.dispatchEvent(new Event('visibilitychange'))")
                live.locator('.st').filter(has_text='Q4 0:45.5').wait_for()
                assert len(calls) == 3
                ph4 = live_wp.win_prob(0, 6, live_wp.tau(4, 45.5), fit)
                assert box.locator('[data-wp="home"]').inner_text() == fmt(ph4)
                page.clock.run_for(31_000)
                live.locator('.st').filter(has_text='Final').wait_for()
                assert len(calls) == 4 and box.is_hidden()
                assert live.get_attribute('data-state') == 'final'
                assert live.locator('.sc[data-side="home"]').inner_text() == '101'
                # the game that just ended moves down to its tip position among the finals
                assert page.eval_on_selector_all('.tool-grid > .gcard', cards) == \
                    ['fixture-0', 'fixture-1', 'fixture-2', 'fixture-3', 'fixture-4']
                page.clock.run_for(30 * 60_000)
                assert len(calls) == 4, 'polling continued after the final'
                checks.append(f'Auto-refresh · a live game re-fetched every 30 s (Q2 4:00 → Q3 7:30, BOS {fmt(ph)}; '
                              f'Q4 0:45.5 tied, BOS {fmt(ph4)}); a hidden tab waited and caught up when shown; '
                              'the final hid the live readout, moved the card down among the finals and stopped the polling')
                # opened before tip: the next fetch comes at the tip, then 30 s
                pre = event('pre', 'STATUS_SCHEDULED', 0, '0:00', 0, 0)
                pre['date'] = '2026-11-20T03:04:00Z'
                feed[:] = [pre, event('in', 'STATUS_IN_PROGRESS', 1, '11:40', 2, 0), event('in', 'STATUS_IN_PROGRESS', 1, '11:10', 2, 3)]
                calls.clear()
                tab = browser.new_page()
                tab.clock.install(time='2026-11-20T03:00:00Z')
                tab.route('**/apis/site/v2/sports/basketball/nba/scoreboard?**', serve)
                tab.goto(f'{base}/fixture/index.html')
                tab.locator('[data-gid="fixture-3"]').wait_for(state='attached')
                tab.wait_for_timeout(200)
                assert len(calls) == 1 and tab.locator('[data-gid="fixture-3"] [data-live-wp]').is_hidden()
                tab.clock.run_for(3 * 60_000 + 50_000)
                assert len(calls) == 1, 'fetched before the tip'
                tab.clock.run_for(15_000)
                tab.locator('[data-gid="fixture-3"] .st').filter(has_text='Q1 11:40').wait_for()
                assert len(calls) == 2 and tab.locator('[data-gid="fixture-3"] [data-live-wp]').is_visible()
                tab.clock.run_for(31_000)
                tab.locator('[data-gid="fixture-3"] .st').filter(has_text='Q1 11:10').wait_for()
                assert len(calls) == 3
                tab.close()
                checks.append('Auto-refresh · a page opened 4 min before tip fetched again at the tip, then every 30 s while live')
                browser.close()
        finally:
            server.shutdown()
    result='\n'.join(checks)+'\n'
    (out.parent/'CHECKS.txt').write_text(result)
    print(result)


if __name__=='__main__':
    main()
