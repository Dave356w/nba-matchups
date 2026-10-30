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
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_site as ui  # noqa: E402
import dfs  # noqa: E402
import ledger  # noqa: E402
from public_site import PublicSite  # noqa: E402


def fixture():
    rows = []
    for i, (gp, won, margin) in enumerate(((3, 1, 12), (3, 0, -5), (3, 1, 6), (30, None, None))):
        row = dict.fromkeys(ledger.COLUMNS, np.nan)
        row.update(game_id=f'fixture-{i}', basis='native', season=2027,
            slate_date='2026-11-19', tip_utc='2026-11-20T03:00:00Z',
            home='BOS', away='NYK', model_tag=ui.MODEL_TAG_V9,
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
            ou_sigma=18.5, ou_p_over=.62, ou_tag='totals_hl25_form5_ou1',
            lock_ou_snapshot_utc='2026-11-19T23:00:00Z', lock_ou_tag='totals_hl25_form5_ou1',
            lock_ou_total=212.0, lock_ou_p_over=.62, lock_ou_book='dk', lock_ou_line=205.5,
            lock_ou_over=-110, lock_ou_under=-110,
            home_won=np.nan if won is None else won,
            home_pts=np.nan if margin is None else 100+margin,
            away_pts=np.nan if margin is None else 100)
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


def main():
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
            pages = PublicSite(ui,native,day,'UI verification · Pacific time').render()
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
                        assert page.get_by_role('link',name='Research',exact=True).count()==0
                        if filename=='index.html' and page.locator('.gcard').count():
                            top=page.locator('.gcard').first.bounding_box()['y']
                            assert top < height-100,(device,'first game is below the fold',top)
                        if filename=='dfs.html' and page.locator('#dfs-data').count():
                            # the live slate: lineups, or (every game started) a clear "no lineup"
                            page.wait_for_function("!/Building/.test(document.getElementById('dfs-msg').textContent)")
                            msg = page.locator('#dfs-msg').inner_text()
                            built = page.locator('.dfs-lineup').count()
                            assert built or msg.startswith('No legal lineup'), msg
                            checks.append(f'{device}: DFS lineups · the browser optimizer ran on the live slate'
                                          f" ({built} lineups{'' if built else ': every game started'})")
                        page.screenshot(path=str(out/f'{device}-{filename[:-5]}.jpg'),full_page=True,type='jpeg',quality=85)
                        checks.append(f'{device}: {label} · navigation, overflow and screenshot passed')
                    page.goto(f'{base}/fixture/index.html')
                    page.locator('[data-show="tracked"]').click()
                    assert page.locator('.gcard:visible').count()==4
                    assert page.get_by_text('Tracked picks · 5 h lock').count()==4
                    page.locator('.gcard summary').first.click()
                    assert page.get_by_role('heading',name='Recorded before play',exact=True).is_visible()
                    page.get_by_role('button',name='Toggle colour theme').click()
                    assert page.locator('html').get_attribute('data-theme') in ('dark','light')
                    page.screenshot(path=str(out/f'{device}-fixture-tracked.jpg'),full_page=True,type='jpeg',quality=85)
                    page.goto(f'{base}/fixture/results.html')
                    assert page.locator('[data-result-row]').count()==16
                    for result in ('Win','Loss','Push','Pending'):
                        assert page.locator('.result-badge',has_text=result).count()>0,result
                    page.get_by_label('Test',exact=True).select_option('spread')
                    assert page.locator('[data-result-row]:visible').count()==4
                    page.get_by_label('Stage',exact=True).select_option('established')
                    assert page.locator('[data-result-row]:visible').count()==1
                    page.get_by_label('Stage',exact=True).select_option('early')
                    assert page.locator('[data-result-row]:visible').count()==3
                    page.screenshot(path=str(out/f'{device}-fixture-results.jpg'),full_page=True,type='jpeg',quality=85)
                    page.goto(f'{base}/fixture/performance.html')
                    assert page.get_by_text('1–1–1',exact=True).count()==1
                    page.get_by_text('Benchmarks and return chart',exact=True).first.click()
                    page.screenshot(path=str(out/f'{device}-fixture-performance.jpg'),full_page=True,type='jpeg',quality=85)
                    checks.append(f'{device}: labelled fixture · all outcomes, overlapping tests, filters, details, theme and chart passed')
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
                    page.wait_for_function("/simulated slates/.test(document.getElementById('dfs-msg').textContent)")
                    assert page.locator('.dfs-lineup').count() == 10
                    assert page.locator('.dfs-lineup header', has_text='top 5% in').count() == 10
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
                    page.select_option('#dfs-exprule', 'salary')
                    page.select_option('#dfs-mode', 'gpp')
                    page.wait_for_function("/simulated slates/.test(document.getElementById('dfs-msg').textContent)")
                    # each tournament lineup shows its players' points in the slate it won
                    assert page.locator('.dfs-lineup li.sim:not(.hd)').count() == 80
                    assert page.locator('.dfs-lineup', has_text='simulated slate').count() == 10
                    seed = page.locator('#dfs-msg').inner_text()
                    page.click('#dfs-reseed')
                    page.wait_for_function(f"document.getElementById('dfs-msg').textContent !== {json.dumps(seed)} && /seed/.test(document.getElementById('dfs-msg').textContent)")
                    assert page.locator('.dfs-lineup').count() == 10
                    page.click('#dfs-reset')
                    page.wait_for_function("/seed 356\\)/.test(document.getElementById('dfs-msg').textContent)")
                    page.screenshot(path=str(out/f'{device}-fixture-dfs.jpg'),full_page=True,type='jpeg',quality=85)
                    checks.append(f'{device}: DFS fixture · Main, Early and Late slates each build 10 lineups from their own games; tournament mode builds 10 simulated lineups; exposure by salary (2% per $1, max 70%) and flat caps hold in both modes; tournament lineups show the points of the slate they won; New draws reseeds')
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
                assert live.locator('.tracked-line').count()==4
                checks.append('Live-score overlay updated status and scores without changing four frozen selections')
                browser.close()
        finally:
            server.shutdown()
    result='\n'.join(checks)+'\n'
    (out.parent/'CHECKS.txt').write_text(result)
    print(result)


if __name__=='__main__':
    main()
