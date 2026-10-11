"""Same-row, book-separated grading of standalone v6 research forecasts.

Prices enter here only, after forecasts have been fitted and saved.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import market

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'diagnostics/v6_direct_iterations'


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-35,35)))


def interval(g, values, denominator=None):
    """Paired calendar-week bootstrap, retaining weeks with abstentions."""
    d = pd.DataFrame(dict(week=pd.to_datetime(g.slate_date).dt.to_period('W').astype(str),
        value=np.asarray(values), denominator=np.ones(len(g)) if denominator is None
        else np.asarray(denominator)))
    b = d.groupby('week')[['value','denominator']].sum()
    ix = np.random.default_rng(81).integers(0,len(b),(2000,len(b)))
    den = b.denominator.to_numpy()[ix].sum(1)
    num = b.value.to_numpy()[ix].sum(1)
    return tuple(np.quantile(num[den>0]/den[den>0],[.025,.975])) if (den>0).any() else (np.nan,np.nan)


def bets(g, p, threshold=0, lean=False):
    p=np.asarray(p)
    he,ae=p*g.hpayout.to_numpy()-1,(1-p)*g.apayout.to_numpy()-1
    home=p>=.5 if lean else he>=ae
    ev=np.where(home,he,ae)
    keep=np.ones(len(g),bool) if lean else ev>threshold
    won=np.where(home,g.home_won,1-g.home_won)
    payout=np.where(home,g.hpayout,g.apayout)
    q=np.where(home,g.q,1-g.q)
    fav=g.q>=.5
    fav_units=np.where(fav,g.home_won*g.hpayout-1,(1-g.home_won)*g.apayout-1)
    return dict(keep=keep, units=np.where(keep,won*payout-1,0),
                null=np.where(keep,q*payout-1,0),fav_units=np.where(keep,fav_units,0),
                variance=q*(1-q)*payout**2,ev=ev)


def report(scores, returns, selected):
    s,r=pd.DataFrame(scores),pd.DataFrame(returns)
    names=['archived_v6',selected,'phase_projected_r10','phase_stale_r10','stage_stale_r10']
    lines=['# Direct v6 iterations — 2026-10-11', '',
        'Built and tested 41 standalone alternatives to v6 in two passes. No market price is a predictor. '
        'The strongest leads improve v6 slightly but do not demonstrate better log loss than the market '
        'or a repeatable profitable betting rule. Keep production v6 unchanged.', '',
        '## Same-row results', '',
        'Lower log loss is better. Each season/market below contains 1,209 reconstructed regular-season games. '
        '2025 means 2024–25; 2026 means 2025–26. The v6 comparator is chronological walk-forward, '
        'not the ledger’s leave-one-season-out probability.', '',
        '| Candidate | 2024–25 ESPN BET LL | 2025–26 Kalshi LL | Kalshi gap |',
        '|---|---:|---:|---:|']
    for n in names:
        a=s[(s.candidate==n)&(s.season==2025)&(s.book=='espnbet')&(s.segment=='all')].iloc[0]
        b=s[(s.candidate==n)&(s.book=='kalshi')&(s.segment=='all')].iloc[0]
        lines.append(f'| {n} | {a.logloss:.6f} | {b.logloss:.6f} | {b.delta_market:+.6f} |')
    lines += ['| Market | 0.582108 | 0.567740 | — |', '',
        '`projected_r10` was selected using only the first-pass 2024–25 log losses. '
        'Its 2025–26 gain versus v6 is −0.001021 (paired weekly bootstrap 95% CI '
        '−0.003354 to +0.001267). This selector preserves the season boundary, but the '
        'research menu was designed retrospectively; it is not prospective evidence.', '',
        'The second pass combines the observed first-pass leads and therefore reuses both evaluation '
        'seasons. Its best 2025–26 model, `stage_stale_r10`, gains −0.002211 versus v6 '
        '(95% CI −0.004504 to +0.000047). Its 2024–25 gain is only −0.000196. '
        '`phase_projected_r10` has the larger two-season point improvements: '
        '−0.002644 / −0.001400, both unresolved. None is a confirmed upgrade.', '',
        '## ROI at closing quotes', '',
        'One unit per bet. Lean bets every game on P ≥ 0.5; EV rules choose the side with the '
        'larger P × posted payout − 1 and bet only when that EV exceeds the threshold. '
        'ROI is net units / bets. Kalshi posted odds include the repository’s fee approximation. '
        'The market-null ROI is the same bets’ expected profit under the no-vig market probability. '
        'The favourite baseline uses exactly those selected games and their posted prices.', '',
        '| 2025–26 Kalshi candidate | Rule | Bets | ROI | 95% CI | Market null | Favourite ROI |',
        '|---|---|---:|---:|---:|---:|---:|']
    for n in names:
        for rule in ['lean','ev0']:
            a=r[(r.candidate==n)&(r.book=='kalshi')&(r.rule==rule)].iloc[0]
            lines.append(f'| {n} | {rule} | {a.bets} | {a.roi:+.2%} | '
                f'{a.roi_lo:+.2%} to {a.roi_hi:+.2%} | {a.market_null:+.2%} | {a.favorite_roi:+.2%} |')
    lines += ['',
        'No candidate/rule produced positive ROI in both seasons at the respective ESPN BET / Kalshi '
        'prices. All tested Kalshi EV-filtered rules lost money in 2025–26. '
        'The strongest blanket-lean point ROI is +0.72% for `phase_projected_r10`, '
        'but its interval spans losses and its 2024–25 lean ROI is negative. '
        'A lower global log loss does not guarantee calibration of the model–market disagreements '
        'that drive selective bets.', '',
        '## Sportsbook cross-check', '',
        '| 2025–26 book | Games | v6 LL | Selected projected_r10 LL | Market LL |',
        '|---|---:|---:|---:|---:|']
    for book in ['dk','espnbet']:
        a=s[(s.candidate==selected)&(s.season==2026)&(s.book==book)&(s.segment=='all')].iloc[0]
        lines.append(f'| {book} | {a.n} | {a.v6_logloss:.6f} | {a.logloss:.6f} | {a.market_logloss:.6f} |')
    lines += ['',
        'The small ESPN BET early-calendar segment is unresolved; isolated tiny advantages there '
        'do not establish an edge across the main season. Kalshi and sportsbook rows often represent '
        'the same games and are not independent replications.', '',
        '## Build and validation', '',
        '- First pass: nine mixed shooting/possession/FT decay profiles (15/25/40 game half-lives), '
        'four standardized ridge strengths, three seasonal recency half-lives, phase/stage interactions, '
        'injury-adjusted talent, separate stale-talent terms, injury interactions, and three carryover weights.',
        '- Second pass: eight higher-penalty fits plus four combinations of temporal and injury/talent changes.',
        '- Every fold fits on seasons strictly before the evaluated season. Train-only standard deviations '
        'are used for ridge. Features and outcome labels for the evaluated season never enter its fit. '
        'Raw team prefixes exclude the current game; the feature builder checks date boundaries.',
        '- Games 1–9 keep original carryover forecasts for every midseason variant. Carryover variants '
        'change only that early route. Missing feature rows retain the archived production fallback.',
        '- Baseline probability reproduction: maximum error below 9e−16 for both seasons and early routes. '
        'Raw feature recomputation matches all exported training/test tables within 6.1e−13, including finite/missing masks.',
        '- First-pass selection is saved in selection.json. All fits, predictions, book/season/early-versus-midseason scores, '
        'all ROI thresholds (0%, 2.5%, 5%, 10%), nulls, favourite baselines and weekly 95% intervals are saved beside this report.', '',
        '## Limits and next decision', '',
        'These are reconstructed, retrospective results. No native or preseason rows are included. '
        'The 95% intervals are descriptive and unadjusted for testing many variants; selected winners '
        'are optimistic. Two seasons cannot settle small improvements. Closing quote ROI does not '
        'reconstruct decision-time price, executable liquidity, slippage or exact contract-count fee rounding.', '',
        'Mixed-decay variants keep the fold’s original prior-season composite factor weights; they refit '
        'the base/availability logits rather than relearning factor weights for each decay menu. '
        'Availability and roster windows remain at 25 games. The projected-talent arm approximates '
        'pregame roster changes using the archived injury report, not actual participation. '
        'This is a bounded screen, not an exhaustive search.', '',
        'All 1,071 midseason evaluation games in each exported fold were on the availability route. '
        'Base-route logits were trained but have no independent held-out route evidence in this sample.', '',
        'Next: freeze one injury/talent + phase challenger for shadow forecasts on native games, '
        'and broaden outer-season evaluation before promotion. The larger unresolved limitation is '
        'predicting the market’s team-strength correction; none of these variants closes most of that gap.', '',
        '## Reproduce', '',
        '```bash', 'python research/export_v6_inputs.py',
        'python research/v6_direct_iterations.py --inputs research/output/v6_inputs',
        'python research/v6_direct_grading.py', 'python validate_data_files.py',
        'python -m pytest tests/ -q', '```', '',
        'The research-only Actions workflow supplies historical caches and runs the same pipeline. '
        'Initial portable-input export: [run 38123453021]'
        '(https://github.com/Dave356w/nba-matchups/actions/runs/38123453021), '
        'artifact 11695898849. Provenance and checksums are in protocol.json. '
        'No production model, model tag, bot ledger, or site files are changed.', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))


def main():
    f=pd.read_csv(OUT/'forecasts.csv',dtype={'game_id':str})
    names=list(f.columns[3:])
    raw=pd.read_csv(ROOT/'data/nba_reconstructed.csv',dtype={'game_id':str})
    assert len(f)==len(raw)==2418 and raw.basis.eq('reconstructed').all()
    joined=raw.merge(f.drop(columns=['slate_date','season']),on='game_id',validate='one_to_one')
    scores,returns=[],[]
    for kind in ['sportsbook','kalshi']:
        prefix='kalshi_' if kind=='kalshi' else ''
        g=joined[joined[prefix+'close_q_home'].notna()].copy()
        g['q']=g[prefix+'close_q_home']
        g['book']='kalshi' if kind=='kalshi' else g.close_book
        g['hpayout']=g[prefix+'close_home_ml'].map(market.decimal_payout)
        g['apayout']=g[prefix+'close_away_ml'].map(market.decimal_payout)
        assert g[['q','hpayout','apayout']].notna().all().all()
        for (season,book),whole in g.groupby(['season','book']):
            for segment in ['all','games10plus','games1to9']:
                early=np.minimum(whole.gp_home,whole.gp_away)<10
                d=whole if segment=='all' else whole[~early if segment=='games10plus' else early]
                if not len(d):
                    continue
                meta=dict(market=kind,season=int(season),book=book,segment=segment,n=len(d))
                y=d.home_won.to_numpy()
                lv=market.logloss(d.archived_v6.to_numpy(),y)
                lm=market.logloss(d.q.to_numpy(),y)
                for name in names:
                    p=d[name].to_numpy()
                    loss=market.logloss(p,y)
                    dl,dh=interval(d,loss-lv);ml,mh=interval(d,loss-lm)
                    scores.append(dict(**meta,candidate=name,logloss=loss.mean(),
                        v6_logloss=lv.mean(),market_logloss=lm.mean(),delta_v6=(loss-lv).mean(),
                        delta_v6_lo=dl,delta_v6_hi=dh,delta_market=(loss-lm).mean(),
                        delta_market_lo=ml,delta_market_hi=mh,brier=np.mean((p-y)**2),
                        market_brier=np.mean((d.q.to_numpy()-y)**2)))
                    if segment!='all':
                        continue
                    for label,t,lean in [('lean',0,True),('ev0',0,False),('ev2.5',.025,False),
                                         ('ev5',.05,False),('ev10',.1,False)]:
                        b=bets(d,p,t,lean);keep=b['keep'];n=int(keep.sum())
                        if not n:
                            continue
                        lo,hi=interval(d,b['units'],keep)
                        vl,vh=interval(d,b['units']-b['null'],keep)
                        returns.append(dict(**meta,candidate=name,rule=label,bets=n,
                            units=b['units'].sum(),roi=b['units'].sum()/n,roi_lo=lo,roi_hi=hi,
                            market_null=b['null'].sum()/n,favorite_roi=b['fav_units'].sum()/n,
                            excess_null_lo=vl,excess_null_hi=vh,
                            market_null_se=np.sqrt(b['variance'][keep].sum())/n))
    pd.DataFrame(scores).to_csv(OUT/'logloss.csv',index=False)
    pd.DataFrame(returns).to_csv(OUT/'roi.csv',index=False)
    # Choose a single variant on 2024-25 results, then inspect untouched 2025-26.
    s=pd.DataFrame(scores)
    tr=s[(s.season==2025)&(s.segment=='all')&~s.candidate.isin(['archived_v6','v6_refit'])]
    from research.v6_direct_iterations import configs
    second_pass={c['name'] for c in configs()[-12:]}
    tr=tr[~tr.candidate.isin(second_pass)]
    winner=tr.sort_values('logloss').iloc[0].candidate
    (OUT/'selection.json').write_text(json.dumps(dict(rule='lowest 2024-25 ESPN BET log loss',
        candidate=winner,training_season=2025,evaluation_season=2026,
        caveat='Research menu designed retrospectively; season holdout is not prospective evidence.'),indent=2))
    report(scores,returns,winner)
    print('Graded',len(scores),'score rows and',len(returns),'ROI rows. 2025 selection:',winner)


if __name__=='__main__':
    main()
