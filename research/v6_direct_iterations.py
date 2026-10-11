"""Direct market-free v6 refits from portable pregame folds and team logs.

Usage: python research/v6_direct_iterations.py --inputs PATH
Every fit uses seasons earlier than the evaluated season. Market prices
enter only grading (log loss and quote-based flat-stake ROI).
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nba_composite as nc
import player_availability as pav
import cold_start
import market
from research.v6_direct_grading import sigmoid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'diagnostics/v6_direct_iterations'
PROFILES = {
    'baseline': (25,25,25), 'shoot15': (15,25,25), 'shoot40': (40,25,25),
    'pos15': (25,15,25), 'pos40': (25,40,25),
    'shoot15_pos40': (15,40,25), 'shoot40_pos15': (40,15,25),
    'ft40': (25,25,40), 'all15': (15,15,15), 'all40': (40,40,40),
}


def configs():
    out = [dict(name='v6_refit', profile='baseline', form='base', ridge=None, recency=None)]
    out += [dict(name=f'decay_{p}', profile=p, form='base', ridge=None, recency=None)
            for p in PROFILES if p!='baseline']
    out += [dict(name=f'ridge_{r:g}', profile='baseline', form='base', ridge=r, recency=None)
            for r in (.1,1,10,100)]
    out += [dict(name=f'recent_{h}', profile='baseline', form='base', ridge=None, recency=h)
            for h in (1,2,3)]
    out += [dict(name=f'{form}_r{r}', profile='baseline', form=form, ridge=r, recency=None)
            for form in ('phase','stage','projected','stale','injury_interactions') for r in (1,10)]
    # Second pass, motivated by the first pass: exploratory on the same seasons.
    out += [dict(name=f'{form}_r{r}', profile='baseline', form=form, ridge=r, recency=None)
            for form in ('phase','stage','projected','stale') for r in (30,100)]
    out += [dict(name=f'{form.replace("+","_")}_r10', profile='baseline', form=form,
                 ridge=10, recency=None) for form in
            ('phase+stale','stage+stale','phase+projected','stage+projected')]
    return out


def prefix_totals(log, half_life):
    """Index i contains games 0..i-1 only; identical to production decay."""
    cols = nc.COLS + nc.X3
    a = log.reindex(columns=cols).to_numpy(float)
    t = np.zeros((len(a)+1,len(cols)))
    decay = .5**(1/half_life)
    for i in range(len(a)):
        t[i+1] = decay*t[i]+a[i]
    return t


class RawFeatures:
    def __init__(self, inputs):
        self.logs, self.totals, self.leagues = {}, {}, {}
        for p in sorted(inputs.glob('log_*.csv')):
            _, year, team = p.stem.split('_')
            g = pd.read_csv(p, parse_dates=['date'])
            g = g.sort_values('date').reset_index(drop=True)
            self.logs.setdefault(int(year), {})[team] = g
            for hl in (15,25,40):
                self.totals[int(year),team,hl] = prefix_totals(g,hl)
        for y, logs in self.logs.items():
            self.leagues[y] = nc.league_3p(logs)

    def side(self, year, team, i, profile, pct):
        shoot, pos, ft = PROFILES[profile]
        hl = [shoot,pos,pos,ft,shoot,pos,pos,ft]
        vals, corrections = {}, {}
        cols = nc.COLS + nc.X3
        for h in set(hl):
            a = self.totals[year,team,h][i]
            f = nc.features_from_totals(a[:len(nc.COLS)])
            t = dict(zip(cols,a))
            reg = nc.regress_3p(t,'O',pct)
            corr = nc.features_from_totals(np.array([reg[c] for c in nc.COLS]))-f
            vals[h], corrections[h] = f,corr
        f = np.array([vals[h][k] for k,h in enumerate(hl)])
        luck = np.array([corrections[h][k] for k,h in enumerate(hl)])
        a = self.totals[year,team,ft][i]
        ftp = 100*a[nc.COLS.index('TFT')]/a[nc.COLS.index('TFTA')]
        return f,luck,ftp

    def apply(self, frame, weights, profile):
        d = frame.copy()
        out = []
        for r in d.itertuples():
            y, date = int(r.year), pd.Timestamp(r.date)
            for team,i in [(r.home,int(r.gp_home)),(r.away,int(r.gp_away))]:
                dates=self.logs[y][team].date
                assert 0<i<=len(dates)
                assert dates.iloc[i-1]<date
                assert i==len(dates) or dates.iloc[i]>=date
            pct = self.leagues[y].get(date,np.nan)
            h = self.side(y,r.home,int(r.gp_home),profile,pct)
            a = self.side(y,r.away,int(r.gp_away),profile,pct)
            delta = nc.composite(h[0]-a[0],weights['sd'],weights['w'])
            luck = nc.composite(h[1]-a[1],weights['sd'],weights['w'])
            out.append((delta,delta*r.phase,luck,h[2]-a[2]))
        d[['delta','d_phase','luck_def','ft_diff']] = np.asarray(out)
        return d


def add_features(d):
    d = d.copy()
    d['ft_diff'] = 100*d.ftp_gap
    for f in ['luck_def','talent_diff','ft_diff','av_min','av_bpm','delta']:
        d[f+'_phase'] = d[f]*d.phase
        d[f+'_stage'] = d[f]*(np.minimum(d.gp_home,d.gp_away)<25)
    d['talent_projected'] = d.talent_diff-d.newly_out_diff+d.returning_listed_diff
    d['talent_projected_phase'] = d.talent_projected*d.phase
    d['talent_projected_stage'] = d.talent_projected*(np.minimum(d.gp_home,d.gp_away)<25)
    d['injury_talent'] = d.av_bpm*np.abs(d.talent_diff)
    d['injury_strength'] = d.av_min*np.abs(d.delta)
    return d


def features(route, form):
    f = list(pav.FEATURES_V6 if route=='avail' else nc.V6_FEATURES)
    forms=form.split('+')
    temporal=next((x for x in forms if x in ('phase','stage')),None)
    if temporal:
        f += [x+'_'+temporal for x in ['luck_def','talent_diff','ft_diff','delta']
              if x+'_'+temporal != 'delta_phase']
        if route=='avail':
            f += [x+'_'+temporal for x in ['av_min','av_bpm']]
    if route=='avail':
        if 'projected' in forms:
            f = [x.replace('talent_diff','talent_projected') for x in f]
        if 'stale' in forms:
            f += ['newly_out_diff','returning_listed_diff']
        if form=='injury_interactions':
            f += ['injury_talent','injury_strength']
    return f


def fit_predict(train, test, feats, cfg, year):
    ok = np.isfinite(train[feats].to_numpy(float)).all(1)
    tr = train.loc[ok]
    assert (tr.year<year).all()
    x, xt = tr[feats].to_numpy(float),test[feats].to_numpy(float)
    y = tr.win.to_numpy(float)
    if cfg['ridge'] is None and cfg['recency'] is None:
        m = nc.fit_logit(x,y,feats)
        return nc.predict(m,xt), dict(n_train=len(tr),model=m)
    sd = np.ones(x.shape[1]) if cfg['ridge'] is None else x.std(0)
    sd[sd<1e-8] = 1
    x, xt = x/sd,xt/sd
    x,xt = np.column_stack([np.ones(len(x)),x]),np.column_stack([np.ones(len(xt)),xt])
    w = np.ones(len(tr))
    if cfg['recency'] is not None:
        w = .5**((year-1-tr.year.to_numpy(float))/cfg['recency'])
        w /= w.mean()
    penalty = .001 if cfg['ridge'] is None else cfg['ridge']
    pen = np.diag([0]+[2*penalty]*(x.shape[1]-1))
    beta = np.zeros(x.shape[1])
    for _ in range(75):
        p = sigmoid(x@beta)
        step = np.linalg.solve(x.T@(x*(w*p*(1-p))[:,None])+pen,
                               x.T@(w*(p-y))+pen@beta)
        beta -= step
        if abs(step).max()<1e-9:
            break
    m = dict(features=feats,intercept=float(beta[0]),coef=(beta[1:]/sd).tolist())
    return sigmoid(xt@beta),dict(n_train=len(tr),model=m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs',type=Path,required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    raw = RawFeatures(args.inputs)
    archive = pd.read_csv(OUT/'baseline.csv',dtype={'game_id':str})
    result, fitted, checks, feature_checks = [],[],[],[]
    for year in (2025,2026):
        weights = json.loads((args.inputs/f'weights_{year}.json').read_text())
        tables = {k:add_features(pd.read_csv(args.inputs/f'{k}_{year}.csv',parse_dates=['date']))
                  for k in ('base_train','avail_train','test')}
        for name,d in tables.items():
            reproduced=raw.apply(d,weights,'baseline')
            errors={}
            for col in ['delta','d_phase','luck_def','ft_diff']:
                assert np.array_equal(np.isfinite(d[col]),np.isfinite(reproduced[col]))
                errors[col]=float(np.nanmax(np.abs(d[col]-reproduced[col])))
                assert errors[col]<1e-7,(year,name,col,errors[col])
            feature_checks.append(dict(season=year,table=name,n=len(d),errors=errors))
        cache = {'baseline':tables}
        test = tables['test']
        ref = archive[archive.year==year].copy()
        key = ['slate_date','home','away']
        test = test.merge(ref[key+['game_id','p_prod']],on=key,validate='one_to_one')
        assert len(test)==1071
        pred = ref[['game_id','slate_date','year','p_prod']].copy().rename(columns={'year':'season','p_prod':'archived_v6'})
        pred = pred.set_index('game_id')
        for cfg in configs():
            profile = cfg['profile']
            if profile not in cache:
                cache[profile] = {k:add_features(raw.apply(v,weights,profile).assign(
                    ftp_gap=lambda d:d.ft_diff/100)) for k,v in tables.items()}
            tabs = cache[profile]
            te = tabs['test'].merge(test[key+['game_id','p_prod']],on=key,validate='one_to_one')
            p = te.p_prod.to_numpy(float).copy()
            for route in ('base','avail'):
                feats = features(route,cfg['form'])
                hit = (te.route==route)&np.isfinite(te[feats].to_numpy(float)).all(1)
                if hit.any():
                    pr,m = fit_predict(tabs['avail_train' if route=='avail' else 'base_train'],
                                       te.loc[hit],feats,cfg,year)
                    p[hit] = pr
                    fitted.append(dict(season=year,candidate=cfg['name'],route=route,**m))
            assert np.isfinite(p).all()
            pred[cfg['name']] = pred.archived_v6
            pred.loc[te.game_id,cfg['name']] = p
            if cfg['name']=='v6_refit':
                error=float(np.max(np.abs(p-te.p_prod.to_numpy())))
                checks.append(dict(season=year,max_baseline_error=error))
                assert error<1e-7, ('baseline does not reproduce',year,error)
            print(year,cfg['name'],'done',flush=True)
        # Carryover-only iterations: current games 10+ stay unchanged.
        for rho in (.1,.25,.5,.75):
            train = pd.concat([cold_start.early_games(y,weights,logs=raw.logs[y],
                prior_logs=raw.logs[y-1],rho=rho) for y in (2023,2024,2025) if y<year])
            te = cold_start.early_games(year,weights,logs=raw.logs[year],
                prior_logs=raw.logs[year-1],rho=rho,min_gp=1,window=10)
            te['slate_date'] = te.date.dt.strftime('%Y-%m-%d')
            te=te.merge(ref[key+['game_id','p_prod']],on=key,validate='one_to_one')
            m=nc.fit_logit(train[cold_start.FEATURES].to_numpy(float),train.win,cold_start.FEATURES)
            pr=nc.predict(m,te[cold_start.FEATURES].to_numpy(float))
            if rho==.25:
                error=float(np.max(np.abs(pr-te.p_prod.to_numpy())))
                checks.append(dict(season=year,max_early_baseline_error=error))
                assert error<1e-7, ('early baseline mismatch',year,error)
                continue
            name=f'early_rho_{rho:g}'
            pred[name]=pred.archived_v6
            pred.loc[te.game_id,name]=pr
            fitted.append(dict(season=year,candidate=name,route='early',model=m,n_train=len(train)))
        result.append(pred.reset_index())
    p=pd.concat(result,ignore_index=True)
    p.to_csv(OUT/'forecasts.csv',index=False)
    (OUT/'fits.json').write_text(json.dumps(fitted,indent=2))
    (OUT/'baseline_checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'feature_checks.json').write_text(json.dumps(feature_checks,indent=2))
    print('All direct fits complete; baseline reproduced.',flush=True)


if __name__=='__main__':
    main()
