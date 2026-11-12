#!/usr/bin/env python3
"""Chronological diagnostic of the fixed rolling-50 rule; never a forward save.

Historical closes cannot reproduce the first saved forecast in the five-hour
window. Reconstructed base forecasts use leave-one-season-out fits, which may
include later seasons. This is exploratory calibration replay, not a wholly
out-of-time test of the forecasting pipeline. No parameter selection.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis
import ledger
import market
import realized_forward
import wp_calibration


def replay(data, book, season):
    """Earlier slates only; prior season plus current season; latest 50/cell."""
    g = data[data.close_book.eq(book) & data.season.isin([season-1, season])].copy()
    g = g[g.model_tag.astype(str).str.endswith(analysis.FAMILY_TAGS)
          & g.basis.eq('reconstructed') & ~g.game_id.astype(str).isin(analysis.VOID_LOCKS)]
    g = g.drop_duplicates('game_id', keep='last').sort_values(['slate_date','game_id'])
    cells, out = {}, []
    for date, slate in g.groupby('slate_date', sort=True):
        pending = []
        for _, row in slate.iterrows():
            p, y = float(row.p_home), float(row.home_won)
            q = market.devig(row.close_home_ml, row.close_away_ml)
            stage = wp_calibration.stage(row)
            if not (np.isfinite(p) and 0 <= p <= 1 and y in (0,1) and np.isfinite(q) and stage):
                continue
            home = p >= .5
            pb = wp_calibration.band(p if home else 1-p, wp_calibration.MODEL_BOUNDS)
            qb = wp_calibration.band(q if home else 1-q, wp_calibration.MARKET_BOUNDS)
            key = ('pooled' if stage in ('early','established') else 'opener', pb, qb)
            prior = [r for r in cells.get(key, []) if int(row.season)-1 <= r['season'] <= int(row.season)][-50:]
            n, w = len(prior), sum(r['won'] for r in prior)
            if int(row.season) == season:
                rate = w/n if n else np.nan
                adjusted = wp_calibration.band_adjusted(q if home else 1-q, w, n, [r['market'] for r in prior])
                out.append(dict(version=realized_forward.V1, game_id=str(row.game_id),
                    slate_date=str(date), season=int(row.season), stage=stage, book=book,
                    model_p_home=p, market_p_home=q, realized_p_home=rate if home else 1-rate,
                    adjusted_p_home=adjusted if home else 1-adjusted,
                    home_ml=float(row.close_home_ml), away_ml=float(row.close_away_ml), home_won=y,
                    model_band=pb, market_band=qb, cell_games=n, cell_wins=w,
                    previous_games=sum(r['season']==season-1 for r in prior),
                    latest_evidence_date=max((r['date'] for r in prior), default='')))
            pending.append((key, dict(season=int(row.season), date=str(date), game_id=str(row.game_id), won=int(y if home else 1-y), market=float(q if home else 1-q))))
        # No outcome from this slate enters any of its estimates.
        for key, member in pending:
            cells.setdefault(key, []).append(member)
    return pd.DataFrame(out)


def bootstrap(g, repeats=2000, column='realized_p_home'):
    """Paired slate bootstrap: preserve correlation among games on a date."""
    dates = sorted(g.slate_date.unique())
    if len(dates) < 2:
        return {}
    per_date = []
    for date in dates:
        d = g[g.slate_date.eq(date)]
        delta = ((d[column]-d.home_won)**2-(d.market_p_home-d.home_won)**2).sum()
        bets, units = 0, 0.
        for _, r in d.iterrows():
            side = realized_forward.position(r[column],r.home_ml,r.away_ml)
            if side:
                bets += 1
                units += market.unit_profit(r.home_ml if side=='home' else r.away_ml,(side=='home')==bool(r.home_won))
        per_date.append([len(d),delta,bets,units])
    rng = np.random.default_rng(356)
    a = np.asarray(per_date)
    draws = a[rng.integers(0,len(a),size=(repeats,len(a)))].sum(axis=1)
    return dict(brier_delta_vs_market_95=np.quantile(draws[:,1]/draws[:,0],[.025,.975]).tolist(),
                realized_roi_95=np.quantile(draws[draws[:,2]>0,3]/draws[draws[:,2]>0,2],[.025,.975]).tolist())


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='diagnostics/realized_wp_replay')
    args = parser.parse_args()
    data = ledger.load(ledger.RECON_PATH)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    results, all_predictions = {}, []
    for season, book in [(2025,'espnbet'), (2026,'dk')]:
        g = replay(data,book,season)
        # Determinism and exclusion of the target slate's outcomes.
        shuffled = replay(data.sample(frac=1, random_state=356),book,season)
        pd.testing.assert_frame_equal(g,shuffled)
        check_date = sorted(g.slate_date.unique())[len(g.slate_date.unique())//2]
        changed = data.copy()
        mask = changed.slate_date.eq(check_date)
        changed.loc[mask,'home_won'] = 1-pd.to_numeric(changed.loc[mask,'home_won'])
        perturbed = replay(changed,book,season)
        np.testing.assert_allclose(g.loc[g.slate_date.eq(check_date),'realized_p_home'],
            perturbed.loc[perturbed.slate_date.eq(check_date),'realized_p_home'], equal_nan=True)
        covered = g[g.realized_p_home.notna()].copy()
        assert (covered.latest_evidence_date < covered.slate_date).all() and covered.cell_games.le(50).all()
        grades = covered[['game_id','home_won']].assign(basis='native')  # scorer API only; never persisted as native data
        scores = realized_forward.score(covered, grades)
        # v2 (band-adjusted) on the identical games; reported beside v1's raw rate.
        v2 = covered.assign(realized_p_home=covered.adjusted_p_home)
        scores['estimators']['realized v2 (band-adjusted)'] = realized_forward.score(v2, grades)['estimators']['realized']
        scores['estimators']['realized v1 (raw cell rate)'] = scores['estimators'].pop('realized')
        scores['estimators']['realized v2 (band-adjusted)'] = scores['estimators'].pop('realized v2 (band-adjusted)')
        scores['slate_bootstrap_v2'] = bootstrap(covered, column='adjusted_p_home')
        scores['certainty_errors_v2'] = int(((covered.adjusted_p_home.eq(0)&covered.home_won.eq(1)) |
                                             (covered.adjusted_p_home.eq(1)&covered.home_won.eq(0))).sum())
        scores.update(total_available=len(g), uncovered=len(g)-len(covered),
            median_cell_games=float(covered.cell_games.median()), slate_bootstrap=bootstrap(covered),
            certainty_errors=int(((covered.realized_p_home.eq(0)&covered.home_won.eq(1)) |
                                  (covered.realized_p_home.eq(1)&covered.home_won.eq(0))).sum()))
        label = f'{season-1}-{str(season)[2:]} {book}'
        results[label] = scores
        all_predictions.append(g)
    payload = dict(method='Fixed latest-50 joint cell, v1 raw rate and v2 band-adjusted (game no-vig WP + cell rate − cell mean no-vig WP); previous + current season; earlier slates only; either-side +EV 1u.',
        source_sha256=hashlib.sha256(Path(ledger.RECON_PATH).read_bytes()).hexdigest(),
        limitations=['Exploratory retrospective replay, separate from immutable forward record.',
          'Prices are historical closes, not first valid snapshots within five hours.',
          'Base models are leave-one-season-out; earlier test years can use later training seasons.',
          'No prior-season same-book seed exists for either primary cohort; cold-start cells abstain.',
          '2024-25 ESPN BET is a separate bookmaker sensitivity, not DraftKings evidence.',
          'Replay does not verify execution, publication or visitor views. No thresholds/windows optimized.'],
        results=results)
    (root/'RESULTS.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    pd.concat(all_predictions,ignore_index=True).to_csv(root/'predictions.csv',index=False)
    lines = ['# Rolling-50 retrospective research', '', *['- '+x for x in payload['limitations']], '',
             '| Cohort | Covered / available | Estimator | Brier | Log loss | +EV bets | Units | ROI |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for label,scores in results.items():
        if label != next(iter(results)):
            lines += ['| Cohort | Covered / available | Estimator | Brier | Log loss | +EV bets | Units | ROI |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for name,stats in scores['estimators'].items():
            loss = stats['log_loss']
            loss = f'{loss:.4f}' if isinstance(loss,float) else str(loss)
            lines.append(f"| {label} | {scores['graded_games']} / {scores['total_available']} | {name} | {stats['brier']:.4f} | {loss} | {stats['bets']} | {stats['net_units']:+.2f} | {stats['roi']:.1%} |" if stats['roi'] is not None else
                f"| {label} | {scores['graded_games']} / {scores['total_available']} | {name} | {stats['brier']:.4f} | {loss} | 0 | 0 | — |")
        lines += ['', f"{label}: contradicted 0%/100% estimates v1 {scores['certainty_errors']}, v2 {scores['certainty_errors_v2']}; median cell size {scores['median_cell_games']:.0f}.",
                  'Slate bootstrap 95% intervals, v1: '+json.dumps(scores['slate_bootstrap']),
                  'Slate bootstrap 95% intervals, v2: '+json.dumps(scores['slate_bootstrap_v2']), '']
    lines += ['No historical observation is added to the shadow forward files or public records.',
              'Reproduce: `python research/realized_wp_replay.py`. Causal evidence selection and input-order checks passed.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
