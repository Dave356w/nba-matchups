"""Append-only, production-only shadow forecasts. Never reconstruct past captures.

See docs/realized_wp_forward_registration.md for the pre-registered rules.
No output from this module belongs in public/.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

import ledger
import market
import wp_calibration as calibration

V1 = 'realized_wp_rolling50_v1'  # raw cell rate; closed, kept, never pooled
VERSION = 'realized_wp_rolling50_bandadj_v2'
REGISTERED_BY_VERSION = {V1: '2026-11-12T21:50:58Z', VERSION: '2026-11-12T22:35:12Z'}
REGISTERED = REGISTERED_BY_VERSION[VERSION]


def expected_pick_probability(version, q_pick, wins, games, members):
    """The selected team's estimate each version registered, from its proof."""
    if version == V1:
        return wins/games
    return calibration.band_adjusted(q_pick, wins, games, [m['market_probability'] for m in members])
COLUMNS = ['snapshot_id', 'version', 'captured_utc', 'forecast_utc', 'tip_utc',
           'game_id', 'slate_date', 'season', 'stage', 'home', 'away', 'model_tag',
           'code_revision', 'book', 'model_pick', 'model_p_home', 'market_p_home',
           'realized_p_home', 'displayed_realized_wp', 'home_ml', 'away_ml',
           'cell_games', 'cell_wins', 'previous_games', 'current_games', 'evidence_id']
EVIDENCE_COLUMNS = ['evidence_id', 'evidence_json']


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,
                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load(path, columns):
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != columns:
        raise ValueError(f'{path}: invalid shadow schema')
    return frame


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append(path, rows, columns):
    """Preserve existing values exactly; reject conflicting immutable IDs."""
    old = load(path, columns)
    if old[columns[0]].duplicated().any():
        raise ValueError(f'{path}: duplicate immutable ID')
    by_id = {r[columns[0]]: r for r in old.to_dict('records')}
    added = []
    for row in rows:
        row = {key: str(row[key]) for key in columns}
        key = row[columns[0]]
        if key in by_id:
            # Capture time is assigned only on first save, not part of identity.
            compare = [c for c in columns if c != 'captured_utc']
            if any(by_id[key][c] != row[c] for c in compare):
                raise ValueError(f'{path}: conflicting immutable ID {key}')
        else:
            by_id[key] = row
            added.append(row)
    if added:
        frame = pd.concat([old, pd.DataFrame(added, columns=columns)], ignore_index=True)
        atomic_text(path, frame.to_csv(index=False))
    return len(added)


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def snapshot(row, estimate, now, code_revision):
    """Return a new forward capture, or None. Uses the actual recording clock."""
    tip = pd.to_datetime(row.get('tip_utc'), utc=True, errors='coerce')
    forecast = pd.to_datetime(row.get('snapshot_utc'), utc=True, errors='coerce')
    now = pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_convert('UTC')
    if (now.tzinfo is None or pd.isna(tip) or pd.isna(forecast)
            or now < pd.Timestamp(REGISTERED) or not forecast <= now < tip
            or pd.notna(row.get('home_won')) or not estimate or not estimate['games']):
        return None
    p = float(row['p_home'])
    q = market.devig(row.get('pre_home_ml'), row.get('pre_away_ml'))
    r = float(estimate['probability'])
    if (row.get('pre_book') != 'dk' or not all(math.isfinite(v) and 0 <= v <= 1 for v in (p, q, r))):
        return None
    home = p >= .5
    proof = dict(window=calibration.WINDOW_GAMES, model_range=estimate['model_range'],
                 market_range=estimate['market_range'], members=estimate['evidence'])
    proof_id = digest(proof)
    values = dict(version=VERSION, captured_utc=now.isoformat(), forecast_utc=forecast.isoformat(),
                  tip_utc=tip.isoformat(), game_id=str(row['game_id']), slate_date=str(row['slate_date']),
                  season=str(int(row['season'])), stage=calibration.stage(row), home=str(row['home']),
                  away=str(row['away']), model_tag=str(row['model_tag']), code_revision=code_revision,
                  book='dk', model_pick='home' if home else 'away', model_p_home=p, market_p_home=q,
                  realized_p_home=r if home else 1-r, displayed_realized_wp=f'{100*r:.1f}%',
                  home_ml=float(row['pre_home_ml']), away_ml=float(row['pre_away_ml']),
                  cell_games=estimate['games'], cell_wins=estimate['wins'],
                  previous_games=estimate['previous_games'], current_games=estimate['current_games'],
                  evidence_id=proof_id)
    # An unchanged native forecast rendered again is not an independent capture.
    identity = {k: str(v) for k, v in values.items() if k not in ('captured_utc', 'code_revision')}
    values['snapshot_id'] = digest(identity)
    return values, dict(evidence_id=proof_id, evidence_json=canonical(proof))


def capture(site, data_dir='data', now=None, code_revision=None):
    code_revision = code_revision or revision()
    day = site.native[site.native.slate_date.astype(str).eq(site.today)]
    count = 0
    for _, row in day.iterrows():
        estimate = site.calibration.estimate(row)
        item = snapshot(row, estimate, now if now is not None else ledger.utc_now(), code_revision)
        if not item:
            continue
        season, root = int(row.season), Path(data_dir)
        # Persist proof before assigning the recording time. If processing or
        # disk I/O crosses tip, keep only an orphan proof, never a late capture.
        append(root / f'realized_wp_cells_{season}.csv', [item[1]], EVIDENCE_COLUMNS)
        path = root / f'realized_wp_snapshots_{season}.csv'
        old = load(path, COLUMNS).set_index('snapshot_id')
        item = snapshot(row, estimate, now if now is not None else ledger.utc_now(), code_revision)
        if not item:
            continue
        values = item[0]
        # Unchanged content survives a code-only deploy with its original revision.
        if values['snapshot_id'] in old.index:
            values['code_revision'] = old.loc[values['snapshot_id'], 'code_revision']
        count += append(path, [values], COLUMNS)
    return count


def read_all(data_dir='data'):
    files = sorted(Path(data_dir).glob('realized_wp_snapshots_*.csv'))
    return pd.concat([load(p, COLUMNS) for p in files], ignore_index=True) if files else pd.DataFrame(columns=COLUMNS)


def official(frame):
    if frame.empty:
        return frame.copy()
    capture_at = pd.to_datetime(frame.captured_utc, utc=True)
    tip = pd.to_datetime(frame.tip_utc, utc=True)
    forecast = pd.to_datetime(frame.forecast_utc, utc=True)
    # A stale source forecast does not become fresh merely by rendering it.
    eligible = (capture_at >= tip-pd.Timedelta(hours=5)) & (capture_at < tip)
    eligible &= (forecast >= tip-pd.Timedelta(hours=5)) & (forecast <= capture_at)
    return (frame[eligible].assign(_capture_at=capture_at[eligible]).sort_values(['_capture_at', 'snapshot_id'])
            .drop_duplicates(['version', 'game_id'], keep='first').drop(columns='_capture_at').copy())


def position(p, home_ml, away_ml):
    ev_home = p * market.decimal_payout(home_ml) - 1
    ev_away = (1-p) * market.decimal_payout(away_ml) - 1
    if max(ev_home, ev_away) <= 0:
        return None
    return 'home' if ev_home >= ev_away else 'away'


def score(frame, native):
    """Paired probability scores and fixed-price returns; grades only from native."""
    actual = {str(r.game_id): float(r.home_won) for _, r in native.iterrows()
              if r.get('basis') == 'native' and pd.notna(r.get('home_won')) and r.home_won in (0, 1)}
    graded = [(r, actual[str(r.game_id)]) for _, r in frame.iterrows() if str(r.game_id) in actual]
    result = dict(official_games=len(frame), graded_games=len(graded), pending_games=len(frame)-len(graded), estimators={})
    for name, column in [('model', 'model_p_home'), ('market', 'market_p_home'), ('realized', 'realized_p_home')]:
        brier, losses, correct, units, bets = [], [], 0, 0., 0
        for row, y in graded:
            p = float(row[column])
            brier.append((p-y)**2)
            py = p if y else 1-p
            losses.append(-math.log(py) if py else math.inf)
            correct += int((p >= .5) == bool(y))
            side = position(p, float(row.home_ml), float(row.away_ml))
            if side:
                bets += 1
                units += market.unit_profit(float(row.home_ml if side == 'home' else row.away_ml), (side == 'home') == bool(y))
        loss = sum(losses)/len(losses) if losses else None
        result['estimators'][name] = dict(brier=sum(brier)/len(brier) if brier else None,
            log_loss='infinity' if loss == math.inf else loss,
            winner_accuracy=correct/len(graded) if graded else None,
            bets=bets, net_units=units, roi=units/bets if bets else None)
    units = sum(market.unit_profit(float(r.home_ml if float(r.market_p_home) >= .5 else r.away_ml),
                (float(r.market_p_home) >= .5) == bool(y)) for r, y in graded)
    result['market_favorite'] = dict(bets=len(graded), net_units=units, roi=units/len(graded) if graded else None)
    return result


def report(native, data_dir='data', out_dir='diagnostics/realized_wp_forward'):
    all_rows = read_all(data_dir)
    selected = official(all_rows)
    versions = {}
    for version in sorted(set(all_rows.version) | {VERSION}):
        rows = selected[selected.version.eq(version)]
        versions[version] = dict(pooled=score(rows, native), stages={
            stage: score(rows[rows.stage.eq(stage)], native) for stage in ('opener', 'early', 'established')})
    actual = {str(r.game_id): float(r.home_won) for _, r in native.iterrows()
              if r.get('basis') == 'native' and pd.notna(r.get('home_won')) and r.home_won in (0, 1)}
    observations = []
    for row in selected.to_dict('records'):
        y = actual.get(str(row['game_id']))
        item = dict(snapshot_id=row['snapshot_id'], version=row['version'], game_id=row['game_id'],
                    captured_utc=row['captured_utc'], tip_utc=row['tip_utc'], home_won=y, estimators={})
        for name in ('model', 'market', 'realized'):
            p = float(row[f'{name}_p_home'])
            side = position(p, float(row['home_ml']), float(row['away_ml']))
            price = float(row[f'{side}_ml']) if side else None
            ev = (p if side == 'home' else 1-p)*market.decimal_payout(price)-1 if side else None
            units = market.unit_profit(price, (side == 'home') == bool(y)) if side and y is not None else None
            item['estimators'][name] = dict(p_home=p, side=side, price=price,
                                           expected_net_return=ev, net_units=units)
        observations.append(item)
    payload = dict(shadow_research=True, registered_utc=REGISTERED_BY_VERSION, captures=len(all_rows),
                   versions=versions, official_observations=observations)
    atomic_text(Path(out_dir)/'RESULTS.json', json.dumps(payload, indent=2, allow_nan=False)+'\n')
    lines = ['# Realized WP shadow forward research', '',
             f'Current version {VERSION}: this game\'s no-vig WP + (cell win rate − cell mean no-vig WP), clipped to 0–100%.',
             f'{V1} (raw cell win rate) is closed; its captures stay graded under their own version, never pooled.', '',
             'First valid saved forecast within five hours of tip. No backfill. Each source forecast must also be within that window.',
             'Flat 1u only at positive expected return; either side; at most one position per estimator per game.',
             'All probability scores use the same official graded games, including abstentions. Saved prices are paper quotes.',
             'Build-time saves do not prove publication, viewing or executable prices. Public records and picks are unchanged.',
             '', f'Saved distinct pre-tip captures: {len(all_rows)}.', '']
    for version, values in versions.items():
        lines += [f'## {version}', '', f"Official games: {values['pooled']['official_games']}; graded: {values['pooled']['graded_games']}; pending: {values['pooled']['pending_games']}.", '']
        for scope, scores in [('pooled', values['pooled']), *values['stages'].items()]:
            lines += [f'### {scope}', '', '| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |',
                      '| --- | --- | --- | --- | --- | --- | --- |']
            for name, stats in scores['estimators'].items():
                fmt = lambda v: '—' if v is None else f'{v:.4f}' if isinstance(v, float) else str(v)
                lines += ['| '+' | '.join([name]+[fmt(stats[k]) for k in ('brier','log_loss','winner_accuracy','bets','net_units','roi')])+' |']
            lines += ['', 'Same-game market favorite: '+canonical(scores['market_favorite']), '']
    atomic_text(Path(out_dir)/'RESULTS.md', '\n'.join(lines)+'\n')
    return payload


def validate(data_dir='data'):
    """Fail closed on malformed saves, missing proofs or noncausal evidence."""
    errors = []
    for path in sorted(Path(data_dir).glob('realized_wp_snapshots_*.csv')):
        try:
            frame = load(path, COLUMNS)
            proofs = load(path.with_name(path.name.replace('snapshots', 'cells')), EVIDENCE_COLUMNS)
            if frame.snapshot_id.duplicated().any() or proofs.evidence_id.duplicated().any():
                raise ValueError('duplicate immutable IDs')
            evidence = {}
            for _, proof in proofs.iterrows():
                value = json.loads(proof.evidence_json)
                if digest(value) != proof.evidence_id:
                    raise ValueError('evidence hash mismatch')
                evidence[proof.evidence_id] = value
            for _, r in frame.iterrows():
                identity = {k: str(r[k]) for k in COLUMNS if k not in ('snapshot_id','captured_utc','code_revision')}
                if digest(identity) != r.snapshot_id:
                    raise ValueError('snapshot hash mismatch')
                at, forecast, tip = [pd.to_datetime(r[k], utc=True, errors='raise') for k in ('captured_utc','forecast_utc','tip_utc')]
                if r.version not in REGISTERED_BY_VERSION:
                    raise ValueError('unregistered version, book or stage')
                if not pd.Timestamp(REGISTERED_BY_VERSION[r.version]) <= at < tip or not forecast <= at:
                    raise ValueError('invalid forward timestamps')
                if r.book != 'dk' or r.stage not in ('opener','early','established'):
                    raise ValueError('unregistered version, book or stage')
                if not r.code_revision or r.model_pick not in ('home','away'):
                    raise ValueError('missing provenance')
                p, q, realized = [float(r[k]) for k in ('model_p_home','market_p_home','realized_p_home')]
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in (p, q, realized)):
                    raise ValueError('invalid probabilities')
                if q != market.devig(float(r.home_ml), float(r.away_ml)):
                    raise ValueError('invalid saved price or no-vig probability')
                home = p >= .5
                if r.model_pick != ('home' if home else 'away'):
                    raise ValueError('inconsistent model side')
                n, w, previous, current = [int(r[k]) for k in ('cell_games','cell_wins','previous_games','current_games')]
                if not 0 <= w <= n <= 50 or n == 0 or previous+current != n:
                    raise ValueError('invalid cell count')
                proof = evidence[r.evidence_id]
                members = proof['members']
                pick = expected_pick_probability(r.version, q if home else 1-q, w, n, members)
                if not math.isclose(realized if home else 1-realized, pick, abs_tol=1e-12):
                    raise ValueError('cell estimate disagrees with frozen forecast')
                if (not r.displayed_realized_wp.endswith('%')
                        or abs(float(r.displayed_realized_wp[:-1]) - 100*pick) > .05 + 1e-9):
                    raise ValueError('displayed rate mismatch')
                if proof['window'] != 50 or len(members) != n or sum(m['won'] for m in members) != w:
                    raise ValueError('evidence count mismatch')
                if len(set(m['game_id'] for m in members)) != n:
                    raise ValueError('duplicate cell observation')
                season = int(r.season)
                pb = calibration.band(p if home else 1-p, calibration.MODEL_BOUNDS)
                qb = calibration.band(q if home else 1-q, calibration.MARKET_BOUNDS)
                if (proof['model_range'] != list(calibration.MODEL_BOUNDS[pb:pb+2])
                        or proof['market_range'] != list(calibration.MARKET_BOUNDS[qb:qb+2])):
                    raise ValueError('evidence cell boundary mismatch')
                if sum(m['season'] == season-1 for m in members) != previous:
                    raise ValueError('season composition mismatch')
                for m in members:
                    if (m['game_id'] == r.game_id or m['date'] >= r.slate_date
                            or not ((m['season'] == season and m['source'] == 'live') or m['season'] == season-1)
                            or m['won'] not in (0,1)
                            or m['model_band'] != pb or m['market_band'] != qb
                            or calibration.band(m['model_probability'], calibration.MODEL_BOUNDS) != pb
                            or calibration.band(m['market_probability'], calibration.MARKET_BOUNDS) != qb
                            or not math.isfinite(market.decimal_payout(m['price']))
                            or (m['stage'] not in ('early','established') if r.stage in ('early','established') else m['stage'] != 'opener')):
                        raise ValueError('noncausal or mismatched cell evidence')
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(f'{path}: {exc}')
    return errors
