"""Cell ROI forward log (docs/cell_roi_forward_registration.md).

At each valid native v11 five-hour lock, before tip, save once the game's
model-WP x market-WP cell and that cell's trailing record as the card shows
it (latest 200 results on strictly earlier slates, at each result's own
price), read with the LOCK probability and prices. Results are never saved
here; the report grades from the native ledger. Monitoring only: nothing
here selects, filters or grades a pick, or enters a model or a record.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import analysis
import ledger
import market
import realized_forward as rf
import wp_calibration as calibration

VERSION = 'cell_roi_lock_v1'
REGISTERED = '2026-11-18T18:40:00Z'
COLUMNS = ['game_id', 'version', 'captured_utc', 'lock_snapshot_utc', 'tip_utc', 'slate_date',
           'season', 'stage', 'home', 'away', 'lock_model_tag', 'code_revision', 'pick',
           'pick_side', 'lock_p_pick', 'lock_q_pick', 'pick_ml', 'model_band', 'market_band',
           'cell_games', 'cell_wins', 'cell_priced', 'cell_units', 'cell_roi', 'cell_roi_se',
           'previous_games', 'current_games']
OUT_DIR = Path('diagnostics/cell_roi_forward')


def _number(value):
    return pd.to_numeric(value, errors='coerce')


def lock_view(row):
    """The row as its lock saw it: lock P, tag, book and prices in the
    fields RollingCalibration.estimate reads; None without a valid lock."""
    if str(row.get('game_id')) in analysis.VOID_LOCKS:
        return None
    if not str(row.get('lock_model_tag', '')).endswith(analysis.FAMILY_TAGS) or row.get('lock_book') != 'dk':
        return None
    p, home_ml, away_ml = (_number(row.get(k)) for k in ('lock_p_home', 'lock_home_ml', 'lock_away_ml'))
    if not (np.isfinite(p) and 0 <= p <= 1 and np.isfinite(market.devig(home_ml, away_ml))):
        return None
    view = row.copy()
    view['p_home'], view['model_tag'], view['pre_book'] = p, row['lock_model_tag'], 'dk'
    view['pre_home_ml'], view['pre_away_ml'] = home_ml, away_ml
    return view


def cell_record(evidence):
    """1u ROI on the model pick over a cell's members, each at its own price."""
    units = [u for u in (market.unit_profit(r['price'], r['won']) for r in evidence) if np.isfinite(u)]
    total = math.fsum(units)
    # same rule as the pages: no SE for one result, or when all are wins or all losses
    se = float('nan') if len(units) < 2 or all(u > 0 for u in units) or all(u < 0 for u in units) \
        else float(np.std(units, ddof=1) / math.sqrt(len(units)))
    return len(units), total, (total / len(units) if units else float('nan')), se


def entry(row, calib, now, code_revision):
    """The game's log row at `now`, or None (no valid lock, not before tip,
    already final, or tipped before the registration)."""
    tip = pd.to_datetime(row.get('tip_utc'), utc=True, errors='coerce')
    lock = pd.to_datetime(row.get('lock_snapshot_utc'), utc=True, errors='coerce')
    now = pd.Timestamp(now).tz_convert('UTC') if pd.Timestamp(now).tzinfo else None
    if (now is None or pd.isna(tip) or pd.isna(lock) or tip <= pd.Timestamp(REGISTERED)
            or not lock <= now < tip or pd.notna(_number(row.get('home_won')))):
        return None
    view = lock_view(row)
    est = calib.estimate(view) if view is not None else None
    if est is None:
        return None
    home = float(view['p_home']) >= .5
    priced, units, roi, se = cell_record(est['evidence'])
    fmt = lambda x: '' if not np.isfinite(x) else repr(float(x))  # noqa: E731
    p_pick = float(view['p_home']) if home else 1 - float(view['p_home'])
    return dict(game_id=str(row['game_id']), version=VERSION, captured_utc=now.isoformat(),
                lock_snapshot_utc=lock.isoformat(), tip_utc=tip.isoformat(),
                slate_date=str(row['slate_date']), season=str(int(row['season'])),
                stage=calibration.stage(view), home=str(row['home']), away=str(row['away']),
                lock_model_tag=str(row['lock_model_tag']), code_revision=code_revision,
                pick=str(row['home'] if home else row['away']), pick_side='home' if home else 'away',
                lock_p_pick=repr(p_pick), lock_q_pick=repr(float(est['market_probability'])),
                pick_ml=repr(float(view['pre_home_ml'] if home else view['pre_away_ml'])),
                model_band=calibration.band(p_pick, calibration.MODEL_BOUNDS),
                market_band=calibration.band(est['market_probability'], calibration.MARKET_BOUNDS),
                cell_games=est['games'], cell_wins=est['wins'], cell_priced=priced,
                cell_units=repr(units), cell_roi=fmt(roi), cell_roi_se=fmt(se),
                previous_games=est['previous_games'], current_games=est['current_games'])


def path_for(season, data_dir='data'):
    return Path(data_dir) / f'cell_roi_forward_{int(season)}.csv'


def capture(site, data_dir='data', now=None, code_revision=None):
    """Append each locked, not-yet-logged game of the native ledger; write-once."""
    now = now if now is not None else ledger.utc_now()
    code_revision = code_revision or rf.revision()
    count = 0
    for season, rows in site.native.groupby('season'):
        path = path_for(season, data_dir)
        logged = set(rf.load(path, COLUMNS).game_id)
        new = [e for e in (entry(r, site.calibration, now, code_revision)
                           for _, r in rows.iterrows() if str(r.game_id) not in logged) if e]
        count += rf.append(path, new, COLUMNS) if new else 0
    return count


def read_all(data_dir='data'):
    files = sorted(Path(data_dir).glob('cell_roi_forward_*.csv'))
    return pd.concat([rf.load(p, COLUMNS) for p in files], ignore_index=True) if files else pd.DataFrame(columns=COLUMNS)


def validate(data_dir='data'):
    """Problems in the log: identity, timing, bands and internal arithmetic."""
    errors = []
    for path in sorted(Path(data_dir).glob('cell_roi_forward_*.csv')):
        try:
            frame = rf.load(path, COLUMNS)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if frame.game_id.duplicated().any():
            errors.append(f'{path}: duplicate game')
        for r in frame.itertuples():
            where = f'{path}: {r.game_id}'
            try:
                lock, cap, tip = (pd.Timestamp(x) for x in (r.lock_snapshot_utc, r.captured_utc, r.tip_utc))
                p, q, ml = float(r.lock_p_pick), float(r.lock_q_pick), float(r.pick_ml)
                n, w, priced = int(r.cell_games), int(r.cell_wins), int(r.cell_priced)
                units = float(r.cell_units)
            except (ValueError, TypeError) as exc:
                errors.append(f'{where}: unreadable ({exc})')
                continue
            if r.version != VERSION or str(path) != str(path_for(r.season, path.parent)):
                errors.append(f'{where}: wrong version or season file')
            if not (lock <= cap < tip) or tip <= pd.Timestamp(REGISTERED):
                errors.append(f'{where}: not captured between lock and tip')
            if not (.5 <= p <= 1 and 0 < q < 1 and math.isfinite(ml) and abs(ml) >= 100):
                errors.append(f'{where}: invalid lock pick values')
            if (int(r.model_band) != calibration.band(p, calibration.MODEL_BOUNDS)
                    or int(r.market_band) != calibration.band(q, calibration.MARKET_BOUNDS)):
                errors.append(f'{where}: cell does not match the lock values')
            if not (0 <= w <= n <= calibration.WINDOW_GAMES and priced <= n
                    and int(r.previous_games) + int(r.current_games) == n):
                errors.append(f'{where}: inconsistent cell counts')
            roi = float(r.cell_roi) if r.cell_roi else float('nan')
            if (priced and not math.isclose(roi * priced, units, abs_tol=1e-9)) or (not priced and r.cell_roi):
                errors.append(f'{where}: ROI does not match units')
    return errors


def graded(frame, native):
    """Log rows with the game's 1u return on the logged pick at its lock price."""
    result = {str(r.game_id): float(r.home_won) for r in native.itertuples()
              if r.basis == 'native' and _number(r.home_won) in (0, 1)}
    rows = []
    for r in frame.itertuples():
        if r.game_id not in result:
            continue
        won = (result[r.game_id] == 1) == (r.pick_side == 'home')
        rows.append(dict(r._asdict(), won=int(won), units=market.unit_profit(float(r.pick_ml), won)))
    return pd.DataFrame(rows)


def _roi(units):
    units = np.asarray(units, float)
    se = (float('nan') if len(units) < 2 or (units > 0).all() or (units < 0).all()
          else units.std(ddof=1) / math.sqrt(len(units)))
    spread = f" ± {100 * se:.1f} pp (1 SE)" if np.isfinite(se) else ''
    return f"{100 * units.mean():+.1f}%{spread}, {units.sum():+.2f}u over {len(units)}"


def report(native, data_dir='data', out_dir=OUT_DIR):
    """diagnostics/cell_roi_forward/RESULTS.md; descriptive until the registered
    season-end reading."""
    frame = read_all(data_dir)
    g = graded(frame, native)
    lines = ['# Cell ROI forward log', '',
             f'Version {VERSION}; terms and the fixed season-end reading in '
             '`docs/cell_roi_forward_registration.md`. Monitoring only: never a rule or filter.', '',
             f'Logged locks: {len(frame)}; graded: {len(g)}; pending: {len(frame) - len(g)}.', '']
    labels_m = ['50–60%', '60–70%', '70–80%', '80–100%']
    labels_q = ['<50%', '50–60%', '60–70%', '70–80%', '80–100%']
    for name, part in (('Games 1+ (primary)', g[g.stage.isin(analysis.PUBLIC_STAGES)] if len(g) else g),
                       ('Opening games', g[g.stage.eq('opener')] if len(g) else g)):
        lines += [f'## {name}', '']
        if not len(part):
            lines += ['No graded games yet.', '']
            continue
        lines += [f'Forward 1u on the model pick: {_roi(part.units)}.', '',
                  '| Model WP | Market WP | Forward | Trailing cell ROI at lock (mean) |', '| --- | --- | --- | --- |']
        for (mb, qb), c in part.groupby([part.model_band.astype(int), part.market_band.astype(int)]):
            trail = pd.to_numeric(c.cell_roi, errors='coerce')
            lines.append(f'| {labels_m[mb]} | {labels_q[qb]} | {_roi(c.units)} | '
                         + (f'{100 * trail.mean():+.1f}%' if trail.notna().any() else '—') + ' |')
        x = pd.to_numeric(part.cell_roi, errors='coerce')
        ok = x.notna()
        if ok.sum() >= 3 and x[ok].std() > 0:
            X = np.column_stack([np.ones(ok.sum()), x[ok]])
            y = part.units[ok].to_numpy(float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            cov = resid @ resid / (len(y) - 2) * np.linalg.inv(X.T @ X)
            lines += ['', f'Return on trailing cell ROI (registered reading at season end): slope '
                      f'{beta[1]:+.3f} ± {1.96 * math.sqrt(cov[1, 1]):.3f} (95%), n = {len(y)}. Interim, descriptive.']
        lines.append('')
    rf.atomic_text(Path(out_dir) / 'RESULTS.md', '\n'.join(lines))
    return len(g)
