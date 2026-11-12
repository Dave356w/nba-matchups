"""Display-only rolling joint calibration of v11 ML forecasts.

Use the latest 50 actual outcomes in each model-winner × market-WP cell:
previous-season replay seeds plus current-season native five-hour locks.
Early and 10+ share evidence; opening games remain separate. No market
pseudo-games, model changes, price changes, or writes to saved results.

The estimate is band-adjusted (v2, docs/realized_wp_forward_registration_v2.md):
this game's no-vig market WP plus the cell's win rate minus the cell's mean
no-vig market WP, so a game is not credited or penalised for where it sits
inside a wide market band. The raw cell rate stays available as `cell_rate`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import analysis
import market

WINDOW_GAMES = 50
MODEL_BOUNDS = (.5, .6, .7, .8, 1.0)
MARKET_BOUNDS = (0., .5, .6, .7, .8, 1.0)


def band(value, bounds):
    """Lower-inclusive cells; the final cell includes probability 1."""
    return min(int(np.searchsorted(bounds, value, side='right') - 1), len(bounds) - 2)


def band_adjusted(q_game, wins, games, member_market):
    """q_game + (wins/games − mean member market WP), clipped to [0, 1]."""
    if not games:
        return float('nan')
    # Rate first: identical prices leave the raw cell rate exactly unchanged.
    return min(1., max(0., wins/games + (q_game - math.fsum(member_market)/games)))


def stage(row):
    gp = np.minimum(pd.to_numeric(row.get('gp_home'), errors='coerce'),
             pd.to_numeric(row.get('gp_away'), errors='coerce'))
    if not np.isfinite(gp):
        return None
    return 'established' if gp >= 10 else 'early' if gp >= 1 else 'opener' if gp == 0 else None


class RollingCalibration:
    def __init__(self, native, replay=None):
        # Reuse the registered exclusions, family, timing and stage boundaries.
        predictions = analysis.moneyline_model_predictions(native, replay)
        self.rows = []
        for _, row in predictions.iterrows():
            y = pd.to_numeric(row.get('home_won'), errors='coerce')
            if y not in (0, 1):
                continue
            source = row.prediction_source
            prefix = 'lock' if source == 'live' else 'close'
            if row.get(f'{prefix}_book') != 'dk':
                continue
            q = market.devig(row.get(f'{prefix}_home_ml'), row.get(f'{prefix}_away_ml'))
            if not np.isfinite(q):
                continue
            p_home = pd.to_numeric(row.lock_p_home if source == 'live' else row.p_home, errors='coerce')
            home = p_home >= .5
            p_pick = p_home if home else 1 - p_home
            q_pick = q if home else 1 - q
            self.rows.append(dict(game_id=str(row.game_id), season=row.season,
                date=str(row.slate_date), stage=row.prediction_stage, source=source,
                model_band=band(p_pick, MODEL_BOUNDS), market_band=band(q_pick, MARKET_BOUNDS),
                won=int(y if home else 1-y), model_probability=float(p_pick),
                market_probability=float(q_pick),
                price=float(row.get(f'{prefix}_home_ml' if home else f'{prefix}_away_ml'))))

    def estimate(self, row):
        """Estimate the CURRENT model winner using current pregame DK prices.

        Date selection reconstructs the pre-slate view even for final cards:
        the target game's own outcome and every same/later slate are excluded.
        """
        p = pd.to_numeric(row.get('p_home'), errors='coerce')
        experience = stage(row)
        if (not np.isfinite(p) or not 0 <= p <= 1 or experience is None
                or not str(row.get('model_tag', '')).endswith(analysis.FAMILY_TAGS)
                or row.get('pre_book') != 'dk'):
            return None
        q_home = market.devig(row.get('pre_home_ml'), row.get('pre_away_ml'))
        if not np.isfinite(q_home):
            return None
        home = p >= .5
        p_pick, q_pick = (p, q_home) if home else (1-p, 1-q_home)
        pb, qb = band(p_pick, MODEL_BOUNDS), band(q_pick, MARKET_BOUNDS)
        season = pd.to_numeric(row.get('season'), errors='coerce')
        if not np.isfinite(season):
            return None
        prior = [r for r in self.rows
                 if ((r['season'] == season and r['source'] == 'live') or r['season'] == season-1)
                 and (r['stage'] in ('early', 'established') if experience in ('early', 'established') else r['stage'] == experience)
                 and r['date'] < str(row.get('slate_date'))
                 and r['game_id'] != str(row.get('game_id'))
                 and r['model_band'] == pb and r['market_band'] == qb]
        # Date and ID provide deterministic order, independent of outcomes.
        prior = sorted(prior, key=lambda r: (r['date'], r['game_id']))[-WINDOW_GAMES:]
        wins, n = sum(r['won'] for r in prior), len(prior)
        previous = sum(r['season'] == season-1 for r in prior)
        member_market = [r['market_probability'] for r in prior]
        return dict(probability=band_adjusted(q_pick, wins, n, member_market),
                    cell_rate=wins/n if n else np.nan,
                    cell_market=math.fsum(member_market)/n if n else np.nan,
                    market_probability=q_pick, games=n, wins=wins, losses=n-wins,
                    previous_games=previous, current_games=n-previous,
                    model_range=MODEL_BOUNDS[pb:pb+2], market_range=MARKET_BOUNDS[qb:qb+2],
                    evidence=prior)
