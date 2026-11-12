"""Display-only, in-season joint calibration of v11 ML forecasts.

The prior is 100 copies of the current no-vig market probability. Evidence
comes only from valid, graded native five-hour locks in the same season,
model-winner × market-WP cell, on earlier slates. Early and 10+ games
share evidence; opening games remain separate.
No model, market price, saved pick or result is written or changed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import analysis
import market

PRIOR_GAMES = 100
MODEL_BOUNDS = (.5, .6, .7, .8, 1.0)
MARKET_BOUNDS = (0., .5, .6, .7, .8, 1.0)


def band(value, bounds):
    """Lower-inclusive cells; the final cell includes probability 1."""
    return min(int(np.searchsorted(bounds, value, side='right') - 1), len(bounds) - 2)


def stage(row):
    gp = np.minimum(pd.to_numeric(row.get('gp_home'), errors='coerce'),
             pd.to_numeric(row.get('gp_away'), errors='coerce'))
    if not np.isfinite(gp):
        return None
    return 'established' if gp >= 10 else 'early' if gp >= 1 else 'opener' if gp == 0 else None


class SeasonCalibration:
    def __init__(self, native):
        # Reuse the registered exclusions, family, timing and stage boundaries.
        live = analysis.model_predictions(native)
        self.rows = []
        for _, row in live.iterrows():
            y = pd.to_numeric(row.get('home_won'), errors='coerce')
            if y not in (0, 1) or row.get('lock_book') != 'dk':
                continue
            q = market.devig(row.get('lock_home_ml'), row.get('lock_away_ml'))
            if not np.isfinite(q):
                continue
            p_home = pd.to_numeric(row.lock_p_home, errors='coerce')
            home = p_home >= .5
            p_pick = p_home if home else 1 - p_home
            q_pick = q if home else 1 - q
            self.rows.append(dict(game_id=str(row.game_id), season=row.season,
                date=str(row.slate_date), stage=row.prediction_stage,
                model_band=band(p_pick, MODEL_BOUNDS), market_band=band(q_pick, MARKET_BOUNDS),
                won=int(y if home else 1-y)))

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
        prior = [r for r in self.rows if r['season'] == row.get('season')
                 and (r['stage'] in ('early', 'established') if experience in ('early', 'established') else r['stage'] == experience)
                 and r['date'] < str(row.get('slate_date'))
                 and r['game_id'] != str(row.get('game_id'))
                 and r['model_band'] == pb and r['market_band'] == qb]
        wins, n = sum(r['won'] for r in prior), len(prior)
        return dict(probability=(wins + PRIOR_GAMES*q_pick)/(n + PRIOR_GAMES),
                    market_probability=q_pick, games=n, wins=wins, losses=n-wins,
                    model_range=MODEL_BOUNDS[pb:pb+2], market_range=MARKET_BOUNDS[qb:qb+2])
