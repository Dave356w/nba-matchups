"""Display-only, prior-slate team records and season four-factor comparisons.

Read saved inputs only. Missing/stale inputs never trigger a fetch or change a
forecast, lock, selection or grade.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import nba_composite as nc


class MatchupContext:
    def __init__(self, logs=None, boxes=None):
        # Season-qualified keys prevent another season's games leaking in.
        self.logs = logs or {}
        self.boxes = boxes or {}

    @classmethod
    def saved(cls, day):
        logs, boxes = {}, {}
        for _, row in day.iterrows():
            year = int(row.season)
            if year not in boxes:
                path = Path('data') / f'nba_box_{year}.csv'
                try:
                    boxes[year] = pd.read_csv(path, dtype={'game_id': str}) if path.exists() else pd.DataFrame()
                except (OSError, ValueError):
                    boxes[year] = pd.DataFrame()
            for team in (row.away, row.home):
                key = (year, team)
                if key in logs:
                    continue
                logs[key] = pd.DataFrame()
                # read_gamelog uses this file without network when it exists.
                if (nc.CACHE / f'gl_{team}_{year}.html').is_file():
                    try:
                        log = nc.read_gamelog(team, year)
                        if log is not None:
                            logs[key] = log
                    except (OSError, ValueError, KeyError, IndexError):
                        pass
        return cls(logs, boxes)

    def team(self, row, side):
        """A complete current-season sample strictly before this slate.

        Compare its length with the forecast's games-played count; a partial
        cache must not look like a team's full-season record or averages.
        Player boxes provide a deduplicated record fallback, never factors.
        """
        year, team = int(row.season), row[side]
        expected = pd.to_numeric(row.get(f'gp_{side}'), errors='coerce')
        out = dict(record=None, factors=None, games=None)
        if expected == 0:
            return dict(record=(0, 0), factors=None, games=0)
        log = self.logs.get((year, team), pd.DataFrame())
        if len(log):
            prior = log[pd.to_datetime(log.date, errors='coerce') < pd.Timestamp(row.slate_date)].copy()
            pts = pd.to_numeric(prior.pts, errors='coerce')
            opp = pd.to_numeric(prior.opp_pts, errors='coerce')
            prior = prior[np.isfinite(pts) & np.isfinite(opp) & pts.ne(opp)]
            if len(prior) and (not np.isfinite(expected) or len(prior) == expected):
                won = pd.to_numeric(prior.pts) > pd.to_numeric(prior.opp_pts)
                out.update(record=(int(won.sum()), int((~won).sum())), games=len(prior))
                totals = prior.reindex(columns=nc.COLS).apply(pd.to_numeric, errors='coerce')
                if np.isfinite(totals.to_numpy()).all():
                    with np.errstate(divide='ignore', invalid='ignore'):
                        factors = nc.features_from_totals(totals.sum().to_numpy(), orb_pct=True)
                    if np.isfinite(factors).all():
                        # Restore natural signs for the public comparison.
                        out['factors'] = factors * np.array([1, -1, 1, 1, -1, 1, -1, -1])
        if out['record'] is None:
            boxes = self.boxes.get(year, pd.DataFrame())
            if len(boxes):
                prior = boxes[boxes.team.eq(team) &
                              (pd.to_datetime(boxes.date, errors='coerce') < pd.Timestamp(row.slate_date))]
                games = prior[['game_id', 'margin']].drop_duplicates()
                # Conflicting player rows cannot create two results for a game.
                if not games.game_id.duplicated().any():
                    margin = pd.to_numeric(games.margin, errors='coerce')
                    margin = margin[np.isfinite(margin) & margin.ne(0)]
                    if len(margin) and (not np.isfinite(expected) or len(margin) == expected):
                        out.update(record=(int((margin > 0).sum()), int((margin < 0).sum())), games=len(margin))
        return out
