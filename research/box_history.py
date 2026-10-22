#!/usr/bin/env python3
"""Build ESPN player box-score history (and commit the finished seasons).

  python research/box_history.py --seasons 2016-2019 2021-2026 [--commit]

The model's roster talent (games 1-9 and 10+), the availability fit and the
research arms read every box-score season the base model trains on. This
fetches each season into the research cache (research/output/box_<season>.csv,
via player_availability.fetch_box) with the previous season's BBR advanced
page each needs (bbr_cache). Seasons already on disk are skipped, so a stopped
run resumes. --commit also writes each finished season (its last possible
regular-season date before today, ET) to data/nba_box_<season>.csv, and its
preseason box scores (model v8's game-0 rosters) to data/nba_box_pre_<season>
.csv; player_availability.box_file / pre_box_file read them before the
cache: the "Box-score history"
workflow commits them, so a fit or backfill on a branch whose Actions cache
has expired does not refetch ~10 seasons from ESPN. The current season is
never written here (the daily build extends its file).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402


def finished(t, today):
    """True once season t's last possible regular-season date is before today."""
    return pav.season_dates(t)[-1] < pd.Timestamp(today)


def build(seasons, cache_dir=pav.DEFAULT_CACHE, fetch=None, bpm=None,
          data_dir=None, commit=False, today=None, fetch_pre=None):
    """Fetch (or reuse) each season's box scores and last season's BPM;
    with commit, write each finished season to data_dir/nba_box_<t>.csv and
    its preseason box scores (model v8's game-0 rosters; opening night =
    the first regular-season box) to data_dir/nba_box_pre_<t>.csv.
    Return {season: summary line}."""
    fetch = fetch or pav.fetch_box
    bpm = bpm or pav.load_bpm
    fetch_pre = fetch_pre or pav.fetch_preseason_box
    data_dir = data_dir or pav.DATA_DIR
    today = today or datetime.now(ZoneInfo("America/New_York")).date()
    out = {}
    for t in seasons:
        cached = pav.box_file(t, cache_dir, data_dir) is not None
        t0 = time.time()
        box = fetch(t, cache_dir=cache_dir)
        table = bpm(t - 1)
        _, _, cov_min = pav.player_values(box, table)
        games = box["game_id"].nunique() if len(box) else 0
        out[t] = (f"season {t}: {games} games, {len(box)} player rows, "
                  f"last-season BPM covers {100 * cov_min:.0f}% of minutes "
                  f"({'on disk' if cached else f'fetched in {time.time() - t0:.0f}s'})")
        if commit and len(box):
            if finished(t, today):
                pav.save_box(box, os.path.join(data_dir, f"nba_box_{t}.csv"))
                out[t] += f"; written to data/nba_box_{t}.csv"
                pre = fetch_pre(t, pd.Timestamp(box["date"].min()), cache_dir=cache_dir,
                                data_dir=data_dir)
                if len(pre):
                    pav.save_pre_box(pre, os.path.join(data_dir, f"nba_box_pre_{t}.csv"))
                    out[t] += (f"; preseason {pre['game_id'].nunique()} games, "
                               f"{pre['team'].nunique()} teams")
            else:
                out[t] += "; not finished, not written"
        print(out[t], flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=["2016-2019", "2021-2026"])
    ap.add_argument("--cache", default=pav.DEFAULT_CACHE)
    ap.add_argument("--commit", action="store_true",
                    help="write finished seasons to data/nba_box_<season>.csv")
    a = ap.parse_args(argv)
    build(nc.parse_years(a.seasons), a.cache, commit=a.commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
