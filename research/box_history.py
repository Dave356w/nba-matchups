#!/usr/bin/env python3
"""Build ESPN player box-score history for the roster-talent test.

  python research/box_history.py --seasons 2016 2017 2018 2019 2021 2022

research/team_quality.py's talent arm (minutes share x last-season BPM
over the previous box score) could be fitted only on 2022-23 onward, the
seasons whose box scores the injury-report work had cached. This fetches
the earlier seasons the base model trains on into the same cache
(research/output/box_<season>.csv, via player_availability.fetch_box) and
the previous season's BBR advanced page each needs (bbr_cache). Seasons
already cached are skipped, so a stopped run resumes. Research only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nba_composite as nc  # noqa: E402
import player_availability as pav  # noqa: E402


def build(seasons, cache_dir=pav.DEFAULT_CACHE, fetch=None, bpm=None):
    """Fetch (or reuse) each season's box scores and last season's BPM;
    return {season: summary line}."""
    fetch = fetch or pav.fetch_box
    bpm = bpm or pav.load_bpm
    out = {}
    for t in seasons:
        cached = os.path.exists(os.path.join(cache_dir, f"box_{t}.csv"))
        t0 = time.time()
        box = fetch(t, cache_dir=cache_dir)
        table = bpm(t - 1)
        _, _, cov_min = pav.player_values(box, table)
        games = box["game_id"].nunique() if len(box) else 0
        out[t] = (f"season {t}: {games} games, {len(box)} player rows, "
                  f"last-season BPM covers {100 * cov_min:.0f}% of minutes "
                  f"({'cached' if cached else f'fetched in {time.time() - t0:.0f}s'})")
        print(out[t], flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=["2016-2019", "2021-2022"])
    ap.add_argument("--cache", default=pav.DEFAULT_CACHE)
    a = ap.parse_args(argv)
    build(nc.parse_years(a.seasons), a.cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
