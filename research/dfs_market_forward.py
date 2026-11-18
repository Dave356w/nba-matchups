"""Compare archived pregame NBA market estimates with Sleeper and a 50/50 blend.

Run from the repository root: python research/dfs_market_forward.py
MAE uses the same played rows for all three forecasts. Per-stat rows require
a market estimate for that stat. The weighted subtotal and full Yahoo score
require every component of their respective scoring formulas.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dfs
import dfs_market
import ledger


def evaluate(native, sleeper, market_rows, fetch=dfs.final_summary):
    priced = {r["game_id"] for r in market_rows}
    games = {str(r.game_id): ledger.parse_utc(r.tip_utc) for r in native.itertuples()
             if str(r.game_id) in priced and r.home_won in (0, 1)}
    props = dfs_market.index(market_rows, games, dfs.norm_name, ledger.utc_now())
    forecasts = {}
    for r in sorted(sleeper.itertuples(), key=lambda r: (r.kind == "pre", r.snapshot_utc)):
        gid = str(r.game_id)
        snapshot = ledger.parse_utc(r.snapshot_utc)
        if gid in games and games[gid] and snapshot and snapshot < games[gid]:
            forecasts[gid, r.team, dfs.norm_name(r.name)] = r

    def final(gid):
        try:
            return gid, dfs.box_stats(fetch(gid))
        except (OSError, ValueError) as exc:
            print(f"DFS market final box {gid} unavailable ({exc})", file=sys.stderr)
            return gid, []

    errors = defaultdict(list)
    groups = {k: (k,) for k in dfs.SCORING}
    groups.update(weighted_pra=("pts", "reb", "ast"), yahoo=tuple(dfs.SCORING))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for gid, rows in pool.map(final, sorted(games)):
            for actual in rows:
                key = gid, actual["team"], actual["key"]
                line = forecasts.get(key)
                market = props.get(key, {})
                if (line is None or not market or (dfs._num(line.minutes) or 0) <= 0
                        or str(line.injury_status).lower() in dfs.OUT_STATUSES):
                    continue
                for label, components in groups.items():
                    if any(k not in market for k in components):
                        continue
                    weights = {k: 1 if label in dfs.SCORING else dfs.SCORING[k] for k in components}
                    sleeper_stats = {k: dfs._num(getattr(line, k)) for k in components}
                    if None in sleeper_stats.values():
                        continue
                    sl = sum(sleeper_stats[k] * weights[k] for k in components)
                    mk = sum(market[k]["mean"] * weights[k] for k in components)
                    result = sum(actual[k] * weights[k] for k in components)
                    errors[label].append((abs(mk - result), abs(sl - result),
                                          abs((mk + sl) / 2 - result)))
    return {label: (len(rows), *(mean(values) for values in zip(*rows)))
            for label, rows in errors.items()}


if __name__ == "__main__":
    results = evaluate(ledger.load(ledger.NATIVE_PATH), ledger.load_sleeper(), dfs_market.load())
    print(f"{'stat':<14} {'N':>6} {'Market MAE':>12} {'Sleeper MAE':>12} {'Blend MAE':>12}")
    for stat in (*dfs.SCORING, "weighted_pra", "yahoo"):
        n, market, sleeper, blend = results.get(stat, (0, math.nan, math.nan, math.nan))
        print(f"{stat:<14} {n:>6} {market:>12.2f} {sleeper:>12.2f} {blend:>12.2f}")
