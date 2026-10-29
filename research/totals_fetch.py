#!/usr/bin/env python3
"""Cache the sportsbooks' opening and closing game totals (O/U) from ESPN.

  python research/totals_fetch.py [--out diagnostics/totals_signal/totals_odds.csv]
  python research/totals_fetch.py --season 2024     # a whole season, by scoreboard

For every reconstructed and native game id, the ESPN core odds feed's
DraftKings and ESPN BET items: open / close total and both over / under
prices. Research input only (research/totals_signal.py); the model, the
ledger and the pages never read it. Resumable: ids already cached are
skipped. Requests are throttled.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ledger  # noqa: E402
import market  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "diagnostics" / "totals_signal" / "totals_odds.csv"
COLS = ["game_id", "basis", "book", "open_total", "open_over", "open_under",
        "close_total", "close_over", "close_under"]


def _num(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None


def parse_totals(js):
    """ESPN core odds JSON -> [dict(book, open_*/close_* total and prices)]."""
    books = dict(market.BOOKS)
    out = []
    for item in js.get("items", []):
        book = books.get(str(item.get("provider", {}).get("id")))
        if book is None:
            continue
        row = dict(book=book)
        for when in ("open", "close"):
            d = item.get(when) or {}
            for k in ("total", "over", "under"):
                row[f"{when}_{k}"] = _num((d.get(k) or {}).get("american"))
        out.append(row)
    return out


HISTORY = OUT.parent / "totals_history_{season}.csv"


def fetch_season(season, out, threads=4):
    """A season without reconstructed rows (e.g. 2023-24 = 2024): game ids,
    teams and finals from the scoreboard (regular season only), then each
    game's DraftKings / ESPN BET open and close totals. Resumable."""
    from concurrent.futures import ThreadPoolExecutor
    have = pd.read_csv(out, dtype={"game_id": str}) if out.exists() else None
    done = set(have["game_id"]) if have is not None else set()
    start, end = pd.Timestamp(f"{season - 1}-10-15"), pd.Timestamp(f"{season}-04-20")
    games = []
    for d in pd.date_range(start, end):
        try:
            for g in market.scoreboard(d.strftime("%Y-%m-%d")):
                if g["season_type"] == 2 and g["completed"]:
                    games.append(dict(g, slate_date=d.strftime("%Y-%m-%d")))
        except Exception as e:  # noqa: BLE001 - a missing day is skipped
            print(f"{d:%Y-%m-%d}: {e!r}", flush=True)
    todo = [g for g in games if g["game_id"] not in done]
    print(f"season {season}: {len(games)} games, {len(todo)} to fetch", flush=True)

    def one(g):
        try:
            found = parse_totals(market.get_json(market.ODDS.format(eid=g["game_id"])))
        except Exception as e:  # noqa: BLE001
            print(f"{g['game_id']}: {e!r}", flush=True)
            found = []
        base = dict(game_id=g["game_id"], season=season, slate_date=g["slate_date"],
                    home=g["home"], away=g["away"], home_pts=g["home_pts"], away_pts=g["away_pts"])
        return [dict(base, **r) for r in found] or [dict(base, book=None)]
    rows = []
    with ThreadPoolExecutor(threads) as ex:
        for n, r in enumerate(ex.map(one, todo), 1):
            rows += r
            if n % 200 == 0 or n == len(todo):
                frame = pd.DataFrame(rows)
                have = frame if have is None else pd.concat([have, frame], ignore_index=True)
                have.to_csv(out, index=False)
                rows = []
                print(f"{n}/{len(todo)} games", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--season", type=int, help="fetch a whole season from the scoreboard")
    a = ap.parse_args()
    if a.season:
        fetch_season(a.season, Path(str(HISTORY).format(season=a.season)) if a.out == str(OUT)
                     else Path(a.out))
        return
    out = Path(a.out)
    have = pd.read_csv(out, dtype={"game_id": str}) if out.exists() else pd.DataFrame(columns=COLS)
    done = set(have["game_id"].astype(str))
    ids = []
    for basis, path in (("reconstructed", ledger.RECON_PATH), ("native", ledger.NATIVE_PATH)):
        g = ledger.load(path)
        ids += [(str(i), basis) for i in g["game_id"].astype(str) if str(i) not in done]
    rows = []
    for n, (gid, basis) in enumerate(ids, 1):
        try:
            found = parse_totals(market.get_json(market.ODDS.format(eid=gid)))
        except Exception as e:  # noqa: BLE001 - one game's feed failing skips it
            print(f"{gid}: {e!r}", flush=True)
            found = []
        rows += [dict(game_id=gid, basis=basis, **r) for r in found] or \
            [dict(game_id=gid, basis=basis, book=None)]
        if n % 100 == 0 or n == len(ids):
            have = pd.concat([have, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
            have.to_csv(out, index=False)
            rows = []
            print(f"{n}/{len(ids)} games", flush=True)
        time.sleep(a.sleep)


if __name__ == "__main__":
    main()
