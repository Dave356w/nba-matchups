#!/usr/bin/env python3
"""Cache the sportsbooks' opening and closing game totals (O/U) from ESPN.

  python research/totals_fetch.py [--out diagnostics/totals_signal/totals_odds.csv]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sleep", type=float, default=0.35)
    a = ap.parse_args()
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
