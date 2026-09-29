#!/usr/bin/env python3
"""Which sportsbooks does ESPN's odds endpoint list, and when?

  python research/odds_providers.py [--dates 2024-10-25 2025-12-10 ...] [--per-date 2]

`market.parse_dk_odds` keeps only provider id DK_PROVIDER_ID (DraftKings) and
returns None when it is absent. The reconstructed rows have no DK close for
2024-25 and none before late November 2025, which suggests ESPN listed a
different book (e.g. ESPN BET) before then. This dumps, for a few games per
sample date, every provider in the odds response with its moneyline
open/current/close and closing spread (line and prices), so we can see
what exists before changing the parser.

Raw responses are saved to research/output/odds_providers/ for inspection.
Research only: nothing here changes the model, the ledger or MODEL_TAG.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market  # noqa: E402

# Regular-season dates spanning 2024-25 and the 2025-26 changeover window.
DEFAULT_DATES = ["2024-10-25", "2024-12-15", "2025-02-10", "2025-04-10",
                 "2025-10-25", "2025-11-15", "2025-11-28", "2025-12-10"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "output", "odds_providers")


def _ml(team_odds, stage):
    """American moneyline for one side at 'open' / 'current' / 'close'."""
    return market.american(market._dig(team_odds or {}, stage, "moneyLine"))


def summarize_providers(js):
    """Odds JSON -> one dict per listed provider (inline items only)."""
    rows = []
    for it in js.get("items", []):
        prov = it.get("provider") or {}
        h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
        rows.append(dict(
            id=str(prov.get("id")), name=prov.get("name"),
            priority=prov.get("priority"),
            ref_only=set(it) <= {"$ref"},
            home_open=_ml(h, "open"), home_cur=_ml(h, "current"),
            home_close=_ml(h, "close"),
            away_open=_ml(a, "open"), away_cur=_ml(a, "current"),
            away_close=_ml(a, "close"),
            home_ml=market.american(h.get("moneyLine")),
            away_ml=market.american(a.get("moneyLine")),
            home_close_line=market.spread_line(market._dig(h, "close", "pointSpread")),
            home_close_spread=market.american(market._dig(h, "close", "spread")),
            away_close_line=market.spread_line(market._dig(a, "close", "pointSpread")),
            away_close_spread=market.american(market._dig(a, "close", "spread")),
            top_spread=it.get("spread"),
        ))
    return rows


def _resolve_refs(js):
    """Some core responses list items as bare {'$ref': url}; fetch those."""
    items = []
    for it in js.get("items", []):
        if set(it) <= {"$ref"} and it.get("$ref"):
            try:
                items.append(market.get_json(it["$ref"]))
            except Exception as e:  # noqa: BLE001
                items.append({"provider": {"id": "?", "name": f"ref failed {e!r}"}})
        else:
            items.append(it)
    return {**js, "items": items}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--per-date", type=int, default=2)
    args = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)

    seen = {}
    for date in args.dates:
        try:
            games = market.scoreboard(date)
        except Exception as e:  # noqa: BLE001
            print(f"{date}: scoreboard failed {e!r}")
            continue
        print(f"\n=== {date}: {len(games)} games on scoreboard")
        for g in games[:args.per_date]:
            try:
                js = market.get_json(market.ODDS.format(eid=g["game_id"]))
            except Exception as e:  # noqa: BLE001
                print(f"  {g['away']}@{g['home']} ({g['game_id']}): odds failed {e!r}")
                continue
            with open(os.path.join(OUT_DIR, f"{date}_{g['game_id']}.json"), "w") as f:
                json.dump(js, f, indent=1)
            js = _resolve_refs(js)
            rows = summarize_providers(js)
            dk = market.parse_dk_odds(js)
            print(f"  {g['away']}@{g['home']} ({g['game_id']}): "
                  f"{len(rows)} provider(s); parse_dk_odds -> "
                  f"{'DK found' if dk else 'None'}")
            for r in rows:
                seen.setdefault((r["id"], r["name"]), []).append(date)
                print(f"    id={r['id']:>5} {str(r['name'])[:22]:<22} "
                      f"prio={r['priority']}  "
                      f"home open/cur/close {r['home_open']}/{r['home_cur']}/{r['home_close']}  "
                      f"away {r['away_open']}/{r['away_cur']}/{r['away_close']}  "
                      f"(top-level {r['home_ml']}/{r['away_ml']})  "
                      f"close spread home {r['home_close_line']} @ {r['home_close_spread']} "
                      f"away {r['away_close_line']} @ {r['away_close_spread']} "
                      f"(top-level spread {r['top_spread']})")
            time.sleep(0.3)

    print("\n=== providers seen (id, name): dates")
    for (pid, name), dates in sorted(seen.items(), key=lambda kv: kv[0][0]):
        print(f"  {pid:>5} {name}: {sorted(set(dates))}")


if __name__ == "__main__":
    main()
