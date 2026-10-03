#!/usr/bin/env python3
"""Probe Kalshi's public market-data API for NBA game markets (prints only).

  python research/kalshi_probe.py

Lists sports series whose ticker or title mentions NBA, the open events of
each with their markets and quotes, and one market's 1-minute candlesticks,
so the pricing code can be written against the real payloads. Writes nothing.
"""
import json
import time
import urllib.parse
import urllib.request

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def get(path, **params):
    url = f"{BASE}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": "nba-matchups/1.0",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    """KXNBAGAME open events (one request), then one market's 1-minute
    candlesticks (the close kalshi.py reads). Few requests: Kalshi 429s."""
    js = get("/events", series_ticker="KXNBAGAME", status="open",
             with_nested_markets="true", limit=200)
    evs = js.get("events", [])
    print(f"KXNBAGAME: {len(evs)} open events")
    for ev in evs:
        ms = ev.get("markets", [])
        print(ev.get("event_ticker"), "|", ev.get("title"), "|",
              [(m.get("ticker"), m.get("yes_bid_dollars"), m.get("yes_ask_dollars"))
               for m in ms])
    if not evs:
        return
    mt = evs[0]["markets"][0]["ticker"]
    now = int(time.time())
    time.sleep(2)
    js = get(f"/series/KXNBAGAME/markets/{mt}/candlesticks",
             start_ts=now - 2 * 3600, end_ts=now, period_interval=1)
    cs = js.get("candlesticks", [])
    print(f"\ncandlesticks {mt}: {len(cs)} keys {sorted(js)}")
    for c in cs[-3:]:
        print(json.dumps(c))


if __name__ == "__main__":
    main()
