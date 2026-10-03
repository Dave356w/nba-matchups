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
    series = []
    try:
        js = get("/series", category="Sports")
        for s in js.get("series", []):
            if "NBA" in (s.get("ticker", "") + s.get("title", "")).upper():
                series.append(s)
                print("series", s.get("ticker"), "|", s.get("title"))
    except Exception as e:  # noqa: BLE001
        print("series list failed", repr(e))
    tickers = [s["ticker"] for s in series] or ["KXNBAGAME"]
    if "KXNBAGAME" not in tickers:
        tickers.append("KXNBAGAME")
    sample = None
    for t in tickers:
        try:
            js = get("/events", series_ticker=t, status="open",
                     with_nested_markets="true", limit=20)
        except Exception as e:  # noqa: BLE001
            print(t, "events failed", repr(e))
            continue
        evs = js.get("events", [])
        print(f"\n=== {t}: {len(evs)} open events")
        for ev in evs[:12]:
            print("event", json.dumps({k: ev.get(k) for k in
                  ("event_ticker", "title", "sub_title", "strike_date",
                   "category")}))
            for m in ev.get("markets", [])[:3]:
                keep = {k: v for k, v in m.items() if not isinstance(v, (list, dict))
                        and k not in ("rules_primary", "rules_secondary")}
                print("  market", json.dumps(keep))
                if sample is None and t == "KXNBAGAME":
                    sample = (t, m["ticker"])
    if sample:
        t, mt = sample
        now = int(time.time())
        try:
            js = get(f"/series/{t}/markets/{mt}/candlesticks",
                     start_ts=now - 3 * 3600, end_ts=now, period_interval=1)
            cs = js.get("candlesticks", [])
            print(f"\ncandlesticks {mt}: {len(cs)}")
            for c in cs[-3:]:
                print(json.dumps(c))
        except Exception as e:  # noqa: BLE001
            print("candlesticks failed", repr(e))
    # a settled game too, to see the shape after the market closes
    try:
        js = get("/events", series_ticker="KXNBAGAME", status="settled",
                 with_nested_markets="true", limit=3)
        for ev in js.get("events", [])[:2]:
            print("\nsettled", ev.get("event_ticker"), "|", ev.get("title"))
            for m in ev.get("markets", [])[:2]:
                print("  ", json.dumps({k: m.get(k) for k in
                      ("ticker", "yes_sub_title", "result", "open_time",
                       "close_time", "expected_expiration_time",
                       "last_price", "status")}))
    except Exception as e:  # noqa: BLE001
        print("settled failed", repr(e))


if __name__ == "__main__":
    main()
