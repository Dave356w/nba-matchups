"""Kalshi exchange prices for preseason games (data/nba_preseason.csv only).

Kalshi lists every NBA game, preseason included, in the KXNBAGAME series:
one event per game (ticker KXNBAGAME-26OCT03MIATOR = date, away, home) with
one binary market per team (ticker suffix = that team's code; YES pays $1
if it wins). research/kalshi_probe.py printed the payloads this reads.

A side's price is the cost of buying its YES at the ask, Kalshi's taker fee
included (market.kalshi_cost), written as an American price, so the ledger's
break-even, ROI and null arithmetic is unchanged. q (the market's P) is the
bid/ask midpoints normalised (mid_q), since preseason books can be wide;
without a bid, the two costs normalised. Book label "kalshi".

  pregame  : the live asks, read by the hourly build while before tip
             (one request per build for every open game).
  close    : the last 1-minute candle ending at or before the ESPN tip time,
             read once when the game is graded. Never a price after tip.

Kalshi rate-limits (HTTP 429), so requests are few, spaced and retried.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import market

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNBAGAME"
BOOK = "kalshi"
# Kalshi team code -> Basketball-Reference code, where they differ. The short
# aliases (GS, NO, NY, SA, WSH) are accepted when parsing but Kalshi's tickers
# use GSW, NOP, NYK, SAS and WAS (checked against the live API, 2026-10-06).
KALSHI2BBR = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO", "GS": "GSW",
              "NO": "NOP", "NY": "NYK", "SA": "SAS", "WSH": "WAS"}
# Basketball-Reference code -> the code in Kalshi's tickers (market_ticker).
# Not the inverse of KALSHI2BBR: inverting it built GS / NY tickers, which do
# not exist, so Warriors, Pelicans, Knicks, Spurs and Wizards games got no close.
BBR2KALSHI = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}
MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
CLOSE_WINDOW = 90 * 60          # seconds of 1-minute candles read before tip
SPACING = 0.6                   # seconds between requests


def get(path, tries=4, **params):
    url = f"{BASE}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=market.HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                out = json.loads(r.read())
            time.sleep(SPACING)
            return out
        except urllib.error.HTTPError as e:
            last = e
            if e.code != 429 and e.code < 500:
                raise
        except Exception as e:  # noqa: BLE001 - network: retry then raise
            last = e
        time.sleep(2.0 * 2 ** k)
    raise last


def team(code):
    code = str(code).upper()
    bbr = KALSHI2BBR.get(code, code)
    return bbr if bbr in market.BBR_TEAMS else None


def date_code(date):
    """'2026-10-03' -> '26OCT03', the date in Kalshi's event tickers."""
    y, m, d = str(date)[:10].split("-")
    return f"{y[2:]}{MONTHS[int(m) - 1]}{d}"


def _dollars(m, key):
    v = m.get(f"{key}_dollars")
    if v is None and m.get(key) is not None:
        v = float(m[key]) / 100.0            # older payloads: cents
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < 1 else None


def parse_events(js):
    """/events payload -> {(date code, frozenset{bbr, bbr}): {bbr: market}}."""
    out = {}
    for ev in js.get("events", []):
        et = str(ev.get("event_ticker", ""))
        if not et.startswith(SERIES + "-"):
            continue
        dcode = et.split("-")[1][:7]
        sides = {}
        for m in ev.get("markets", []):
            t = team(str(m.get("ticker", "")).rsplit("-", 1)[-1])
            if t:
                sides[t] = m
        if len(sides) == 2:
            out[(dcode, frozenset(sides))] = sides
    return out


def open_games():
    """Every open KXNBAGAME event, keyed as parse_events. One request per
    page (200 events); preseason and regular season alike."""
    out, cursor = {}, None
    for _ in range(5):
        params = dict(series_ticker=SERIES, status="open",
                      with_nested_markets="true", limit=200)
        if cursor:
            params["cursor"] = cursor
        js = get("/events", **params)
        out.update(parse_events(js))
        cursor = js.get("cursor")
        if not cursor:
            break
    return out


def mid_q(home, away):
    """Market P(home) from each side's (bid, ask) midpoint, normalised; NaN
    unless both sides have a bid and an ask. Preseason books can be wide
    (18c / 76c), where normalised asks say little about the market's view."""
    try:
        mh, ma = (sum(home) / 2, sum(away) / 2)
    except TypeError:                        # a None bid or ask
        return float("nan")
    return mh / (mh + ma) if mh > 0 and ma > 0 else float("nan")


def odds_from_quotes(home, away, kind):
    """(bid, ask) per side -> {f'{kind}_home_ml', f'{kind}_away_ml'} (the YES
    asks with the taker fee: the bettable prices) and f'{kind}_q_home'
    (mid_q; when a bid is missing, the two costs normalised). {} when either
    side has no ask."""
    h = market.american_from_prob(market.kalshi_cost(home[1]))
    a = market.american_from_prob(market.kalshi_cost(away[1]))
    if h is None or a is None:
        return {}
    q = mid_q(home, away)
    if not q == q:                           # NaN
        q = market.devig(h, a)
    return {f"{kind}_home_ml": h, f"{kind}_away_ml": a,
            f"{kind}_q_home": round(q, 5)}


def pregame_odds(games, date, home, away):
    """The live pair for one game from open_games(): {'book', 'cur_home_ml',
    'cur_away_ml', 'cur_q_home', 'tickers'} or {} when Kalshi has no
    two-sided ask."""
    sides = games.get((date_code(date), frozenset((home, away))))
    if not sides:
        return {}
    o = odds_from_quotes(
        (_dollars(sides[home], "yes_bid"), _dollars(sides[home], "yes_ask")),
        (_dollars(sides[away], "yes_bid"), _dollars(sides[away], "yes_ask")), "cur")
    if not o:
        return {}
    return {"book": BOOK, **o,
            "tickers": (sides[home]["ticker"], sides[away]["ticker"])}


def market_ticker(date, team_bbr, home, away):
    """KXNBAGAME-26OCT03MIATOR-TOR for a BBR team, in Kalshi's own codes
    (BBR2KALSHI); used when grading a game that is no longer open."""
    code = {t: BBR2KALSHI.get(t, t) for t in (home, away)}
    return f"{SERIES}-{date_code(date)}{code[away]}{code[home]}-{code[team_bbr]}"


def _close(c, key):
    """A candle's closing quote (dollars) for `key` (yes_ask / yes_bid)."""
    q = c.get(key) or {}
    v = q.get("close_dollars")
    if v is None and q.get("close") is not None:
        v = float(q["close"]) / 100.0       # older payloads: cents
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < 1 else None


def quote_at(ticker, tip_ts):
    """(bid, ask) of the last 1-minute candle ending at or before `tip_ts`
    (unix seconds) within CLOSE_WINDOW that has an ask; (None, None) when
    there is none. Candles ending after tip are never read."""
    js = get(f"/series/{SERIES}/markets/{ticker}/candlesticks",
             start_ts=int(tip_ts - CLOSE_WINDOW), end_ts=int(tip_ts),
             period_interval=1)
    best = None
    for c in js.get("candlesticks", []):
        end, ask = c.get("end_period_ts"), _close(c, "yes_ask")
        if end is not None and end <= tip_ts and ask is not None \
                and (best is None or end > best[0]):
            best = (end, _close(c, "yes_bid"), ask)
    return (None, None) if best is None else best[1:]


def close_odds(date, home, away, tip_utc):
    """{'book', 'close_home_ml', 'close_away_ml'} at the last pregame minute
    (the shape market.pick_close returns), or {} when either side has no
    ask in the window. Event tickers list away then home; if that ticker is
    unknown (404) the reverse order is tried."""
    from ledger import parse_utc
    tip_ts = parse_utc(tip_utc).timestamp()
    for h_, a_ in ((home, away), (away, home)):
        try:
            h = quote_at(market_ticker(date, home, h_, a_), tip_ts)
            a = quote_at(market_ticker(date, away, h_, a_), tip_ts)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        if h == (None, None) and a == (None, None):
            continue                         # unknown ticker answered empty
        o = odds_from_quotes(h, a, "close")
        return {"book": BOOK, **o} if o else {}
    return {}
