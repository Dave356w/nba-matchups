"""Market data and price arithmetic shared by every surface.

One home for each market statistic, for the reason the MLB project learned
the hard way: two copies of an odds conversion or a standard error drift, and
a reader cannot tell which one a page used.

Sources
  ESPN site scoreboard  -> schedule, ESPN event ids, status, final scores.
  ESPN core odds        -> sportsbook moneylines: current (pregame
                           snapshot), open, close. ESPN listed ESPN BET
                           (provider 58) until late November 2025 and
                           DraftKings (100) after, so every price carries a
                           `book` label and a row takes all its prices from
                           ONE book. Provider 59 ("ESPN Bet - Live Odds")
                           carries in-game prices and is never read.

Conventions
  * American moneylines are ints; nothing strictly between -100 and +100 is a
    price, and such inputs are rejected rather than coerced.
  * `q` is the no-vig (devigged) probability: implied / (implied_h + implied_a).
  * `breakeven` is the RAW implied probability of the posted price. The gap
    q - breakeven is the per-side hold, so it is negative for any side priced
    with vig.
"""
from __future__ import annotations

import json
import time
import urllib.request

import numpy as np

HEADERS = {
    "User-Agent": "nba-matchups/1.0 (+https://github.com/Dave356w/nba-matchups)",
    "Accept": "application/json",
}
DK_PROVIDER_ID = "100"
ESPNBET_PROVIDER_ID = "58"
# Books we read, in preference order: (ESPN provider id, label). Anything not
# listed here (notably 59, ESPN BET live/in-game odds) is ignored.
BOOKS = ((DK_PROVIDER_ID, "dk"), (ESPNBET_PROVIDER_ID, "espnbet"))
BOOK_NAMES = {"dk": "DraftKings", "espnbet": "ESPN BET"}
SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
              "scoreboard?dates={ds}&limit=100")
ODDS = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
        "events/{eid}/competitions/{eid}/odds")

# ESPN abbreviation -> Basketball-Reference code (the model's team key).
ESPN2BBR = {
    "BKN": "BRK", "CHA": "CHO", "GS": "GSW", "NO": "NOP", "NY": "NYK",
    "PHX": "PHO", "SA": "SAS", "UTAH": "UTA", "UTA": "UTA", "WSH": "WAS",
}
BBR_TEAMS = frozenset((
    "ATL BOS BRK CHO CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN "
    "NOP NYK OKC ORL PHI PHO POR SAC SAS TOR UTA WAS").split())

# ESPN season.type: 1 preseason, 2 regular season, 3 postseason, 5 play-in.
REGULAR_SEASON = 2

ODDS_LADDER = (
    (None, -250, "≤ -250"),
    (-249, -175, "-249 to -175"),
    (-174, -130, "-174 to -130"),
    (-129, -100, "-129 to -100"),
    (100, 129, "+100 to +129"),
    (130, 174, "+130 to +174"),
    (175, 249, "+175 to +249"),
    (250, None, "≥ +250"),
)


# ------------------------------------------------------------ fetching -----
def get_json(url, tries=3, sleep=1.5):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 - network: retry then raise
            last = e
            time.sleep(sleep * (2 ** k))
    raise last


def _dig(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def bbr_code(espn_abbr):
    return ESPN2BBR.get(espn_abbr, espn_abbr)


def parse_scoreboard(js):
    """ESPN scoreboard JSON -> list of games keyed by BBR team codes."""
    out = []
    for ev in js.get("events", []):
        try:
            comp = ev["competitions"][0]
            teams, scores = {}, {}
            for c in comp["competitors"]:
                teams[c["homeAway"]] = bbr_code(c["team"]["abbreviation"])
                scores[c["homeAway"]] = c.get("score")
        except (KeyError, IndexError, TypeError):
            continue
        status = _dig(comp, "status", "type") or _dig(ev, "status", "type") or {}

        def _int(x):
            try:
                return int(x)
            except (TypeError, ValueError):
                return None
        out.append(dict(
            game_id=str(ev["id"]), tip_utc=ev.get("date"),
            season_type=_int(_dig(ev, "season", "type")),
            away=teams.get("away"), home=teams.get("home"),
            away_pts=_int(scores.get("away")), home_pts=_int(scores.get("home")),
            state=status.get("state"), completed=bool(status.get("completed")),
        ))
    return out


def scoreboard(date):
    """Games on an ET calendar date (YYYY-MM-DD)."""
    return parse_scoreboard(get_json(SCOREBOARD.format(ds=date.replace("-", ""))))


SUMMARY = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
           "summary?event={eid}")


def parse_injuries(js, home, away):
    """ESPN summary JSON -> injury rows for both teams, or None if absent.

    One row per listed player: team, player_id, name, status (e.g. "Out",
    "Doubtful", "Questionable", "Day-To-Day"), detail. A team with nobody
    listed gets one row with status "NONE" and no player, so "nothing
    listed" is recorded and distinguishable from "not captured" (None).
    """
    if not isinstance(js, dict) or "injuries" not in js:
        return None
    by_team = {}
    for t in js.get("injuries") or []:
        team = bbr_code((t.get("team") or {}).get("abbreviation"))
        for x in t.get("injuries") or []:
            st = x.get("status")
            if isinstance(st, dict):
                st = st.get("name") or st.get("type") or st.get("description")
            det = x.get("details") or {}
            detail = " ".join(str(v) for v in (det.get("type"), det.get("detail"))
                              if v) or x.get("shortComment") or \
                _dig(x, "type", "description") or ""
            ath = x.get("athlete") or {}
            by_team.setdefault(team, []).append(dict(
                team=team, player_id=str(ath.get("id") or ""),
                name=ath.get("displayName") or "", status=str(st or ""),
                detail=str(detail)))
    rows = []
    for team in (home, away):
        rows += by_team.get(team) or [dict(team=team, player_id="", name="",
                                           status="NONE", detail="")]
    return rows


def game_injuries(game_id, home, away):
    """Pregame injury list for one game from ESPN's summary (see parse_injuries)."""
    return parse_injuries(get_json(SUMMARY.format(eid=game_id)), home, away)


def american(x):
    """American price from an int, '+120', or ESPN's {'american': ...} dict."""
    if isinstance(x, dict):
        x = x.get("american", x.get("value"))
    if x is None:
        return None
    try:
        v = int(float(str(x).replace("+", "")))
    except (TypeError, ValueError):
        return None
    return v if (v <= -100 or v >= 100) else None


def spread_line(x):
    """A point-spread line ('-5.5', {'american': '+3'}, 'PK') -> float, else None.

    Unlike `american`, small values are valid (a line of -1.5 is a real line).
    """
    if isinstance(x, dict):
        for k in ("american", "alternateDisplayValue", "value"):
            v = spread_line(x.get(k))
            if v is not None:
                return v
        return None
    if isinstance(x, str) and x.strip().upper() in ("PK", "PICK", "EVEN"):
        return 0.0
    try:
        v = float(str(x).replace("+", "")) if x is not None else None
    except (TypeError, ValueError):
        return None
    return v if v is not None and np.isfinite(v) and abs(v) <= 60 else None


def _home_spread(h, a):
    """The home line from both sides' lines; None if missing or inconsistent."""
    if h is None and a is None:
        return None
    if h is None:
        return -a + 0.0
    if a is not None and abs(h + a) > 1e-9:
        return None
    return h


def _parse_provider(item):
    def side(which):
        td = item.get(which) or {}
        cur = american(_dig(td, "current", "moneyLine"))
        if cur is None:
            cur = american(td.get("moneyLine"))
        return dict(cur=cur, open=american(_dig(td, "open", "moneyLine")),
                    close=american(_dig(td, "close", "moneyLine")),
                    close_line=spread_line(_dig(td, "close", "pointSpread")),
                    close_spread=american(_dig(td, "close", "spread")))
    h, a = side("homeTeamOdds"), side("awayTeamOdds")
    return dict(cur_home_ml=h["cur"], cur_away_ml=a["cur"],
                open_home_ml=h["open"], open_away_ml=a["open"],
                close_home_ml=h["close"], close_away_ml=a["close"],
                close_spread=_home_spread(h["close_line"], a["close_line"]),
                close_home_spread_odds=h["close_spread"],
                close_away_spread_odds=a["close_spread"])


def parse_book_odds(js):
    """ESPN core odds JSON -> {book label: moneylines} for the books in BOOKS."""
    ids = dict(BOOKS)
    out = {}
    for it in js.get("items", []):
        book = ids.get(str(_dig(it, "provider", "id")))
        if book and book not in out:
            out[book] = dict(_parse_provider(it), book=book)
    return out


def parse_dk_odds(js):
    """ESPN core odds JSON -> DraftKings moneylines, or None without DK."""
    return parse_book_odds(js).get("dk")


def _pair_ok(o, kind):
    return bool(o) and o.get(f"{kind}_home_ml") is not None \
        and o.get(f"{kind}_away_ml") is not None


def pick_pregame(books):
    """The first book (preference order) with a current price pair, or {}."""
    for _, book in BOOKS:
        if _pair_ok(books.get(book), "cur"):
            return books[book]
    return {}


def pick_close(books, prefer=None):
    """The book to grade with: `prefer` (the row's pregame book) if it has a
    close pair, else the first book in preference order that does. Open and
    close then come from that one book. {} when no book has a close."""
    order = ([prefer] if prefer else []) + [b for _, b in BOOKS if b != prefer]
    for book in order:
        if _pair_ok(books.get(book), "close"):
            return books[book]
    return {}


def book_odds(game_id):
    return parse_book_odds(get_json(ODDS.format(eid=game_id)))


def dk_odds(game_id):
    return book_odds(game_id).get("dk")


# ------------------------------------------------------------ arithmetic ---
def implied(ml):
    """Raw implied probability (break-even) of one American price; NaN if invalid."""
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(ml) or -100 < ml < 100:
        return float("nan")
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def breakeven_prob(mls):
    """Vectorised `implied`: the win rate a bet at each price must clear."""
    return np.asarray([implied(m) for m in np.asarray(mls, dtype=float).ravel()])


def devig(home_ml, away_ml):
    """No-vig home probability from a two-way moneyline pair; NaN if invalid."""
    ih, ia = implied(home_ml), implied(away_ml)
    if not (np.isfinite(ih) and np.isfinite(ia)) or ih + ia <= 0:
        return float("nan")
    return ih / (ih + ia)


def unit_profit(ml, won):
    """Flat one-unit P/L at an American price; NaN for an invalid price."""
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(ml) or -100 < ml < 100:
        return float("nan")
    if not won:
        return -1.0
    return ml / 100.0 if ml > 0 else 100.0 / -ml


def ats_result(home_margin, home_spread):
    """Home side against the spread: 1 cover, 0 miss, 0.5 push; NaN if invalid.

    `home_spread` is the home line (-5.5 = home gives 5.5 points), so home
    covers when home_margin + home_spread > 0. The away side is 1 - this.
    """
    try:
        v = float(home_margin) + float(home_spread)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    return 1.0 if v > 0 else 0.0 if v < 0 else 0.5


def decimal_payout(ml):
    return 1.0 + unit_profit(ml, True)


def ladder_rung(ml):
    """Rung label for an American price, or None if it is not a real price."""
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ml) or -100 < ml < 100:
        return None
    for lo, hi, label in ODDS_LADDER:
        if (lo is None or ml >= lo) and (hi is None or ml <= hi):
            return label
    return None


def excess_se(probs):
    """SE of (realised rate - mean implied) if every price were correct.

    Wins are a Poisson-binomial sum at the market's own prices, so
    Var = sum p(1-p) and SE = sqrt(sum p(1-p)) / n. The spread comes from the
    prices, never from the outcomes under test, so an all-W or all-L bucket
    does not report a spuriously tiny error bar.
    """
    p = np.asarray(list(probs), dtype=float)
    if not p.size:
        return float("nan")
    return float(np.sqrt(float((p * (1.0 - p)).sum())) / p.size)


def ev_null(probs, breakevens):
    """Expected (realised - breakeven) when the devigged market is RIGHT.

    Against the posted break-even a correctly priced book yields
    mean(q) - mean(breakeven): minus the hold, not zero. Every EV figure is
    printed beside this null so it is never read against zero. Shares the
    excess column's SE (the two differ by a price-fixed constant). NaN, not
    0, on invalid input.
    """
    p = np.asarray(list(probs), dtype=float)
    be = np.asarray(list(breakevens), dtype=float)
    if not p.size or p.size != be.size or not np.isfinite(be).all() \
            or not np.isfinite(p).all():
        return float("nan")
    return float(p.mean() - be.mean())


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return (p - y) ** 2


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
