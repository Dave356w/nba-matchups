"""Market data and price arithmetic shared by every surface.

One home for each market statistic, for the reason the MLB project learned
the hard way: two copies of an odds conversion or a standard error drift, and
a reader cannot tell which one a page used.

Sources
  ESPN site scoreboard  -> schedule, ESPN event ids, status, final scores.
  ESPN core odds        -> DraftKings (provider id 100) moneylines:
                           current (pregame snapshot), open, close.

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


def parse_dk_odds(js):
    """ESPN core odds JSON -> DraftKings moneylines, or None without DK."""
    dk = next((i for i in js.get("items", [])
               if str(_dig(i, "provider", "id")) == DK_PROVIDER_ID), None)
    if dk is None:
        return None

    def side(which):
        td = dk.get(which) or {}
        cur = american(_dig(td, "current", "moneyLine"))
        if cur is None:
            cur = american(td.get("moneyLine"))
        return dict(cur=cur, open=american(_dig(td, "open", "moneyLine")),
                    close=american(_dig(td, "close", "moneyLine")))
    h, a = side("homeTeamOdds"), side("awayTeamOdds")
    return dict(cur_home_ml=h["cur"], cur_away_ml=a["cur"],
                open_home_ml=h["open"], open_away_ml=a["open"],
                close_home_ml=h["close"], close_away_ml=a["close"])


def dk_odds(game_id):
    return parse_dk_odds(get_json(ODDS.format(eid=game_id)))


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
