"""Player availability: box scores, last-season BPM, NBA injury reports.

Shared by the daily build (model v3, games 10+) and the research scripts
(research/availability.py, research/pregame_availability.py), so the
backtest and production compute the availability terms the same way.

  python player_availability.py fit --years 2023-2026 [--cache research/output] [--no-phase]

fits model/logit_avail.json: P(home) = sigma(a + b*delta + c*b2b_net +
d1*av_min + d2*av_bpm + e*delta*phase) on games 10+ (v4; --no-phase drops
the last term for v3), with av_* from the NBA injury report
at least LEAD_MINUTES before tip (Out/Doubtful = out; anyone else on the
previous box score plays) and last-season BBR BPM values. See MODEL.md.
"""
from __future__ import annotations

import argparse
import bisect
import io
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import ledger
import market
import nba_composite as nc

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(ROOT, "research", "output")
REPORT_DIR = os.path.join(DEFAULT_CACHE, "injury_reports")


def _team_names():
    import build_site            # lazy: build_site imports this module
    return build_site.TEAM_NAMES


SUMMARY = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
           "summary?event={eid}")


SHRINK = 1000.0          # decayed on-court minutes for half weight on on/off


REPLACEMENT_BPM = -2.0   # replacement level; unmatched players (rookies) get it


BPM_SHRINK_MP = 500.0    # last-season minutes for half weight on BPM


# ----------------------------------------------------------- box scores ----
def _minutes(x):
    try:
        v = float(str(x).split(":")[0])
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) and v > 0 else 0.0


def _pm(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return 0.0


def parse_players(js):
    """ESPN summary JSON -> [{team, player_id, name, minutes, pm}] (BBR codes).

    Players listed without minutes (DNP, inactive) get minutes 0.
    """
    rows = []
    for t in (js.get("boxscore") or {}).get("players") or []:
        team = market.bbr_code((t.get("team") or {}).get("abbreviation"))
        for block in t.get("statistics") or []:
            keys = block.get("keys") or []
            i_min = keys.index("minutes") if "minutes" in keys else None
            i_pm = keys.index("plusMinus") if "plusMinus" in keys else None
            for a in block.get("athletes") or []:
                st = a.get("stats") or []
                ath = a.get("athlete") or {}
                if not ath.get("id"):
                    continue                 # unidentifiable row: skip
                rows.append(dict(
                    team=team, player_id=str(ath.get("id")),
                    name=ath.get("displayName"),
                    minutes=_minutes(st[i_min]) if i_min is not None
                    and i_min < len(st) else 0.0,
                    pm=_pm(st[i_pm]) if i_pm is not None and i_pm < len(st) else 0.0))
    return rows


# Regular seasons that ran past mid-April (2019-20 bubble, 2020-21 delayed).
SEASON_END = {2020: "08-20", 2021: "05-20"}


def season_dates(y):
    """Every date that can hold a regular-season game of season y."""
    return pd.date_range(f"{y - 1}-10-15", f"{y}-{SEASON_END.get(y, '04-20')}")


def box_rows_for_date(d, sleep=0.15):
    """Box rows for every completed regular-season NBA game on ET date d."""
    ds = pd.Timestamp(d).strftime("%Y-%m-%d")
    rows, n_games = [], 0
    try:
        games = market.scoreboard(ds)
    except Exception as e:  # noqa: BLE001
        print(f"{ds}: scoreboard failed {e!r}", flush=True)
        return rows, 0, 1
    pending = 0
    for g in games:
        if g["season_type"] != market.REGULAR_SEASON \
                or g["home"] not in market.BBR_TEAMS \
                or g["away"] not in market.BBR_TEAMS:
            continue
        if not g["completed"]:
            pending += 1
            continue
        try:
            js = market.get_json(SUMMARY.format(eid=g["game_id"]))
        except Exception as e:  # noqa: BLE001
            print(f"{g['game_id']}: summary failed {e!r}", flush=True)
            continue
        time.sleep(sleep)
        n_games += 1
        for pl in parse_players(js):
            home = pl["team"] == g["home"]
            if pl["team"] not in (g["home"], g["away"]):
                continue
            rows.append(dict(
                game_id=g["game_id"], date=pd.Timestamp(ds), team=pl["team"],
                opp=g["away"] if home else g["home"], home=home,
                margin=(g["home_pts"] - g["away_pts"]) * (1 if home else -1),
                player_id=pl["player_id"], name=pl["name"],
                minutes=pl["minutes"], pm=pl["pm"]))
    return rows, n_games, pending


def fetch_box(y, sleep=0.15, cache_dir=None):
    """Player minutes and +/- for every regular-season game of season y."""
    cache_dir = cache_dir or DEFAULT_CACHE
    path = os.path.join(cache_dir, f"box_{y}.csv")
    if os.path.exists(path):
        # keep_default_na=False: ids like "None" must stay strings, never NaN
        df = pd.read_csv(path, dtype={"game_id": str, "player_id": str, "name": str},
                         keep_default_na=False, parse_dates=["date"])
        return clean_box(df)
    rows, n_games = [], 0
    for d in season_dates(y):
        got, n, _ = box_rows_for_date(d, sleep)
        rows += got
        n_games += n
    df = clean_box(pd.DataFrame(rows))
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"box {y}: {n_games} games, {len(df)} player rows", flush=True)
    return df


# --------------------------------------------------- last-season BPM (v2) --
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name):
    """'Nikola Jokić' / 'P.J. Washington Jr.' -> 'nikola jokic' / 'pj washington'."""
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    return " ".join(_SUFFIX.sub("", s).split())


def parse_advanced(html, stat="BPM"):
    """BBR NBA_{y}_advanced.html -> {norm name: (bpm, mp, games)}, one row per
    player (the multi-team total row where present: the most minutes). `stat`
    picks another column in place of BPM (research: "WS/48", "WS")."""
    m = re.search(r'<table[^>]*id="advanced(?:_stats)?".*?</table>', nc.uncomment(html), re.S)
    if not m:
        return {}
    df = pd.read_html(io.StringIO(m.group(0)))[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    df = df[df["Player"].notna() & (df["Player"] != "Player")].copy()
    df = df[~df["Player"].astype(str).str.contains("League Average")]
    for c in (stat, "MP", "G"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[stat, "MP"])
    df["key"] = df["Player"].map(norm_name)
    df = df.sort_values("MP", ascending=False).drop_duplicates("key")
    return {k: (float(b), float(mp), float(g) if pd.notna(g) else np.nan)
            for k, b, mp, g in zip(df["key"], df[stat], df["MP"], df["G"])}


def load_bpm(y, stat="BPM"):
    """Season y's BPM table (use y - 1 for season y's games: no lookahead)."""
    html = nc.fetch(f"{nc.BASE}/leagues/NBA_{y}_advanced.html",
                    nc.CACHE / f"advanced_{y}.html")
    return parse_advanced(html, stat)


def player_values(box, bpm):
    """player_id -> points per game above replacement per full 48 minutes:
    shrunk last-season BPM minus REPLACEMENT_BPM (0 for unmatched)."""
    names = box.drop_duplicates("player_id").set_index("player_id")["name"]
    mins = box.groupby("player_id")["minutes"].sum()
    vals, hit, hit_min = {}, 0, 0.0
    for pid, nm in names.items():
        rec = bpm.get(norm_name(nm))
        if rec is None:
            vals[pid] = 0.0
            continue
        hit += 1
        hit_min += float(mins.get(pid, 0.0))
        b, mp, _ = rec
        shrunk = REPLACEMENT_BPM + (b - REPLACEMENT_BPM) * mp / (mp + BPM_SHRINK_MP)
        vals[pid] = shrunk - REPLACEMENT_BPM
    return vals, hit / max(len(names), 1), hit_min / max(float(mins.sum()), 1.0)


def arrival_roles(box, bpm):
    """(player_id, date) -> minutes role for a player new to a team: his mean
    minutes / 48 in games played before `date` this season (any team), else
    last season's MP / G / 48, else 0."""
    hist = {}
    for pid, g in box[box["minutes"] > 0].groupby("player_id"):
        g = g.sort_values("date")
        d = g["date"].to_numpy()
        c = np.cumsum(g["minutes"].to_numpy(float))
        hist[pid] = (d, c)
    names = box.drop_duplicates("player_id").set_index("player_id")["name"]

    def role(pid, date):
        h = hist.get(pid)
        if h is not None:
            i = bisect.bisect_left(list(h[0]), np.datetime64(pd.Timestamp(date)))
            if i > 0:
                return float(h[1][i - 1] / i / 48.0)
        rec = bpm.get(norm_name(names.get(pid, "")))
        if rec is not None and rec[2] and np.isfinite(rec[2]) and rec[2] > 0:
            return float(rec[1] / rec[2] / 48.0)
        return 0.0
    return role


def talent_fn(box, value, role):
    """(team, date) -> sum over the players on the team's previous box score
    (minutes > 0) of role(player, date) * value[player]: minutes share x
    last-season value above replacement, the roster as it stood before the
    game (model v5's talent_diff; research/team_quality.py). NaN before the
    team's first game."""
    b = box[box["minutes"] > 0]
    by_team = {tm: g.sort_values("date") for tm, g in b.groupby("team")}

    def f(tm, date):
        g = by_team.get(tm)
        if g is None:
            return float("nan")
        prev = g[g["date"] < pd.Timestamp(date)]
        if not len(prev):
            return float("nan")
        last = prev[prev["date"] == prev["date"].max()]
        return float(sum(role(pid, date) * value.get(pid, 0.0)
                         for pid in last["player_id"]))
    return f


def season_talent(y, cache_dir=None):
    """talent_fn for season y from its box scores (fetch_box, cached) and
    season y - 1's BPM table (no lookahead)."""
    box = fetch_box(y, cache_dir=cache_dir or DEFAULT_CACHE)
    bpm = load_bpm(y - 1)
    value, _, _ = player_values(box, bpm)
    return talent_fn(box, value, arrival_roles(box, bpm))


def clean_box(df):
    """Drop rows without a real player id (fresh and cached data alike)."""
    if not len(df):
        return df
    pid = df["player_id"].astype(str).str.strip()
    df = df[~pid.isin(["", "None", "nan", "NaN"])].copy()
    df["player_id"] = df["player_id"].astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    for c in ("minutes", "pm", "margin"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["home"] = df["home"].astype(str).isin(["True", "true", "1"])
    return df


# ---------------------------------------------------------- availability ---
def team_availability(tg, half_life=nc.HALF_LIFE, shrink=SHRINK, value=None,
                      arrival_role=None, present=None):
    """One team-season of player rows -> {game_id: (av_min, av_oo, av_bpm)}.

    Game k uses only games before k for a_i, role_i and r_i; p_i is game k's
    own minutes (the hindsight part). av_bpm weights each player by
    value[player_id] (points above replacement per 48; v2) and also counts
    players new to the team who play today (a = 0) with role from
    arrival_role(player_id, date). v1 terms (av_min, av_oo) are unchanged.

    present (pregame mode, research/pregame_availability.py) replaces the
    hindsight p_i: {(game_id, player_id): P(plays)} from the injury report;
    a player not in it counts as playing (1) if he was on the team's
    previous box score (listed at all), else 0 (traded, released).
    Arrivals are not counted in pregame mode.
    """
    value = value or {}
    games = (tg[["game_id", "date", "margin"]].drop_duplicates("game_id")
             .sort_values(["date", "game_id"]).reset_index(drop=True))
    order = {g: k for k, g in enumerate(games["game_id"])}
    n = len(games)
    players = sorted(tg["player_id"].unique())
    pidx = {p: j for j, p in enumerate(players)}
    M = np.zeros((n, len(players)))
    PM = np.zeros((n, len(players)))
    L = np.zeros((n, len(players)), bool)          # listed on that box at all
    for r in tg.itertuples(index=False):
        k, j = order[r.game_id], pidx[r.player_id]
        M[k, j] = r.minutes
        PM[k, j] = r.pm
        L[k, j] = True
    v = np.array([value.get(p, 0.0) for p in players])
    margin = games["margin"].to_numpy(float)
    out = {}
    for k in range(n):
        if present is None:
            p = (M[k] > 0).astype(float)
        else:
            gid = games.at[k, "game_id"]
            prev = L[k - 1] if k > 0 else np.zeros(len(players), bool)
            p = np.array([present.get((gid, pl), 1.0 if prev[j] else 0.0)
                          for j, pl in enumerate(players)])
        if k == 0:
            wp = np.zeros(len(players))
            a = role = r = np.zeros(len(players))
        else:
            w = 0.5 ** (np.arange(k)[::-1] / half_life)
            m, pm = M[:k], PM[:k]
            played = m > 0
            wp = (w[:, None] * played).sum(0)
            a = wp / w.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                role = np.where(wp > 0, (w[:, None] * m).sum(0) / wp / 48.0, 0.0)
                on_min = (w[:, None] * m).sum(0)
                on_pm = (w[:, None] * pm).sum(0)
                off_min = (w[:, None] * np.where(played, np.clip(48.0 - m, 1.0, None),
                                                 0)).sum(0)
                off_pm = (w[:, None] * np.where(played, margin[:k, None] - pm, 0)).sum(0)
                r = np.where((on_min > 0) & (off_min > 0),
                             48.0 * (on_pm / on_min - off_pm / off_min), 0.0)
            r = r * on_min / (on_min + shrink)
        known = wp > 0
        dp = (p - a) * known
        av_bpm = float((role * v * dp).sum())
        if arrival_role is not None and present is None:
            date = games.at[k, "date"]
            for j in np.where((p > 0) & ~known)[0]:
                av_bpm += arrival_role(players[j], date) * v[j]
        out[games.at[k, "game_id"]] = (float((role * dp).sum()),
                                       float((role * r * dp).sum()), av_bpm)
    return out


def game_availability(box, value=None, arrival_role=None, present=None):
    """Box rows -> one row per game: date, home, away, av_min, av_oo, av_bpm
    (home - away)."""
    feats = {}
    for _, tg in box.groupby("team"):
        for gid, v in team_availability(tg, value=value, arrival_role=arrival_role,
                                        present=present).items():
            feats[(gid, tg["team"].iloc[0])] = v
    g = box[box["home"]].drop_duplicates("game_id")[["game_id", "date", "team", "opp"]]
    rows = []
    for r in g.itertuples(index=False):
        h, a = feats.get((r.game_id, r.team)), feats.get((r.game_id, r.opp))
        if h is None or a is None:
            continue
        rows.append(dict(game_id=r.game_id,
                         slate_date=pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                         home=r.team, away=r.opp,
                         av_min=h[0] - a[0], av_oo=h[1] - a[1],
                         av_bpm=h[2] - a[2]))
    return pd.DataFrame(rows)


ET = ZoneInfo("America/New_York")


REPORT_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{slot}.pdf"


STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")


SLOT_MINUTES = 15


MAX_LOOKBACK_H = 12
RECENT_H = 6


# ------------------------------------------------------------- reports ----
def slot_names(dt):
    """Filename time parts for an ET datetime: '05_30PM', and '05PM' on the hour
    (the archive has used both)."""
    h12 = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    names = [f"{h12:02d}_{dt.minute:02d}{ampm}"]
    if dt.minute == 0:
        names.insert(0, f"{h12:02d}{ampm}")
    return names


def floor_slot(dt):
    return dt.replace(minute=dt.minute - dt.minute % SLOT_MINUTES, second=0,
                      microsecond=0)


class ReportArchive:
    """Finds and caches injury-report PDFs; remembers misses across runs."""

    def __init__(self, cache_dir=REPORT_DIR, fetch=None, sleep=0.05, now=None):
        # now (ET, naive): slots within RECENT_H of it may not be published
        # yet, so a miss there is not remembered (live use); None = research.
        self.now = now
        self.dir = cache_dir
        os.makedirs(self.dir, exist_ok=True)
        self.miss_path = os.path.join(self.dir, "misses.json")
        try:
            self.misses = set(json.load(open(self.miss_path)))
        except (OSError, ValueError):
            self.misses = set()
        self.fetch = fetch or self._http
        self.sleep = sleep
        self.requests = 0

    def _http(self, url):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
        except urllib.error.HTTPError:
            return None
        return body if body[:4] == b"%PDF" else None

    def get(self, dt):
        """Local path of the report published at ET datetime dt, or None."""
        for slot in slot_names(dt):
            name = f"{dt:%Y-%m-%d}_{slot}"
            path = os.path.join(self.dir, name + ".pdf")
            if os.path.exists(path):
                return path
            if name in self.misses:
                continue
            self.requests += 1
            body = self.fetch(REPORT_URL.format(date=f"{dt:%Y-%m-%d}", slot=slot))
            time.sleep(self.sleep)
            if body:
                open(path, "wb").write(body)
                return path
            if self.now is None or dt < self.now - timedelta(hours=RECENT_H):
                self.misses.add(name)
        return None

    def latest_before(self, cutoff):
        """(report datetime, path) of the last report at or before cutoff."""
        dt = floor_slot(cutoff)
        stop = cutoff - timedelta(hours=MAX_LOOKBACK_H)
        while dt >= stop:
            path = self.get(dt)
            if path:
                return dt, path
            dt -= timedelta(minutes=SLOT_MINUTES)
        return None, None

    def save(self):
        json.dump(sorted(self.misses), open(self.miss_path, "w"))


HEADER_CANON = {"gamedate": "date", "gametime": "time", "matchup": "matchup",
                "team": "team", "playername": "player", "currentstatus": "status",
                "reason": "reason"}


PAIRS = {"game": ("date", "time"), "player": ("name",), "current": ("status",)}


PARSER_VERSION = 4          # bump to invalidate cached parsed-report CSVs


def header_columns(line):
    """[(column, x0)] if this word line is the table header, else None.

    Accepts header words split ('Game', 'Date') or merged ('GameDate'), as
    PDFs without space glyphs give merged words.
    """
    toks, i = [], 0
    while i < len(line):
        t = line[i]["text"].replace(" ", "").lower()
        nxt = line[i + 1]["text"].replace(" ", "").lower() if i + 1 < len(line) else ""
        if t in PAIRS and nxt in PAIRS[t]:
            toks.append((t + nxt, float(line[i]["x0"])))
            i += 2
            continue
        toks.append((t, float(line[i]["x0"])))
        i += 1
    cols = [(HEADER_CANON[t], x) for t, x in toks if t in HEADER_CANON]
    names = {c for c, _ in cols}
    if {"matchup", "player", "status", "reason"} <= names:
        return sorted(cols, key=lambda c: c[1])
    return None


def status_of(text):
    """'Out' / 'OutInjury/Illness' -> 'Out'; else ''."""
    t = str(text).strip()
    for st in STATUSES:
        if t.startswith(st):
            return st
    return ""


def rows_from_words(pages):
    """pdfplumber words per page -> report rows.

    pages: list of word lists ({text, x0, top}). Columns come from the header
    line (Game Date | Game Time | Matchup | Team | Player Name | Current
    Status | Reason); date, time, matchup and team carry down until they
    change. Returns [{game_date, matchup, team, player, status}]; a team
    marked NOT YET SUBMITTED gives a row with player "" and status
    NOT_SUBMITTED (coverage, see report_coverage).
    """
    rows, cols = [], None
    carry = {"date": "", "time": "", "matchup": "", "team": ""}
    for words in pages:
        lines = {}
        for w in words:
            lines.setdefault(round(float(w["top"]) / 3), []).append(w)
        for key in sorted(lines):
            line = sorted(lines[key], key=lambda w: float(w["x0"]))
            hdr = header_columns(line)
            if hdr:
                cols = hdr
                continue
            if cols is None:
                continue
            cells = {}
            for w in line:
                col = None
                for name, x in cols:
                    if x <= float(w["x0"]) + 3:
                        col = name
                if col:
                    cells.setdefault(col, []).append(w["text"])
            cells = {k: " ".join(v) for k, v in cells.items()}
            for k in carry:
                if cells.get(k):
                    carry[k] = cells[k]
            # Read player, status and reason as one stretch: a status word
            # that starts left of its header (long words in a centred column)
            # lands in the player cell, a short one may spill into reason.
            tail = " ".join(cells.get(c, "") for c in ("player", "status", "reason"))
            m = _ROW.match(tail)
            if m:
                rows.append(dict(game_date=carry["date"], matchup=carry["matchup"],
                                 team=carry["team"], player=m.group("player").strip(),
                                 status=m.group("status")))
            elif _NYS.search(tail) and carry["team"]:
                rows.append(dict(game_date=carry["date"], matchup=carry["matchup"],
                                 team=carry["team"], player="", status=NOT_SUBMITTED))
    return rows


# A team that has not filed yet: a coverage row (player ""), never a player.
NOT_SUBMITTED = "Not Yet Submitted"
_NYS = re.compile(r"NOT\s*YET\s*SUBMITTED", re.I)


_DATE = re.compile(r"\b(\d{2}/\d{2}/\d{2,4})\b")


_TIME = re.compile(r"\d{1,2}:\d{2}\s*\(ET\)")


_MATCHUP = re.compile(r"\b([A-Z]{2,3}@[A-Z]{2,3})\b")


_ROW = re.compile(r"^\s*(?P<player>[A-Z][^,]{1,40},\s*[A-Za-z.'\-]+?)\s*"
                  r"(?P<status>Out|Doubtful|Questionable|Probable|Available)"
                  r"(?![a-z])")


def full_team_names():
    out = {}
    for code, nick in _team_names().items():
        city = CITIES[code]
        out[code] = city if city.endswith(nick) else f"{city} {nick}"
    return out


def rows_from_text(pages_text):
    """Fallback: plain-text lines -> report rows, by regular expressions.

    Strips date, time and matchup (carried down), then a full team name
    (spaced or not; carried down), then reads 'Last, First' and the status.
    A team marked NOT YET SUBMITTED gives a coverage row (player "").
    """
    teams = full_team_names()
    rows = []
    carry = {"date": "", "matchup": "", "team": ""}
    for text in pages_text:
        for line in str(text or "").splitlines():
            s = line
            m = _DATE.search(s)
            if m:
                carry["date"] = m.group(1)
                s = s.replace(m.group(0), " ")
            s = _TIME.sub(" ", s)
            m = _MATCHUP.search(s)
            if m:
                carry["matchup"] = m.group(1)
                s = s.replace(m.group(0), " ")
            for full in sorted(teams.values(), key=len, reverse=True):
                for variant in (full, full.replace(" ", "")):
                    if variant in s:
                        carry["team"] = full
                        s = s.replace(variant, " ", 1)
                        break
            m = _ROW.match(s)
            if m and carry["team"]:
                rows.append(dict(game_date=carry["date"], matchup=carry["matchup"],
                                 team=carry["team"], player=m.group("player").strip(),
                                 status=m.group("status")))
            elif _NYS.search(s) and carry["team"]:
                rows.append(dict(game_date=carry["date"], matchup=carry["matchup"],
                                 team=carry["team"], player="", status=NOT_SUBMITTED))
    return rows


_DUMPED = [0]


def parse_report(path):
    """Report PDF -> DataFrame of rows (cached next to the PDF as CSV).

    Column-position parse first; if that finds nothing, the text fallback.
    The first two reports that still yield nothing are dumped to the log.
    """
    csv = path[:-4] + f".v{PARSER_VERSION}.csv"
    if os.path.exists(csv):
        return pd.read_csv(csv, dtype=str, keep_default_na=False)
    import pdfplumber  # research-only dependency, installed by the workflow
    with pdfplumber.open(path) as pdf:
        pages = [p.extract_words(keep_blank_chars=False) for p in pdf.pages]
        rows = rows_from_words(pages)
        if not rows:
            texts = [p.extract_text() or "" for p in pdf.pages]
            rows = rows_from_text(texts)
            if not rows and _DUMPED[0] < 2:
                _DUMPED[0] += 1
                print(f"\n--- unparsed report {os.path.basename(path)}: first page text"
                      f"\n{texts[0][:1500] if texts else ''}\n--- first 40 words: " +
                      "; ".join(f"{w['text']!r}@{float(w['x0']):.0f},{float(w['top']):.0f}"
                                for w in (pages[0][:40] if pages else [])) +
                      "\n---", flush=True)
    df = pd.DataFrame(rows, columns=["game_date", "matchup", "team", "player", "status"])
    df.to_csv(csv, index=False)
    return df


CITIES = {
    "ATL": "Atlanta", "BOS": "Boston", "BRK": "Brooklyn", "CHO": "Charlotte",
    "CHI": "Chicago", "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver",
    "DET": "Detroit", "GSW": "Golden State", "HOU": "Houston", "IND": "Indiana",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis",
    "MIA": "Miami", "MIL": "Milwaukee", "MIN": "Minnesota", "NOP": "New Orleans",
    "NYK": "New York", "OKC": "Oklahoma City", "ORL": "Orlando",
    "PHI": "Philadelphia", "PHO": "Phoenix", "POR": "Portland",
    "SAC": "Sacramento", "SAS": "San Antonio", "TOR": "Toronto", "UTA": "Utah",
    "WAS": "Washington",
}


def team_code(full_name):
    """'Boston Celtics' / 'LA Clippers' / a wrapped 'Oklahoma City' -> BBR code."""
    s = " ".join(str(full_name).split())
    if not s:
        return None
    for code, nick in _team_names().items():
        if s.endswith(nick):
            return code
    hits = [c for c, city in CITIES.items() if city.startswith(s) or s.startswith(city)]
    return hits[0] if len(hits) == 1 else None


def report_name_to_first_last(name):
    """'Porter Jr., Michael' -> 'Michael Porter Jr.'."""
    last, _, first = str(name).partition(",")
    return f"{first.strip()} {last.strip()}".strip()


def report_date(s):
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):     # %Y would read "24" as year 0024
        try:
            return datetime.strptime(str(s).strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ------------------------------------------------------------ coverage ----
# NBA tricodes (report matchups, e.g. "PHX@BKN") that differ from BBR codes.
NBA_TO_BBR = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}


def matchup_teams(s):
    """'NYK@BOS' -> ('NYK', 'BOS') as BBR (away, home) codes, else None."""
    m = _MATCHUP.search(str(s).replace(" ", ""))
    if not m:
        return None
    away, home = m.group(1).split("@")
    return NBA_TO_BBR.get(away, away), NBA_TO_BBR.get(home, home)


def report_coverage(report_rows, slate):
    """What a parsed report says about slate date `slate` (YYYY-MM-DD):
    {'matchups': {(away, home)}, 'teams': {codes with a row},
     'pending': {codes marked NOT YET SUBMITTED}}."""
    out = dict(matchups=set(), teams=set(), pending=set())
    if report_rows is None or not len(report_rows):
        return out
    for r in report_rows.itertuples(index=False):
        if report_date(r.game_date) != slate:
            continue
        mt = matchup_teams(getattr(r, "matchup", ""))
        if mt:
            out["matchups"].add(mt)
        code = team_code(r.team)
        if code is None:
            continue
        out["teams"].add(code)
        if r.status == NOT_SUBMITTED:
            out["pending"].add(code)
    return out


def covers(cov, home, away):
    """True when the report speaks for this game: its matchup (or both teams)
    is on it and neither team is NOT YET SUBMITTED. A game the report does
    not reach -- empty or unparsed report, another slate, a team that has not
    filed -- is not covered and must fall back to the base model, never be
    read as 'nobody is out'."""
    if home in cov["pending"] or away in cov["pending"]:
        return False
    return (away, home) in cov["matchups"] or {home, away} <= cov["teams"]


# ---------------------------------------------------------------- tips ----
def fetch_tips(y, cache_dir=None):
    """game_id -> tip_utc for every regular-season game of season y."""
    cache_dir = cache_dir or DEFAULT_CACHE
    path = os.path.join(cache_dir, f"tips_{y}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        return dict(zip(df["game_id"], df["tip_utc"]))
    rows = []
    for d in season_dates(y):
        try:
            games = market.scoreboard(d.strftime("%Y-%m-%d"))
        except Exception as e:  # noqa: BLE001
            print(f"{d:%Y-%m-%d}: scoreboard failed {e!r}", flush=True)
            continue
        rows += [dict(game_id=g["game_id"], tip_utc=g["tip_utc"]) for g in games
                 if g["season_type"] == market.REGULAR_SEASON]
    df = pd.DataFrame(rows, columns=["game_id", "tip_utc"]).drop_duplicates("game_id")
    df.to_csv(path, index=False)
    return dict(zip(df["game_id"], df["tip_utc"]))


# --------------------------------------------------------- statuses -------
def game_statuses(box, tips, archive, lead_minutes):
    """One row per (game, report-listed player matched to the box): game_id,
    team, player_id, status, played, report time, lead (minutes before tip).
    stats['covered'] is the set of game_ids the report covers (see covers());
    only those may use report-based availability terms."""
    games = box.drop_duplicates(["game_id", "team"])[["game_id", "date", "team"]]
    names = {}                                  # (team, norm name) -> player_id
    for r in box.drop_duplicates(["team", "player_id"]).itertuples(index=False):
        names.setdefault((r.team, norm_name(r.name)), r.player_id)
    played = {(r.game_id, r.player_id): r.minutes > 0
              for r in box.itertuples(index=False)}
    out, stats = [], dict(games=0, with_report=0, listed=0, matched=0, lags=[],
                          reports=0, reports_rows=0, covered=set())
    parsed, coverage = {}, {}
    home_of = dict(box[box["home"].astype(bool)].drop_duplicates("game_id")
                   [["game_id", "team"]].itertuples(index=False))
    for gid, g in games.groupby("game_id"):
        tip = ledger.parse_utc(tips.get(gid))
        stats["games"] += 1
        if tip is None:
            continue
        cutoff = tip.astimezone(ET).replace(tzinfo=None) - timedelta(minutes=lead_minutes)
        rdt, path = archive.latest_before(cutoff)
        if path is None:
            if stats["games"] >= 20 and stats["with_report"] == 0:
                raise SystemExit(
                    "no injury report found for the first 20 games; tried e.g. " +
                    REPORT_URL.format(date=f"{cutoff:%Y-%m-%d}",
                                      slot=slot_names(floor_slot(cutoff))[0]))
            continue
        if path not in parsed:
            try:
                parsed[path] = parse_report(path)
            except Exception as e:  # noqa: BLE001
                print(f"parse failed {os.path.basename(path)}: {e!r}", flush=True)
                parsed[path] = pd.DataFrame(columns=["game_date", "team", "player",
                                                     "status"])
            stats["reports"] += 1
            stats["reports_rows"] += int(len(parsed[path]) > 0)
        rep = parsed[path]
        slate = pd.Timestamp(g["date"].iloc[0]).strftime("%Y-%m-%d")
        stats["with_report"] += 1
        stats["lags"].append((tip.astimezone(ET).replace(tzinfo=None) - rdt)
                             .total_seconds() / 60)
        teams = set(g["team"])
        if (path, slate) not in coverage:
            coverage[(path, slate)] = report_coverage(rep, slate)
        h = home_of.get(gid)
        if len(teams) == 2 and h in teams and covers(coverage[(path, slate)], h,
                                                     (teams - {h}).pop()):
            stats["covered"].add(gid)
        for r in rep.itertuples(index=False):
            code = team_code(r.team)
            if (code not in teams or report_date(r.game_date) != slate
                    or r.status == NOT_SUBMITTED):
                continue
            stats["listed"] += 1
            pid = names.get((code, norm_name(report_name_to_first_last(r.player))))
            if pid is None:
                continue
            stats["matched"] += 1
            out.append(dict(game_id=gid, team=code, player_id=pid, status=r.status,
                            played=bool(played.get((gid, pid), False)),
                            report=f"{rdt:%Y-%m-%d %H:%M}"))
    return pd.DataFrame(out, columns=["game_id", "team", "player_id", "status",
                                      "played", "report"]), stats


def play_rates(st):
    """status -> share of listed players who logged minutes."""
    if not len(st):
        return {}
    return st.groupby("status")["played"].mean().to_dict()


def present_map(st, arm, rates=None):
    """{(game_id, player_id): P(plays)} for one arm (see module docstring)."""
    rates = rates or {}
    out = {}
    for r in st.itertuples(index=False):
        if arm == "od":
            p = 0.0 if r.status in ("Out", "Doubtful") else 1.0
        else:
            p = {"Out": 0.0, "Available": 1.0}.get(r.status, rates.get(r.status, 1.0))
        out[(r.game_id, r.player_id)] = float(p)
    return out


# ------------------------------------------------------------ model v3 -----
MODEL_FILE = "logit_avail.json"
FEATURES = ["delta", "b2b_net", "av_min", "av_bpm"]
# v4 adds the season-phase slope term (nba_composite.PHASE_FEATURES).
FEATURES_V4 = FEATURES + ["d_phase"]
# v5 adds opponent 3-point luck and roster talent (nba_composite.V5_FEATURES).
FEATURES_V5 = FEATURES_V4 + ["luck_def", "talent_diff"]
# v6: + own FT% gap (nba_composite.V6_FEATURES).
FEATURES_V6 = FEATURES_V5 + ["ft_diff"]
LEAD_MINUTES = 30        # fit: last report at least this long before tip
BOX_COLUMNS = ["game_id", "date", "team", "opp", "home", "margin", "player_id",
               "name", "minutes", "pm"]


def update_box(y, today, data_dir="data", sleep=0.15):
    """This season's box rows for games strictly before `today` (ET date),
    kept in data/nba_box_<y>.csv and extended incrementally. A date is marked
    done (data/nba_box_<y>_dates.txt) only when none of its games is pending."""
    path = os.path.join(data_dir, f"nba_box_{y}.csv")
    done_path = os.path.join(data_dir, f"nba_box_{y}_dates.txt")
    if os.path.exists(path):
        df = clean_box(pd.read_csv(path, dtype={"game_id": str, "player_id": str,
                                                "name": str},
                                   keep_default_na=False, parse_dates=["date"]))
    else:
        df = pd.DataFrame(columns=BOX_COLUMNS)
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    today = pd.Timestamp(today)
    new = []
    for d in season_dates(y):
        ds = d.strftime("%Y-%m-%d")
        if d >= today or ds in done:
            continue
        rows, _, pending = box_rows_for_date(d, sleep)
        have = set(df["game_id"]) if len(df) else set()
        new += [r for r in rows if r["game_id"] not in have]
        if not pending:
            done.add(ds)
    if new:
        df = clean_box(pd.concat([d for d in (df, pd.DataFrame(new)) if len(d)],
                                 ignore_index=True))
    os.makedirs(data_dir, exist_ok=True)
    df[BOX_COLUMNS].sort_values(["date", "game_id", "team", "player_id"]) \
        .to_csv(path, index=False)
    open(done_path, "w").write("\n".join(sorted(done)) + "\n")
    return df


def od_present(report_rows, box, gid, teams, slate):
    """{(gid, player_id): 0/1} from report rows for `teams` on `slate`: Out or
    Doubtful -> 0, any other listed status -> 1 (the od rule)."""
    names = {}
    for r in box.drop_duplicates(["team", "player_id"]).itertuples(index=False):
        names.setdefault((r.team, norm_name(r.name)), r.player_id)
    out = {}
    for r in report_rows.itertuples(index=False):
        code = team_code(r.team)
        if (code not in teams or report_date(r.game_date) != slate
                or r.status == NOT_SUBMITTED):
            continue
        pid = names.get((code, norm_name(report_name_to_first_last(r.player))))
        if pid is not None:
            out[(gid, pid)] = 0.0 if r.status in ("Out", "Doubtful") else 1.0
    return out


def upcoming_terms(box, team, gid, date, value, present):
    """(av_min, av_oo, av_bpm) for `team` in a game not yet played: the same
    team_availability computation, with the game appended as a placeholder."""
    tg = box[box["team"] == team]
    if not len(tg):
        return None
    stub = pd.DataFrame([dict(game_id=gid, date=pd.Timestamp(date), team=team,
                              opp="", home=True, margin=0.0,
                              player_id="__pregame__", name="", minutes=0.0,
                              pm=0.0)])
    return team_availability(pd.concat([tg[BOX_COLUMNS], stub], ignore_index=True),
                             value=value, present=present)[gid]


class LiveAvailability:
    """Availability terms and roster talent for today's slate (daily build,
    models v3-v5). A failed injury-report fetch leaves report None (no
    availability terms) but keeps talent."""

    def __init__(self, season, today, now_et, data_dir="data",
                 report_dir=os.path.join("bbr_cache", "injury_reports")):
        self.box = update_box(season, today, data_dir)
        bpm = load_bpm(season - 1)
        self.value, _, self.bpm_minutes = player_values(self.box, bpm)
        self.talent = talent_fn(self.box, self.value, arrival_roles(self.box, bpm))
        self.report_time, self.report = None, None
        try:
            archive = ReportArchive(report_dir, now=now_et)
            self.report_time, path = archive.latest_before(now_et)
            archive.save()
            self.report = parse_report(path) if path else None
        except Exception as e:  # noqa: BLE001 - talent still usable
            print(f"injury report unavailable ({e!r})", flush=True)

    def terms(self, gid, home, away, date):
        """{'av_min', 'av_bpm', 'report'} (home - away), or None -- the base
        model -- when the report does not cover this game (none, empty or
        unparsed, not on it, a team not yet submitted) or a team has no box
        history."""
        if self.report is None:
            return None
        slate = pd.Timestamp(date).strftime("%Y-%m-%d")
        if not covers(report_coverage(self.report, slate), home, away):
            return None
        present = od_present(self.report, self.box, gid, {home, away}, slate)
        th = upcoming_terms(self.box, home, gid, date, self.value, present)
        ta = upcoming_terms(self.box, away, gid, date, self.value, present)
        if th is None or ta is None:
            return None
        return dict(av_min=th[0] - ta[0], av_bpm=th[2] - ta[2],
                    report=f"{self.report_time:%Y-%m-%d %H:%M}")


def season_terms(t, archive, cache_dir=DEFAULT_CACHE, lead_minutes=LEAD_MINUTES):
    """Pregame od availability terms (home - away) for every game of season t,
    from box scores before each game and the NBA report at least
    `lead_minutes` before tip. Independent of the composite weights.
    Only games the report covers get terms; the rest fall back to the base
    model (the same routing as LiveAvailability.terms)."""
    box = fetch_box(t, cache_dir=cache_dir)
    value, _, _ = player_values(box, load_bpm(t - 1))
    st, s = game_statuses(box, fetch_tips(t, cache_dir), archive, lead_minutes)
    archive.save()
    av = game_availability(box, value, present=present_map(st, "od"))
    av = av[av["game_id"].isin(s["covered"])]
    return av[["slate_date", "home", "away", "av_min", "av_bpm"]], s


def with_terms(games, terms):
    """nc.build_games rows joined to season_terms rows (inner)."""
    g = games.copy()
    g["slate_date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    return g.merge(terms, on=["slate_date", "home", "away"], how="inner")


FIXED_V5 = ["luck_def", "talent_diff"]


def fit_fixed(G, base, fixed=FIXED_V5, features=None):
    """The v5 availability logit with the `fixed` coefficients taken from
    `base` (the v5 base logit, fit on the long game-log history) and only
    the other features refit on G (the few report seasons), with the fixed
    terms as an offset. Returns an ordinary model over `features` (default
    FEATURES_V5), so it scores like any availability logit."""
    features = list(features or FEATURES_V5)
    c = dict(zip(base["features"], base["coef"]))
    free = [f for f in features if f not in fixed]
    off = sum(c[f] * G[f].to_numpy(float) for f in fixed)
    m = nc.fit_logit(G[free].to_numpy(float), G["win"], free, offset=off)
    fm = dict(zip(free, m["coef"]))
    fm.update({f: c[f] for f in fixed})
    return {"features": features, "intercept": m["intercept"],
            "coef": [float(fm[f]) for f in features], "fixed": list(fixed)}


def fit(years, cache_dir=DEFAULT_CACHE, lead_minutes=LEAD_MINUTES, phase=True,
        v5=True, v6=True):
    """Fit model/logit_avail.json on games 10+ of `years` (see module doc).
    phase=True (v4) adds delta * season phase to the v3 features; v5=True
    also adds luck_def and talent_diff (needs the seasons' box scores);
    v6=True (with v5) also adds ft_diff, the own FT% gap."""
    v6 = v6 and v5
    weights = nc.load_json("weights.json")
    archive = ReportArchive(os.path.join(cache_dir, "injury_reports"))
    frames = []
    try:
        for t in years:
            terms, s = season_terms(t, archive, cache_dir, lead_minutes)
            g = with_terms(nc.build_games(
                t, weights, talent=season_talent(t, cache_dir) if v5 else None), terms)
            print(f"season {t}: {len(g)} games 10+ with terms; "
                  f"{s['with_report']}/{s['games']} games with a report, "
                  f"{len(s['covered'])} covered",
                  flush=True)
            frames.append(g)
    finally:
        archive.save()
    G = pd.concat(frames, ignore_index=True)
    feats = (FEATURES_V6 if v6 else FEATURES_V5 if v5 else
             FEATURES_V4 if phase else FEATURES)
    G = G[np.isfinite(G[feats].to_numpy(float)).all(axis=1)]
    m = nc.fit_logit(G[feats].to_numpy(float), G["win"], feats)
    if phase or v5:
        m["season_days"] = nc.SEASON_DAYS
    m.update({"half_life": nc.HALF_LIFE, "n_games": int(len(G)),
              "years": list(years), "lead_minutes": lead_minutes, "rule": "od",
              "replacement_bpm": REPLACEMENT_BPM, "bpm_shrink_mp": BPM_SHRINK_MP})
    nc.save_json(m, MODEL_FILE)
    coefs = ", ".join(f"{f}={c:+.4f}" for f, c in zip(m["features"], m["coef"]))
    print(f"saved model/{MODEL_FILE}  intercept={m['intercept']:.3f}  {coefs}  "
          f"n={m['n_games']}")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--years", nargs="+", default=["2023-2026"])
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--lead-minutes", type=int, default=LEAD_MINUTES)
    ap.add_argument("--no-phase", action="store_true",
                    help="fit the v3 features (no season-phase term)")
    ap.add_argument("--v4", action="store_true",
                    help="fit the v4 features (no luck_def / talent_diff)")
    ap.add_argument("--v5", action="store_true",
                    help="fit the v5 features (no ft_diff)")
    a = ap.parse_args(argv)
    fit(nc.parse_years(a.years), a.cache, a.lead_minutes, phase=not a.no_phase,
        v5=not (a.v4 or a.no_phase), v6=not (a.v5 or a.v4 or a.no_phase))
    return 0


if __name__ == "__main__":
    sys.exit(main())
