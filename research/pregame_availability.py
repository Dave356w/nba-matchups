#!/usr/bin/env python3
"""Pregame availability: the injury report as it stood before tip.

  python research/pregame_availability.py --seasons 2025 2026 \
      [--box-seasons 2024 2025 2026] [--lead-minutes 30]

research/availability.py measured a HINDSIGHT ceiling (who actually played).
This replaces "played" with what was knowable before the game: the NBA's
official injury report (ak-static.cms.nba.com/referee/injury/...), the last
edition published at least --lead-minutes before each game's tip. Same v2
player values (last-season BBR BPM), same rating-window logic.

Per player with history for the team, P(plays today) is
  od arm   Out or Doubtful -> 0; anyone else on the roster -> 1
  q arm    Out -> 0; Doubtful / Questionable / Probable -> the share of
           players with that status who played in the TRAINING seasons;
           Available or not listed (on the roster) -> 1
where "on the roster" = listed on the team's previous box score (known
before the game). Players not on the roster and not on the report -> 0.

Walk-forward, games 10+: weights on WEIGHT_YEARS < Y; every logit and the
q-arm play rates fitted on box seasons < Y. Reported per test season and
closing book on identical games: base / hindsight v2 / od / q against the
market and each other, each arm's per-point coefficient against theory,
and the share of the market-minus-base gap explained. Also prints how often
players listed Out / Doubtful / Questionable / Probable actually played.

Research only: no change to the model, the ledger or MODEL_TAG. Reports,
tip times and the parsed report rows are cached under research/output/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import availability as av  # noqa: E402
import backfill_history as bf  # noqa: E402
import build_site  # noqa: E402
import ledger  # noqa: E402
import market  # noqa: E402
import nba_composite as nc  # noqa: E402

ET = ZoneInfo("America/New_York")
REPORT_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{slot}.pdf"
REPORT_DIR = os.path.join(av.OUT_DIR, "injury_reports")
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")
SLOT_MINUTES = 15
MAX_LOOKBACK_H = 12
ARMS = ("od", "q")


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

    def __init__(self, cache_dir=REPORT_DIR, fetch=None, sleep=0.05):
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
PARSER_VERSION = 3          # bump to invalidate cached parsed-report CSVs


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
    change. Returns [{game_date, matchup, team, player, status}].
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
    return rows


_DATE = re.compile(r"\b(\d{2}/\d{2}/\d{2,4})\b")
_TIME = re.compile(r"\d{1,2}:\d{2}\s*\(ET\)")
_MATCHUP = re.compile(r"\b([A-Z]{2,3}@[A-Z]{2,3})\b")
_ROW = re.compile(r"^\s*(?P<player>[A-Z][^,]{1,40},\s*[A-Za-z.'\-]+?)\s*"
                  r"(?P<status>Out|Doubtful|Questionable|Probable|Available)"
                  r"(?![a-z])")


def full_team_names():
    out = {}
    for code, nick in build_site.TEAM_NAMES.items():
        city = CITIES[code]
        out[code] = city if city.endswith(nick) else f"{city} {nick}"
    return out


def rows_from_text(pages_text):
    """Fallback: plain-text lines -> report rows, by regular expressions.

    Strips date, time and matchup (carried down), then a full team name
    (spaced or not; carried down), then reads 'Last, First' and the status.
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
    for code, nick in build_site.TEAM_NAMES.items():
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


# ---------------------------------------------------------------- tips ----
def fetch_tips(y, cache_dir=av.OUT_DIR):
    """game_id -> tip_utc for every regular-season game of season y."""
    path = os.path.join(cache_dir, f"tips_{y}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        return dict(zip(df["game_id"], df["tip_utc"]))
    rows = []
    for d in av.season_dates(y):
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
    team, player_id, status, played, report time, lead (minutes before tip)."""
    games = box.drop_duplicates(["game_id", "team"])[["game_id", "date", "team"]]
    names = {}                                  # (team, norm name) -> player_id
    for r in box.drop_duplicates(["team", "player_id"]).itertuples(index=False):
        names.setdefault((r.team, av.norm_name(r.name)), r.player_id)
    played = {(r.game_id, r.player_id): r.minutes > 0
              for r in box.itertuples(index=False)}
    out, stats = [], dict(games=0, with_report=0, listed=0, matched=0, lags=[],
                          reports=0, reports_rows=0)
    parsed = {}
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
        for r in rep.itertuples(index=False):
            code = team_code(r.team)
            if code not in teams or report_date(r.game_date) != slate:
                continue
            stats["listed"] += 1
            pid = names.get((code, av.norm_name(report_name_to_first_last(r.player))))
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


# ------------------------------------------------------------ evaluation ---
def arm_frame(box, value, present):
    g = av.game_availability(box, value, present=present)
    return g[["slate_date", "home", "away", "av_min", "av_bpm"]]


def season_frame(t, weights, feats):
    """nc.build_games (games 10+) for season t joined to every arm's terms."""
    g = nc.build_games(t, weights)
    g["slate_date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    return g.merge(feats, on=["slate_date", "home", "away"], how="inner")


def arm_features(name):
    return ["delta", "b2b_net", f"{name}_min", f"{name}_bpm"]


def report(m, arms):
    lines = []
    for (season, book), g in m.groupby(["year", "close_book"]):
        y = g["home_won"].to_numpy(float)
        q = g["close_q_home"].to_numpy(float)
        base = g["p_base"].to_numpy(float)
        lines.append(f"\n{season} {market.BOOK_NAMES.get(book, book)} close · "
                     f"games 10+ (n={len(g)})")
        comps = [("base", "market", base, q)]
        for a in arms:
            comps += [(a, "market", g[f"p_{a}"].to_numpy(float), q),
                      (a, "base", g[f"p_{a}"].to_numpy(float), base)]
        comps += [("hind", "q", g["p_hind"].to_numpy(float), g["p_q"].to_numpy(float))]
        for na, nb, a, b in comps:
            s = av.paired(a, b, y)
            lines.append(
                f"  {na:5s} vs {nb:7s} logloss {s['ll_a']:.4f} vs {s['ll_b']:.4f}  "
                f"diff {s['d_ll']:+.4f} ± {1.96 * s['d_ll_se']:.4f}  "
                f"Brier diff {s['d_br']:+.4f} ± {1.96 * s['d_br_se']:.4f}")
        lines.append("  market-minus-base logit gap, R² from availability: " +
                     ", ".join(f"{a} {av.gap_explained(g, [f'{a}_min', f'{a}_bpm']):.3f}"
                               for a in arms))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--box-seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--lead-minutes", type=int, default=30,
                    help="use the last report at least this long before tip")
    a = ap.parse_args(argv)
    archive = ReportArchive()
    box, value, statuses, hind = {}, {}, {}, {}
    try:
        for t in a.box_seasons:
            box[t] = av.fetch_box(t)
            value[t], _, rate_min = av.player_values(box[t], av.load_bpm(t - 1))
            st, s = game_statuses(box[t], fetch_tips(t), archive, a.lead_minutes)
            archive.save()
            statuses[t] = st
            lag = np.median(s["lags"]) if s["lags"] else float("nan")
            print(f"season {t}: {s['with_report']}/{s['games']} team-games with a "
                  f"report (median {lag:.0f} min before tip); {s['reports_rows']}/"
                  f"{s['reports']} reports parsed to rows; {s['listed']} report "
                  f"rows, {100 * s['matched'] / max(s['listed'], 1):.1f}% matched to "
                  f"box players; BPM covers {100 * rate_min:.1f}% of minutes; "
                  f"{archive.requests} archive requests so far", flush=True)
            if s["with_report"] == 0:
                raise SystemExit(f"season {t}: no injury reports found; check "
                                 f"REPORT_URL/slot_names against the archive")
            hind[t] = arm_frame(box[t], value[t], None)
    finally:
        archive.save()

    print("\n== How often listed players played (all box seasons, matched rows)")
    roles = {t: av.arrival_roles(box[t], {}) for t in box}
    for t, st in statuses.items():
        d = dict(zip(box[t]["game_id"], box[t]["date"]))
        st["rotation"] = [roles[t](pid, d[gid]) >= 20 / 48
                          for gid, pid in zip(st["game_id"], st["player_id"])]
    allst = pd.concat(statuses.values(), ignore_index=True)
    for status in STATUSES:
        s = allst[allst["status"] == status]
        if not len(s):
            continue
        rot = s[s["rotation"]]
        per = ", ".join(f"{t}: {100 * st[st['status'] == status]['played'].mean():.0f}%"
                        for t, st in statuses.items()
                        if (st["status"] == status).any())
        print(f"  {status:12s} n={len(s):5d} played {100 * s['played'].mean():5.1f}%"
              f"  | 20+ mpg players n={len(rot):5d} played "
              f"{100 * rot['played'].mean() if len(rot) else float('nan'):5.1f}%  | {per}")

    recon = ledger.graded(ledger.load(ledger.RECON_PATH))
    recon = recon[["slate_date", "home", "away", "close_q_home", "close_book",
                   "home_won"]]
    allm = []
    for y in a.seasons:
        tr_years = av.before(a.box_seasons, y)
        if not tr_years:
            raise SystemExit(f"season {y}: no earlier box seasons to train on")
        rates = play_rates(pd.concat([statuses[t] for t in tr_years]))
        print(f"\n=== season {y}: logits and q-arm rates on {tr_years}; rates " +
              ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(rates.items())))
        weights = nc.fit_weights(av.before(bf.WEIGHT_YEARS, y))
        frames = {}
        for t in tr_years + [y]:
            f = hind[t].rename(columns={"av_min": "hind_min", "av_bpm": "hind_bpm"})
            for arm in ARMS:
                p = present_map(statuses[t], arm, rates)
                f = f.merge(arm_frame(box[t], value[t], p).rename(
                    columns={"av_min": f"{arm}_min", "av_bpm": f"{arm}_bpm"}),
                    on=["slate_date", "home", "away"])
            frames[t] = season_frame(t, weights, f)
        tr = pd.concat([frames[t] for t in tr_years], ignore_index=True)
        te = frames[y]
        base = nc.fit_logit(tr[nc.LOGIT_FEATURES].to_numpy(float), tr["win"],
                            nc.LOGIT_FEATURES)
        te["p_base"] = nc.predict(base, te[nc.LOGIT_FEATURES].to_numpy(float))
        for arm in ("hind",) + ARMS:
            feats = arm_features(arm)
            fm = nc.fit_logit(tr[feats].to_numpy(float), tr["win"], feats)
            te[f"p_{arm}"] = nc.predict(fm, te[feats].to_numpy(float))
            c = dict(zip(fm["features"], fm["coef"]))
            print(f"  {arm:5s} {arm}_bpm logit per point {c[f'{arm}_bpm']:+.4f} "
                  f"vs theory {av.THEORY_D:+.4f}   ({arm}_min {c[f'{arm}_min']:+.4f})")
        m = te.merge(recon, on=["slate_date", "home", "away"], how="inner")
        print(f"  test games 10+: {len(te)}; matched to reconstructed rows: {len(m)}")
        allm.append(m)
    m = pd.concat(allm, ignore_index=True)
    print("\n== Same games, one book at a time (negative diff = first is better). "
          "hind = who played (hindsight); od / q = pregame report")
    print(report(m, ("hind",) + ARMS))
    out = os.path.join(av.OUT_DIR, "pregame_availability.csv")
    m.to_csv(out, index=False)
    allst.to_csv(os.path.join(av.OUT_DIR, "report_statuses.csv"), index=False)
    print(f"\nwrote {out}: {len(m)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
