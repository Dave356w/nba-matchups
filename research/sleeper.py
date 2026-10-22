#!/usr/bin/env python3
"""Sleeper NBA feed: projected minutes, depth chart and injury designations.

  python research/sleeper.py history --seasons 2023 2024 2025 2026 [--refresh]
  python research/sleeper.py snapshot
  python research/sleeper.py check --season 2027 --box data/nba_box_2027.csv

Research data only: the build, the model and the ledger never read it, and
it writes only under research/output/sleeper/. Seasons are this repo's (end
year: 2026 = 2025-26); Sleeper labels a season by its start year
(`sleeper_season`). Teams are Basketball-Reference codes.

Sleeper's public API (https://docs.sleeper.com) is read-only and needs no key.
Three of its endpoints are read:

``/v1/state/nba``
    The current season, week and season start date.
``/v1/players/nba``
    One ~2.5MB dump of every player with his team, depth-chart slot (PG..C)
    and order (1 = starter) and injury designation (Out, DTD, GTD, IR) as of
    now. Sleeper asks callers to pull it at most once a day, so it is cached
    on disk for `PLAYERS_MAX_AGE_HOURS`.
``api.sleeper.com/projections/nba/{season}/{week}``
    One row per player per game (date, team, opponent, Sleeper game id) with
    a projected stat line from RotoWire; ``sp`` is projected seconds, in
    whole minutes. A listed player with an empty line is not expected to
    play (minutes 0). Undocumented; the dated list shape is the only one
    read (the v1 weekly endpoint has no dates or games).

What the archive is (checked 2026-10-22):
  depth     2019-20 on. Team-games sum to ~240 projected minutes from
            2022-23 (mean 237-242); before that they fall short (206-214),
            so `FULL_FROM_SEASON` = 2023.
  timing    Every row of a week carries one ``last_modified``, stamped after
            the week's last game (2025-26 week 10: 2025-12-29 07:40Z), so
            the archive cannot show when a value was set. Treat it as
            HINDSIGHT, like the reconstructed rows. Against the ESPN box it
            does not look edited after the game (`check`): 2025-26, 1,231
            games, 99.4% of box rows listed, minutes played corr 0.82 (MAE
            4.7); 911 players who did not play kept a projection (mean 10
            min, 72 at 20+) and 2,781 listed at 0 did play (mean 7 min, 140
            at 20+, 3.3% of minutes). This season's first two dates: corr
            0.78, 25 of 27 listed DNPs projected 4-20 min, and no player
            ESPN listed Out before tip (data/nba_injuries.csv) had minutes.
            Consistent with a pregame value; how long before tip is unknown.
  pregame   `snapshot` stamps each fetch (``fetched_utc``) and `pregame`
            keeps the last one before the ledger's tip, the same lock as the
            injury snapshots. That is the clean version.
  players   The ``player`` object inside each archived row is today's (team,
            injury status), so only its name is read. The dump's depth chart
            and injury status describe now and are joined to live snapshots
            only.

The functions below are pure transforms on the fetched JSON, so the rules can
be tested without the network.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market  # noqa: E402
import player_availability as pav  # noqa: E402

BASE_URL = "https://api.sleeper.app/v1"
API_URL = "https://api.sleeper.com"
CACHE_DIR = Path(__file__).resolve().parent / "output" / "sleeper"
PLAYERS_MAX_AGE_HOURS = 24.0
# Sleeper blocks an IP above roughly 1,000 calls a minute. A season is ~26
# calls, but the floor keeps any loop that grows later well clear of that.
MIN_CALL_INTERVAL_SECONDS = 0.1
FIRST_SEASON = 2020        # 2019-20, the first season Sleeper archives
FULL_FROM_SEASON = 2023    # first season whose team-games sum to ~240 minutes
MAX_WEEKS = 30             # a season is ~25 weeks (All-Star break folded in)
SEASON_TYPE = "regular"
STATS = ("pts", "reb", "ast", "stl", "blk", "to", "fga", "fgm", "fta", "ftm",
         "tpa", "tpm")
DEPTH_SLOTS = ("PG", "SG", "SF", "PF", "C")
ET = ZoneInfo("America/New_York")

PROJECTION_COLUMNS = (["sleeper_id", "name", "key", "date", "team", "opp",
                       "sleeper_game_id", "minutes", "projected"] + list(STATS)
                      + ["company", "last_modified_utc"])
PLAYER_COLUMNS = ["sleeper_id", "name", "key", "team", "position",
                  "fantasy_positions", "status", "depth_slot", "depth_order",
                  "injury_status", "injury_body_part", "injury_notes",
                  "news_updated_utc"]

_last_call = [0.0]


class SleeperError(RuntimeError):
    """A Sleeper request failed or returned something unusable."""


def sleeper_season(y) -> int:
    """This repo's season (end year, 2026 = 2025-26) -> Sleeper's (2025)."""
    return int(y) - 1


def normalize_team(value) -> str:
    """Sleeper team code -> BBR code ('BKN' -> 'BRK'), or '' if not a team."""
    team = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    team = team.strip().upper()
    team = pav.NBA_TO_BBR.get(team, team)
    return team if team in market.BBR_TEAMS else ""


def _utc(ms):
    """Sleeper's epoch milliseconds -> 'YYYY-MM-DDTHH:MM:SSZ', or ''."""
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(ms) or ms <= 0:
        return ""
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _full_name(p) -> str:
    p = p if isinstance(p, dict) else {}
    return (p.get("full_name")
            or f"{p.get('first_name') or ''} {p.get('last_name') or ''}").strip()


# --------------------------------------------------------------------- fetching
def get_json(url: str, timeout: float = 30, attempts: int = 3):
    """GET one Sleeper URL with bounded retries and a call-rate floor."""
    last_error = None
    for attempt in range(1, attempts + 1):
        wait = MIN_CALL_INTERVAL_SECONDS - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()
        try:
            request = Request(url, headers=market.HEADERS)
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(1.0 * attempt)
    raise SleeperError(f"Sleeper {url} failed after {attempts} attempts: {last_error}")


def fetch_state(timeout: float = 30) -> dict:
    state = get_json(BASE_URL + "/state/nba", timeout)
    if not isinstance(state, dict) or "season" not in state or "week" not in state:
        raise SleeperError(f"Sleeper /state/nba returned an unexpected shape: {state!r:.200}")
    return state


def fetch_players(cache_dir: str | Path | None = CACHE_DIR, timeout: float = 60,
                  max_age_hours: float = PLAYERS_MAX_AGE_HOURS, now=None) -> tuple[dict, str]:
    """Return the full player dump and the UTC time it was fetched.

    A cached copy younger than `max_age_hours` is reused, which keeps the dump
    to one pull a day however often a snapshot runs.
    """
    now = now or datetime.now(timezone.utc)
    path = Path(cache_dir) / "players_nba.json" if cache_dir else None
    if path is not None and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(cached["fetched_utc"])
            if (now - fetched).total_seconds() < max_age_hours * 3600 and cached.get("players"):
                return cached["players"], cached["fetched_utc"]
        except (OSError, ValueError, KeyError, TypeError):
            pass  # a corrupt cache is refetched, never trusted
    players = get_json(BASE_URL + "/players/nba", timeout)
    if not isinstance(players, dict) or not players:
        raise SleeperError("Sleeper /players/nba returned no players")
    fetched_utc = now.isoformat()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_utc": fetched_utc, "players": players}),
                        encoding="utf-8")
    return players, fetched_utc


def fetch_projections(season, week, timeout: float = 60):
    """One week of dated projections; `season` is Sleeper's (start year)."""
    return get_json(f"{API_URL}/projections/nba/{int(season)}/{int(week)}"
                    f"?season_type={SEASON_TYPE}", timeout)


# ------------------------------------------------------------------ transforms
def projection_frame(raw) -> pd.DataFrame:
    """Dated projection rows -> one row per player per game.

    ``minutes`` is ``sp`` / 60; a listed player whose line is empty (or has
    no ``sp``) is kept with minutes 0 and ``projected`` False, since Sleeper
    lists players it expects to sit. A row without a date or two NBA teams
    has no game and is skipped. The row's own team is used, not the embedded
    player's, which is his team today.
    """
    if not isinstance(raw, list):
        raise SleeperError(f"Unexpected Sleeper projection shape: {type(raw).__name__}")
    rows = []
    for row in raw:
        if not isinstance(row, dict) or row.get("player_id") is None:
            continue
        team, opp = normalize_team(row.get("team")), normalize_team(row.get("opponent"))
        date = str(row.get("date") or "")[:10]
        if not (team and opp and date):
            continue
        stats = row.get("stats") if isinstance(row.get("stats"), dict) else {}
        seconds = pd.to_numeric(stats.get("sp"), errors="coerce")
        minutes = float(seconds) / 60 if pd.notna(seconds) and seconds > 0 else 0.0
        name = _full_name(row.get("player"))
        out = {"sleeper_id": str(row["player_id"]), "name": name,
               "key": pav.norm_name(name), "date": date, "team": team, "opp": opp,
               "sleeper_game_id": str(row.get("game_id") or ""),
               "minutes": round(minutes, 2), "projected": minutes > 0}
        for s in STATS:
            v = pd.to_numeric(stats.get(s), errors="coerce")
            out[s] = float(v) if pd.notna(v) else np.nan
        out["company"] = str(row.get("company") or "")
        out["last_modified_utc"] = _utc(row.get("last_modified"))
        rows.append(out)
    frame = pd.DataFrame(rows, columns=PROJECTION_COLUMNS)
    return frame.drop_duplicates(["sleeper_id", "date", "sleeper_game_id"], keep="last")


def players_frame(raw: dict) -> pd.DataFrame:
    """Flatten the player dump: active players at a basketball position.

    ``team`` is '' for a free agent. Depth and injury fields are as of the
    fetch, so they belong with a live snapshot, never with archived games.
    """
    rows = []
    for pid, p in (raw or {}).items():
        if not isinstance(p, dict) or not p.get("active"):
            continue
        position = str(p.get("position") or "").upper()
        if position not in DEPTH_SLOTS:
            continue          # Sleeper also lists 30 team "DEF" entries
        name = _full_name(p)
        slot = str(p.get("depth_chart_position") or "").upper()
        order = pd.to_numeric(p.get("depth_chart_order"), errors="coerce")
        rows.append({
            "sleeper_id": str(p.get("player_id") or pid),
            "name": name,
            "key": pav.norm_name(name),
            "team": normalize_team(p.get("team")),
            "position": position,
            "fantasy_positions": "/".join(p.get("fantasy_positions") or []),
            "status": str(p.get("status") or ""),
            "depth_slot": slot if slot in DEPTH_SLOTS else "",
            "depth_order": int(order) if pd.notna(order) else pd.NA,
            "injury_status": str(p.get("injury_status") or ""),
            "injury_body_part": str(p.get("injury_body_part") or ""),
            "injury_notes": str(p.get("injury_notes") or ""),
            "news_updated_utc": _utc(p.get("news_updated")),
        })
    frame = pd.DataFrame(rows, columns=PLAYER_COLUMNS)
    frame["depth_order"] = frame["depth_order"].astype("Int64")
    return frame


def build_reference(players: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    """Attach the dump's depth chart and injury status to each projection row.

    ``starter`` is order 1 in a depth slot (five a team). ``minutes_rank`` is
    the rank by projected minutes within the team's game, ties broken by
    chart order and name; ``minutes_share`` is minutes over the team's total.
    Injuries are the projection's call: a player RotoWire does not expect to
    play has minutes 0 and his minutes are on his teammates, so there is no
    separate injury filter, and the designation is carried for display.
    """
    keep = [c for c in PLAYER_COLUMNS if c not in ("name", "key", "team")]
    ref = projections.merge(players[keep], on="sleeper_id", how="left")
    ref["starter"] = ref["depth_order"].eq(1).fillna(False).astype(bool)
    game = ["date", "team", "sleeper_game_id"]
    work = ref.assign(_order=pd.to_numeric(ref["depth_order"], errors="coerce")
                      .astype(float).fillna(np.inf))
    order = work.sort_values(game + ["minutes", "_order", "key"],
                             ascending=[True, True, True, False, True, True])
    ref["minutes_rank"] = (order.groupby(game).cumcount() + 1).reindex(ref.index).astype("Int64")
    ref["team_minutes"] = ref.groupby(game)["minutes"].transform("sum")
    ref["minutes_share"] = np.where(ref["team_minutes"] > 0,
                                    ref["minutes"] / ref["team_minutes"].where(
                                        ref["team_minutes"] > 0), 0.0)
    return ref


def match_espn(frame: pd.DataFrame, espn: pd.DataFrame) -> pd.Series:
    """Map each Sleeper row to an ESPN athlete id, or NA where none is certain.

    `espn` has ESPN ``player_id``, ``name`` and ``team`` (the box score or the
    injury snapshots). Sleeper's NBA dump carries no ESPN id, so names are
    matched: name + team, then name alone (a player traded since the ESPN
    rows). A step only accepts a key unique on the ESPN side, so an
    ambiguous name never matches.
    """
    ids = pd.Series(pd.NA, index=frame.index, dtype="object")
    ref = espn.assign(key=espn["name"].map(pav.norm_name),
                      player_id=espn["player_id"].astype(str))
    ref = ref[ref["key"].ne("") & ref["player_id"].ne("")].drop_duplicates(
        ["player_id", "key", "team"])

    def unique(columns):
        per = ref.drop_duplicates(["player_id"] + columns)
        counts = per.groupby(columns)["player_id"].transform("size")
        return per[counts.eq(1)].set_index(columns)["player_id"]

    by_team = unique(["key", "team"])
    ids[:] = [by_team.get((k, t), pd.NA) for k, t in zip(frame["key"], frame["team"])]
    by_key = unique(["key"])
    want = ids.isna()
    ids[want] = frame.loc[want, "key"].map(by_key)
    return ids


def pregame(snaps: pd.DataFrame, tips: pd.DataFrame) -> pd.DataFrame:
    """The last snapshot of each player's game taken strictly before tip.

    `tips` is the ledger (``slate_date``, ``home``, ``away``, ``tip_utc``).
    A row whose game has no tip, or whose every snapshot is at or after tip,
    is dropped: nothing after tip is ever read.
    """
    sides = pd.concat([tips[["slate_date", "home", "tip_utc"]].rename(columns={"home": "team"}),
                       tips[["slate_date", "away", "tip_utc"]].rename(columns={"away": "team"})])
    sides = sides.rename(columns={"slate_date": "date"}).drop_duplicates(["date", "team"])
    m = snaps.merge(sides, on=["date", "team"], how="inner")
    fetched = pd.to_datetime(m["fetched_utc"], utc=True, format="mixed")
    tip = pd.to_datetime(m["tip_utc"], utc=True, format="mixed")
    m = m[fetched < tip].assign(_f=fetched[fetched < tip])
    m = m.sort_values("_f").drop_duplicates(["date", "team", "sleeper_id"], keep="last")
    return m.drop(columns="_f").reset_index(drop=True)


# ------------------------------------------------------------------- pulling
def fetch_season(y, cache_dir: str | Path | None = CACHE_DIR, refresh: bool = False,
                 fetch=None, completed: bool = True) -> pd.DataFrame:
    """Every archived game of season `y` (this repo's), one row per player.

    Weeks are read from 1 until two empty weeks follow data (or MAX_WEEKS).
    A completed season is cached as proj_<y>.csv; the current season is
    refetched each time, since its weeks are still being written.
    """
    fetch = fetch or fetch_projections  # looked up at call time, so tests can patch it
    path = Path(cache_dir) / f"proj_{int(y)}.csv" if cache_dir else None
    if path is not None and path.exists() and completed and not refresh:
        return pd.read_csv(path, dtype={"sleeper_id": str, "sleeper_game_id": str})
    frames, empty, started = [], 0, False
    for week in range(1, MAX_WEEKS + 1):
        frame = projection_frame(fetch(sleeper_season(y), week))
        if frame.empty:
            empty += 1
            if started and empty >= 2:
                break
            continue
        started, empty = True, 0
        frames.append(frame.assign(week=week))
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=PROJECTION_COLUMNS + ["week"]))
    if path is not None and completed and len(out):
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return out


def team_minutes(proj: pd.DataFrame) -> pd.Series:
    """Projected minutes per team-game; ~240 when the archive is complete."""
    return proj.groupby(["date", "team", "sleeper_game_id"])["minutes"].sum()


def snapshot(cache_dir: str | Path = CACHE_DIR, now=None, timeout: float = 60):
    """Today's and later projections with depth and injury status, stamped.

    Reads the current week and the next (a week turns over on Monday) and
    keeps games on or after today's ET date. Appended to snapshots.csv; read
    it through `pregame` so only snapshots before tip count.
    """
    now = now or datetime.now(timezone.utc)
    state = fetch_state(timeout)
    raw_players, players_utc = fetch_players(cache_dir, timeout, now=now)
    players = players_frame(raw_players)
    season, week = int(state["season"]), int(state.get("week") or 1)
    frames = [projection_frame(fetch_projections(season, w, timeout)) for w in (week, week + 1)]
    proj = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["sleeper_id", "date", "sleeper_game_id"], keep="last")
    today = now.astimezone(ET).strftime("%Y-%m-%d")
    proj = proj[proj["date"] >= today]
    ref = build_reference(players, proj).assign(
        fetched_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"), players_fetched_utc=players_utc)
    path = Path(cache_dir) / "snapshots.csv"
    if path.exists():
        old = pd.read_csv(path, dtype={"sleeper_id": str, "sleeper_game_id": str})
        ref = pd.concat([old, ref], ignore_index=True)
    ref = ref.drop_duplicates(["fetched_utc", "sleeper_id", "sleeper_game_id"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(path, index=False)
    return ref, {"season": season, "week": week, "today": today}


# ------------------------------------------------------------------- readouts
def check(proj: pd.DataFrame, box: pd.DataFrame) -> dict:
    """Archived projected minutes against the box score on the same dates.

    The box (player_availability BOX_COLUMNS) lists who dressed, 0 minutes
    for a DNP. Returns coverage and fit figures; a hindsight archive would
    show DNPs zeroed and minutes near the box, a pregame one neither.
    """
    proj = proj[proj["date"].isin(set(box["date"].astype(str)))].copy()
    box = box.assign(date=box["date"].astype(str))
    proj["espn_id"] = match_espn(proj, box)
    m = box.assign(espn_id=box["player_id"].astype(str)).merge(
        proj[["date", "espn_id", "minutes", "projected"]].dropna(subset=["espn_id"]),
        on=["date", "espn_id"], how="left")
    listed = m["minutes_y"].notna()
    played = m["minutes_x"] > 0
    both = m[listed & played]
    tm = team_minutes(proj)
    return {
        "dates": int(proj["date"].nunique()), "box_rows": int(len(m)),
        "box_rows_listed": int(listed.sum()),
        "played_minutes_listed": float(m.loc[played & listed, "minutes_x"].sum()
                                       / max(m.loc[played, "minutes_x"].sum(), 1e-9)),
        "unmatched_projected": int(proj["espn_id"].isna().sum()),
        "team_minutes_mean": float(tm.mean()) if len(tm) else np.nan,
        "team_minutes_sd": float(tm.std()) if len(tm) > 1 else np.nan,
        "corr_played": float(both["minutes_x"].corr(both["minutes_y"])) if len(both) > 2 else np.nan,
        "mae_played": float((both["minutes_x"] - both["minutes_y"]).abs().mean()) if len(both) else np.nan,
        "dnp_projected": int((listed & ~played & m["projected"].eq(True)).sum()),
        "dnp_listed": int((listed & ~played).sum()),
        "played_unprojected": int((listed & played & m["projected"].eq(False)).sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("history", help="pull the archive (hindsight) for seasons")
    h.add_argument("--seasons", type=int, nargs="+", required=True,
                   help="this repo's seasons (end year: 2026 = 2025-26)")
    h.add_argument("--refresh", action="store_true")
    sub.add_parser("snapshot", help="stamp today's projections (pregame)")
    c = sub.add_parser("check", help="archived minutes vs a box score")
    c.add_argument("--season", type=int, required=True)
    c.add_argument("--box", required=True)
    args = ap.parse_args(argv)

    if args.cmd != "snapshot":
        current = int(fetch_state()["season"]) + 1     # this repo's season
    if args.cmd == "history":
        for y in args.seasons:
            if y < FIRST_SEASON:
                print(f"{y}: before {FIRST_SEASON}, Sleeper has no archive")
                continue
            proj = fetch_season(y, refresh=args.refresh, completed=y < current)
            tm = team_minutes(proj)
            flag = "" if y >= FULL_FROM_SEASON else "  (incomplete: below 240)"
            print(f"{y}: {len(proj)} rows, {proj['date'].nunique()} dates, "
                  f"{len(tm)} team-games, minutes {tm.mean():.1f} ± {tm.std():.1f}{flag}")
    elif args.cmd == "snapshot":
        ref, ctx = snapshot()
        print(f"snapshot {ctx}: {len(ref)} rows in {CACHE_DIR / 'snapshots.csv'}")
    else:
        proj = fetch_season(args.season, completed=args.season < current)
        out = check(proj, pd.read_csv(args.box, dtype={"player_id": str, "game_id": str}))
        for k, v in out.items():
            print(f"{k:>24}: {v:.3f}" if isinstance(v, float) else f"{k:>24}: {v}")


if __name__ == "__main__":
    main()
